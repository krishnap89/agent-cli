"""Tests for the micro-agent execution loop (agent_cli/micro).

Each test drives a real MicroOrchestrator against a temp project, with the LLM
answered by the FakeServer and the Gradle/lint Checker replaced by a scripted
fake (the loop's control flow is what's under test, not Gradle). Covers:

- fast path: a request naming one file in the chat skips Locator and Planner
- fix loop success: a failing check is handed to the Fixer, then passes
- repeated-error stop: the same check error twice stops the step
- bad-format retry: a malformed role reply is retried once, then parsed
"""
import os
import tempfile
import unittest
from pathlib import Path

from agent_cli.config import Config
from agent_cli.coder import Coder
from agent_cli.micro import MicroOrchestrator, CheckResult, PlanStep
from tests.fake_server import FakeServer


def _edit(path, search, replace):
    """Build a SEARCH/REPLACE reply the Editor/Fixer roles produce."""
    return f"{path}\n```\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n```"


class FakeChecker:
    """Stands in for the Gradle/lint Checker: returns scripted CheckResults."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def check(self, files):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return CheckResult(ok=True)


class MicroTestBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(f"cd {self.root} && git init -q && git config user.email t@t && git config user.name t")

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def _coder(self, script):
        self.server = FakeServer(script)
        self.server.start()
        cfg = Config(base_url=self.server.base_url, workdir=self.root,
                     auto_approve=True, stream=False, use_map=False)
        return Coder(cfg, mode="code")

    def _roles_called(self):
        """System-prompt first lines for each request, to see which roles ran."""
        out = []
        for req in self.server.requests:
            sys_msg = req["messages"][0]["content"]
            for role in ("locate", "turn a code request", "applying one small",
                         "fix code that failed", "review a code change", "summarise"):
                if role in sys_msg.lower():
                    out.append(role)
                    break
        return out


class TestFastPath(MicroTestBase):
    def test_single_named_file_skips_locator_and_planner(self):
        (self.root / "calc.py").write_text("x = 1\n")
        # Editor, Reviewer, Summarizer. No Locator/Planner because the request
        # names the one file already in the chat.
        edit = _edit("calc.py", "x = 1", "x = 2")
        coder = self._coder([edit, "VERDICT: OK", "Changed x from 1 to 2."])
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        changed = orch.run("update calc.py to set x to 2")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 2\n")
        roles = self._roles_called()
        # First model call is the Editor, not a Locator or Planner.
        self.assertEqual(roles[0], "applying one small")
        self.assertNotIn("locate", roles)
        self.assertNotIn("turn a code request", roles)


class TestFixLoop(MicroTestBase):
    def test_failing_check_triggers_fixer_then_passes(self):
        (self.root / "calc.py").write_text("x = 1\n")
        edit = _edit("calc.py", "x = 1", "x = 2")
        fix = _edit("calc.py", "x = 2", "x = 3")
        # Editor, Fixer, Reviewer, Summarizer.
        coder = self._coder([edit, fix, "VERDICT: OK", "Set x to 3."])
        coder.add(["calc.py"])

        checker = FakeChecker([
            CheckResult(ok=False, signature="compile:boom", detail="boom"),  # after Editor
            CheckResult(ok=True),                                            # after Fixer
        ])
        orch = MicroOrchestrator(coder, checker=checker)
        changed = orch.run("fix calc.py")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 3\n")
        self.assertIn("fix code that failed", self._roles_called())  # Fixer ran
        self.assertEqual(checker.calls, 2)


class TestRepeatedError(MicroTestBase):
    def test_same_error_twice_stops_step(self):
        (self.root / "calc.py").write_text("x = 1\n")
        edit = _edit("calc.py", "x = 1", "x = 2")
        fix = _edit("calc.py", "x = 2", "x = 9")
        coder = self._coder([edit, fix, fix, fix])
        coder.add(["calc.py"])
        coder.cfg.max_fix_attempts = 5  # generous; the repeated-error rule should stop first

        same = CheckResult(ok=False, signature="compile:same", detail="same error")
        checker = FakeChecker([same, same, same, same])
        orch = MicroOrchestrator(coder, checker=checker)
        orch.run("fix calc.py")

        # Editor -> check(fail, record sig) -> Fixer -> check(same sig) -> STOP.
        # So exactly 2 checks and 1 Fixer call, not five rounds.
        self.assertEqual(checker.calls, 2)
        self.assertEqual(self._roles_called().count("fix code that failed"), 1)


class TestFailedApplyGoesToFixer(MicroTestBase):
    def test_editor_block_that_does_not_apply_is_handed_to_fixer(self):
        """A SEARCH that doesn't match the file must not pass vacuously.

        Reproduces the real-world bug where the Editor's block failed to apply,
        the Checker saw unchanged code and passed, and nothing was written.
        """
        (self.root / "calc.py").write_text("x = 1\n")
        bad = _edit("calc.py", "x = 999", "x = 2")   # SEARCH doesn't match -> fails to apply
        good = _edit("calc.py", "x = 1", "x = 2")    # Fixer gets it right
        coder = self._coder([bad, good, "VERDICT: OK", "Set x to 2."])
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        changed = orch.run("update calc.py to set x to 2")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 2\n")
        self.assertIn("fix code that failed", self._roles_called())  # Fixer ran on the apply failure

    def test_new_file_is_created(self):
        """An empty-SEARCH block creates a file the Locator added but was empty."""
        (self.root / "new.py").write_text("")  # added, but empty -> must be created
        create = "new.py\n```\n<<<<<<< SEARCH\n=======\nvalue = 42\n>>>>>>> REPLACE\n```"
        coder = self._coder([create, "VERDICT: OK", "Created new.py."])
        coder.add(["new.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        changed = orch.run("fill in new.py with value = 42")

        self.assertTrue(changed)
        self.assertEqual((self.root / "new.py").read_text(), "value = 42\n")


class TestScopeGuard(MicroTestBase):
    def test_block_for_off_limits_path_is_rejected_not_created(self):
        """A block aimed at a path outside the step's files must not create a file.

        Reproduces the bug where the model copied the 'path/to/file' placeholder;
        it should become an apply error the Fixer corrects, not a junk file.
        """
        (self.root / "calc.py").write_text("x = 1\n")
        # Editor aims at the placeholder path -> rejected.
        stray = _edit("path/to/file", "x = 1", "x = 2")
        # Fixer uses the real path.
        good = _edit("calc.py", "x = 1", "x = 2")
        coder = self._coder([stray, good, "VERDICT: OK", "Set x to 2."])
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        changed = orch.run("update calc.py to set x to 2")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 2\n")
        # The junk file must NOT have been created.
        self.assertFalse((self.root / "path/to/file").exists())
        self.assertIn("fix code that failed", self._roles_called())  # Fixer corrected it


class TestReviewLoop(MicroTestBase):
    def test_changes_verdict_feeds_back_to_fixer(self):
        """A 'compiles but wrong' step: Reviewer CHANGES -> Fixer -> OK."""
        (self.root / "calc.py").write_text("x = 1\n")
        edit = _edit("calc.py", "x = 1", "x = 2")      # compiles, but "wrong"
        fix = _edit("calc.py", "x = 2", "x = 42")       # Fixer corrects per review note
        coder = self._coder([
            edit,
            "VERDICT: CHANGES\nNOTES: should be 42, not 2",  # Reviewer, pass 1
            fix,                                             # Fixer (review-driven)
            "VERDICT: OK",                                   # Reviewer, pass 2
            "Set x to 42.",                                  # Summarizer
        ])
        coder.add(["calc.py"])
        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True), CheckResult(ok=True)]))
        changed = orch.run("set x in calc.py to 42")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 42\n")
        self.assertEqual(self._roles_called().count("review a code change"), 2)
        self.assertIn("fix code that failed", self._roles_called())

    def test_review_rounds_cap_keeps_edits(self):
        """If the Reviewer never relents, the run stops after AGENT_REVIEW_ROUNDS."""
        import os as _os
        (self.root / "calc.py").write_text("x = 1\n")
        edit = _edit("calc.py", "x = 1", "x = 2")
        fix = _edit("calc.py", "x = 2", "x = 3")
        coder = self._coder([
            edit,
            "VERDICT: CHANGES\nNOTES: nope",   # review pass 1
            fix,                                # fixer
            "VERDICT: CHANGES\nNOTES: still nope",  # review pass 2 (== cap with ROUNDS=1)
            "Changed x.",                       # summarizer
        ])
        coder.add(["calc.py"])
        _os.environ["AGENT_REVIEW_ROUNDS"] = "1"
        try:
            orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True), CheckResult(ok=True)]))
            changed = orch.run("set x in calc.py")
        finally:
            del _os.environ["AGENT_REVIEW_ROUNDS"]
        # Edits stay applied (not undone) even though the Reviewer never said OK.
        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 3\n")
        self.assertEqual(self._roles_called().count("review a code change"), 2)


class TestPlanValidation(MicroTestBase):
    def test_step_path_not_located_is_dropped(self):
        """A Planner step referencing a file the Locator didn't choose loses it."""
        (self.root / "a.py").write_text("v = 1\n")
        # Locator picks a.py; Planner references a.py AND a bogus ghost.py.
        plan = "STEP: bump v\nFILES: a.py, ghost.py"
        edit = _edit("a.py", "v = 1", "v = 2")
        coder = self._coder([
            "a.py",        # Locator
            plan,          # Planner (ghost.py should be dropped)
            edit,          # Editor
            "VERDICT: OK", # Reviewer
            "Bumped v.",   # Summarizer
        ])
        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch._repo_map_text = lambda request: "a.py"  # type: ignore

        changed = orch.run("bump the value")

        self.assertTrue(changed)
        self.assertEqual((self.root / "a.py").read_text(), "v = 2\n")
        self.assertFalse((self.root / "ghost.py").exists())  # never created

    def test_implausible_locator_path_is_ignored_but_plausible_new_kept(self):
        """Locator may name a new file in an existing dir; a junk path is dropped."""
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "existing.py").write_text("x = 0\n")
        # Locator names a plausible new file (pkg/new.py) and a junk path.
        create = "pkg/new.py\n```\n<<<<<<< SEARCH\n=======\ny = 1\n>>>>>>> REPLACE\n```"
        coder = self._coder([
            "pkg/new.py\nnot a real path",   # Locator: one plausible-new, one junk
            "STEP: create the file\nFILES: pkg/new.py",  # Planner
            create,                           # Editor creates it
            "VERDICT: OK",                    # Reviewer
            "Created pkg/new.py.",            # Summarizer
        ])
        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch._repo_map_text = lambda request: "pkg/existing.py"  # type: ignore

        changed = orch.run("add a new helper file")

        self.assertTrue(changed)
        self.assertEqual((self.root / "pkg" / "new.py").read_text(), "y = 1\n")
        # The junk path "not a real path" was never created anywhere.
        self.assertFalse((self.root / "not a real path").exists())


class TestBadFormatRetry(MicroTestBase):
    def test_locator_bad_format_retries_once(self):
        (self.root / "calc.py").write_text("x = 1\n")
        (self.root / "other.py").write_text("y = 1\n")
        # Force the slow path: the request names no in-chat file, so Locator runs.
        # Locator first replies with prose (no paths) -> retry -> valid path.
        # Then Planner, Editor, Reviewer, Summarizer.
        plan = "STEP: set x to 2\nFILES: calc.py"
        edit = _edit("calc.py", "x = 1", "x = 2")
        coder = self._coder([
            "Hmm, let me think about this.",  # Locator bad format
            "calc.py",                          # Locator retry: valid
            plan,                               # Planner
            edit,                               # Editor
            "VERDICT: OK",                      # Reviewer
            "Set x to 2.",                      # Summarizer
        ])
        # No files added to the chat, so the fast path can't fire.

        # A map is needed for Locator; use_map is off, so patch a trivial one in.
        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch._repo_map_text = lambda request: "calc.py\nother.py"  # type: ignore

        changed = orch.run("make the change")

        self.assertTrue(changed)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 2\n")
        # Locator was asked twice (bad format, then retry).
        locator_calls = sum(1 for r in self._roles_called() if r == "locate")
        self.assertEqual(locator_calls, 2)


class TestAutoModel(MicroTestBase):
    """Per-role automatic model switching based on task complexity."""

    def _models_by_role(self):
        """Map role (system-prompt key) -> list of models used for its calls."""
        out = {}
        for req in self.server.requests:
            sys_msg = req["messages"][0]["content"].lower()
            for key in ("locate", "turn a code request", "applying one small",
                        "fix code that failed", "review a code change", "summarise"):
                if key in sys_msg:
                    out.setdefault(key, []).append(req["model"])
                    break
        return out

    def test_off_by_default_uses_single_model(self):
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.model = "solo"
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch.run("set x to 2 in calc.py")

        models = {m for req in self.server.requests for m in [req["model"]]}
        self.assertEqual(models, {"solo"})

    def test_each_role_runs_on_its_mapped_model(self):
        (self.root / "calc.py").write_text("x = 1\n")
        # Fast path: Editor, Reviewer, Summarizer run for a one-file request.
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.auto_model = True
        coder.cfg.model_map = ('{"Editor":"code-m","Reviewer":"review-m",'
                               '"Summarizer":"fast-m"}')
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch.run("set x to 2 in calc.py")

        by_role = self._models_by_role()
        self.assertEqual(by_role["applying one small"], ["code-m"])
        self.assertEqual(by_role["review a code change"], ["review-m"])
        self.assertEqual(by_role["summarise"], ["fast-m"])

    def test_unmapped_role_falls_back_to_cfg_model(self):
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.auto_model = True
        coder.cfg.model = "base-m"
        coder.cfg.model_map = '{"Editor":"code-m"}'  # Reviewer/Summarizer unmapped
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch.run("set x to 2 in calc.py")

        by_role = self._models_by_role()
        self.assertEqual(by_role["applying one small"], ["code-m"])
        self.assertEqual(by_role["review a code change"], ["base-m"])

    def test_default_key_covers_unmapped_roles(self):
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.auto_model = True
        coder.cfg.model = "base-m"
        coder.cfg.model_map = '{"default":"default-m","Editor":"code-m"}'
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch.run("set x to 2 in calc.py")

        by_role = self._models_by_role()
        self.assertEqual(by_role["applying one small"], ["code-m"])
        self.assertEqual(by_role["review a code change"], ["default-m"])

    def test_banner_printed_for_every_role_even_when_repeated(self):
        import io
        from unittest import mock
        from rich.console import Console
        from agent_cli.micro import loop as loop_mod

        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.auto_model = True
        # Editor + Reviewer share one model; Summarizer unmapped -> cfg.model.
        coder.cfg.model = "base-m"
        coder.cfg.model_map = '{"Editor":"code-m","Reviewer":"code-m"}'
        coder.add(["calc.py"])

        buf = io.StringIO()
        with mock.patch.object(loop_mod, "console", Console(file=buf, force_terminal=False, width=200)):
            orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
            orch.run("set x to 2 in calc.py")
        out = buf.getvalue()

        # One banner per role that ran (fast path: Editor, Reviewer, Summarizer),
        # including Reviewer even though it repeats the Editor's model.
        self.assertIn("Editor \N{RIGHTWARDS ARROW} code-m", out)
        self.assertIn("Reviewer \N{RIGHTWARDS ARROW} code-m", out)
        self.assertIn("Summarizer \N{RIGHTWARDS ARROW} base-m", out)

    def test_cfg_model_restored_after_run(self):
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([_edit("calc.py", "x = 1", "x = 2"),
                             "VERDICT: OK", "Set x to 2."])
        coder.cfg.auto_model = True
        coder.cfg.model = "base-m"
        coder.cfg.model_map = '{"Editor":"code-m"}'
        coder.add(["calc.py"])

        orch = MicroOrchestrator(coder, checker=FakeChecker([CheckResult(ok=True)]))
        orch.run("set x to 2 in calc.py")

        self.assertEqual(coder.cfg.model, "base-m")  # override is per-call only


if __name__ == "__main__":
    unittest.main()
