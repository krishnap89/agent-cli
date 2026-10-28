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
from unittest import mock

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


class TestAnalysisPipeline(MicroTestBase):
    """The read-only Locator -> Analyst -> (Diagrammer) -> Writer pipeline."""

    def setUp(self):
        super().setUp()
        # Most tests here script the role sequence exactly, so keep the optional
        # LLM Keyworder call out of it by default; the Keyworder tests opt in.
        patcher = mock.patch.dict(os.environ, {"AGENT_ANALYSIS_LLM_KEYWORDS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, request, script):
        from agent_cli.micro import AnalysisOrchestrator
        coder = self._coder(script)
        AnalysisOrchestrator(coder).run(request)
        return coder

    def test_prose_request_writes_doc_without_diagram(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("class Calc:\n    pass\n")
        coder = self._coder([
            "calc.py",                      # Locator
            "RELEVANT: YES",                # RelevanceGate
            "Calc is a class with no attributes.",  # Analyst
            "# Calc\n\nCalc is a class.",   # Writer (no Diagrammer: not a diagram request)
        ])
        AnalysisOrchestrator(coder).run("explain the Calc class")

        out = self.root / "docs" / "analysis" / "explain-the-calc-class.md"
        self.assertTrue(out.is_file())
        self.assertIn("# Calc", out.read_text())
        # Locator, RelevanceGate, Analyst, Writer ran (no Diagrammer).
        self.assertEqual(len(self.server.requests), 4)

    def test_diagram_request_runs_diagrammer_and_embeds_it(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("class Calc:\n    pass\n")
        coder = self._coder([
            "calc.py",                                  # Locator
            "RELEVANT: YES",                            # RelevanceGate
            "Calc relates to nothing.",                 # Analyst
            "```mermaid\nclassDiagram\n  class Calc\n```",  # Diagrammer
            "# Calc\n\n```mermaid\nclassDiagram\n  class Calc\n```\n\nText.",  # Writer
        ])
        AnalysisOrchestrator(coder).run("show the class diagram")

        self.assertEqual(len(self.server.requests), 5)  # Diagrammer ran
        out = self.root / "docs" / "analysis" / "show-the-class-diagram.md"
        self.assertIn("classDiagram", out.read_text())

    def test_no_source_file_is_edited(self):
        from agent_cli.micro import AnalysisOrchestrator
        src = self.root / "calc.py"
        src.write_text("class Calc:\n    pass\n")
        before = src.read_text()
        coder = self._coder([
            "calc.py",
            "RELEVANT: YES",
            "Calc is a class.",
            "# Calc\n\nText.",
        ])
        AnalysisOrchestrator(coder).run("describe calc.py")

        self.assertEqual(src.read_text(), before)   # source untouched
        self.assertNotIn("calc.py", coder.editable)  # not added as editable

    def test_undo_removes_the_generated_doc(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("class Calc:\n    pass\n")
        coder = self._coder(["calc.py", "RELEVANT: YES", "Calc is a class.", "# Calc\n\nText."])
        AnalysisOrchestrator(coder).run("describe calc.py")
        out = self.root / "docs" / "analysis" / "describe-calc-py.md"
        self.assertTrue(out.is_file())

        coder.undo()
        self.assertFalse(out.exists())  # old was None -> undo deletes it

    def test_irrelevant_files_stop_the_run_before_the_analyst(self):
        from agent_cli.micro import AnalysisOrchestrator
        # The Locator returns unrelated files; the gate says NO, so no doc is
        # written and the Analyst never runs.
        (self.root / "color.py").write_text("WHITE = 1\n")
        coder = self._coder([
            "color.py",                                    # Locator
            "RELEVANT: NO\nREASON: these are color constants, not auth.",  # RelevanceGate
        ])
        wrote = AnalysisOrchestrator(coder).run("show the authentication")

        self.assertFalse(wrote)
        self.assertEqual(len(self.server.requests), 2)  # Locator + gate only
        self.assertFalse((self.root / "docs" / "analysis").exists())

    def test_gate_can_be_disabled(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "color.py").write_text("WHITE = 1\n")
        # No gate reply scripted: with the gate off it must not be called.
        coder = self._coder(["color.py", "Some analysis.", "# Doc\n\nText."])
        with mock.patch.dict(os.environ, {"AGENT_ANALYSIS_GATE": "0"}):
            wrote = AnalysisOrchestrator(coder).run("show the authentication")
        self.assertTrue(wrote)
        self.assertEqual(len(self.server.requests), 3)  # Locator, Analyst, Writer

    def test_unparseable_gate_verdict_proceeds(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("class Calc:\n    pass\n")
        # Gate replies garbage twice (role retries once); the run proceeds anyway.
        coder = self._coder([
            "calc.py", "huh?", "still no verdict",
            "Calc is a class.", "# Calc\n\nText.",
        ])
        wrote = AnalysisOrchestrator(coder).run("describe calc.py")
        self.assertTrue(wrote)

    def test_locator_path_that_does_not_exist_is_dropped(self):
        from agent_cli.micro import AnalysisOrchestrator
        coder = self._coder(["nope/missing.py"])  # Locator names a file that isn't there
        wrote = AnalysisOrchestrator(coder).run("explain missing.py")
        self.assertFalse(wrote)
        # Only the Locator ran; with no real files the pipeline stops before the Analyst.
        self.assertEqual(len(self.server.requests), 1)

    def test_keywords_drop_verbs_and_stopwords(self):
        from agent_cli.micro.analysis import request_keywords
        kw = request_keywords("show the class diagram of the authentication")
        self.assertIn("authentication", kw)
        self.assertIn("auth", kw)          # stem added for the long word
        self.assertNotIn("show", kw)       # analysis verb dropped
        self.assertNotIn("the", kw)        # stopword dropped
        self.assertNotIn("diagram", kw)

    def test_presearch_surfaces_files_the_map_would_miss(self):
        from agent_cli.micro import AnalysisOrchestrator
        # A file that mentions the subject but would be a poor repo-map hit.
        (self.root / "SupabaseAuthDataSource.kt").write_text(
            "class SupabaseAuthDataSource { fun login() {} }\n")
        (self.root / "Color.kt").write_text("val WHITE = 1\n")
        coder = self._coder(["ignored"])  # no real model call: we test _presearch directly
        orch = AnalysisOrchestrator(coder)
        hits = orch._presearch("show the authentication class diagram")
        self.assertIn("SupabaseAuthDataSource.kt", hits)
        self.assertNotIn("Color.kt", hits)

    def test_presearch_hits_are_passed_to_the_locator(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "SupabaseAuthDataSource.kt").write_text(
            "class SupabaseAuthDataSource { fun login() {} }\n")
        coder = self._coder([
            "SupabaseAuthDataSource.kt",   # Locator (picks the pre-search hit)
            "RELEVANT: YES",
            "It authenticates via Supabase.",
            "# Auth\n\nText.",
        ])
        AnalysisOrchestrator(coder).run("explain the authentication")
        # The Locator's user message carries the keyword-search section.
        locator_msg = self.server.requests[0]["messages"][1]["content"]
        self.assertIn("keyword search", locator_msg.lower())
        self.assertIn("SupabaseAuthDataSource.kt", locator_msg)

    def test_llm_keywords_union_with_deterministic_terms(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder(["auth, login, session, authentication"])  # Keyworder reply
        orch = AnalysisOrchestrator(coder)
        with mock.patch.dict(os.environ, {"AGENT_ANALYSIS_LLM_KEYWORDS": "1"}):
            terms = orch._search_terms("explain the authentication")
        # Deterministic terms (authentication, auth) are kept; LLM adds login/session.
        self.assertIn("authentication", terms)
        self.assertIn("auth", terms)
        self.assertIn("login", terms)
        self.assertIn("session", terms)
        # No duplicates despite overlap between the two sources.
        self.assertEqual(len(terms), len(set(terms)))

    def test_llm_keyword_failure_falls_back_to_deterministic(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("x = 1\n")
        # Keyworder replies garbage twice (role retries once) -> RoleError -> fallback.
        coder = self._coder(["???", "still nothing"])
        orch = AnalysisOrchestrator(coder)
        with mock.patch.dict(os.environ, {"AGENT_ANALYSIS_LLM_KEYWORDS": "1"}):
            terms = orch._search_terms("explain the authentication")
        self.assertEqual(terms, ["authentication", "auth"])  # deterministic only

    def test_llm_keywords_skipped_when_disabled(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._coder([])  # no replies: Keyworder must not be called
        orch = AnalysisOrchestrator(coder)
        with mock.patch.dict(os.environ, {"AGENT_ANALYSIS_LLM_KEYWORDS": "0"}):
            terms = orch._search_terms("explain the authentication")
        self.assertEqual(terms, ["authentication", "auth"])
        self.assertEqual(len(self.server.requests), 0)  # no LLM call made

    def test_presearch_excludes_own_analysis_output(self):
        from agent_cli.micro import AnalysisOrchestrator
        # A real source file and a prior analysis doc, both mentioning "auth".
        (self.root / "AuthManager.kt").write_text("class AuthManager { fun login() {} }\n")
        prior = self.root / "docs" / "analysis" / "class-diagram-of-the.md"
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_text("# Auth\n\nThis is about authentication and login tokens.\n")
        coder = self._coder(["ignored"])
        hits = AnalysisOrchestrator(coder)._presearch("the authentication class")
        self.assertIn("AuthManager.kt", hits)
        self.assertNotIn("docs/analysis/class-diagram-of-the.md", hits)  # own output filtered

    def test_locator_naming_an_output_doc_has_it_dropped(self):
        from agent_cli.micro import AnalysisOrchestrator
        (self.root / "AuthManager.kt").write_text("class AuthManager { fun login() {} }\n")
        prior = self.root / "docs" / "analysis" / "old.md"
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_text("# old auth doc\n")
        # Locator names BOTH the stale doc and the real source; the doc is dropped.
        coder = self._coder([
            "docs/analysis/old.md\nAuthManager.kt",
            "RELEVANT: YES",
            "AuthManager authenticates.",
            "# Auth\n\nText.",
        ])
        AnalysisOrchestrator(coder).run("the authentication class")
        # The gate saw only the real source, not the stale doc.
        gate_msg = self.server.requests[1]["messages"][1]["content"]
        self.assertIn("AuthManager.kt", gate_msg)
        self.assertNotIn("docs/analysis/old.md", gate_msg)

    def test_keyworder_parse_accepts_commas_and_strips_junk(self):
        from agent_cli.micro import Keyworder
        kw = Keyworder(lambda m: None)
        self.assertEqual(kw.parse("auth, login, Session\n- token"),
                         ["auth", "login", "session", "token"])

    def test_routing_sends_read_requests_to_analysis(self):
        from agent_cli.micro import is_analysis_request
        self.assertTrue(is_analysis_request("show the class diagram"))
        self.assertTrue(is_analysis_request("explain how auth works"))
        self.assertTrue(is_analysis_request("document the ProductFilter model"))
        self.assertFalse(is_analysis_request("add a --version flag"))
        self.assertFalse(is_analysis_request("rename foo to bar"))


if __name__ == "__main__":
    unittest.main()
