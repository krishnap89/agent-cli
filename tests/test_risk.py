"""The one-line risk note after coder edits: facts + the model's own line."""
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli.coder import Coder
from agent_cli.config import Config
from agent_cli.risk import (CheckFacts, callers_not_updated, model_risk_line, risk_note,
                            strip_risk_line)
from tests.fake_server import FakeServer

FIX = Path(__file__).parent / "fixtures"
REPO_KT = "core/data/src/main/java/com/example/data/UserRepository.kt"
VM_KT = "app/src/main/java/com/example/app/MainViewModel.kt"


class TestRiskLine(unittest.TestCase):
    def test_model_risk_line_takes_the_last_one(self):
        self.assertEqual(model_risk_line("a\nRisk: one\nb\n**Risk:** two"), "two")
        self.assertEqual(model_risk_line("no line here"), "")
        self.assertEqual(model_risk_line("  risk: `low`"), "low")

    def test_strip_risk_line(self):
        self.assertEqual(strip_risk_line("Done.\nRisk: low"), "Done.")

    def test_note_with_all_facts(self):
        note = risk_note(CheckFacts(lint=True, compile=True, tests=True),
                         [("Repo.load", "Vm.refresh", "vm.kt:20")], "nulls")
        self.assertEqual(note, "Risk: lint ok, compile ok · tests passed · 1 caller not updated: "
                               "Vm.refresh (vm.kt:20) · model: nulls")

    def test_note_without_checks_or_model_line(self):
        self.assertEqual(risk_note(CheckFacts(), [], ""),
                         "Risk: not checked · tests not run · model: (no risk line)")

    def test_note_lint_only_and_failures(self):
        self.assertIn("lint ok, not compiled", risk_note(CheckFacts(lint=True), [], "x"))
        note = risk_note(CheckFacts(lint=True, compile=False, tests=False), [], "x")
        self.assertIn("compile FAILED", note)
        self.assertIn("tests FAILED", note)

    def test_many_missed_callers_are_capped(self):
        missed = [("f", f"c{i}", f"a.kt:{i}") for i in range(5)]
        note = risk_note(CheckFacts(), missed, "")
        self.assertIn("5 callers not updated", note)
        self.assertIn("+2 more", note)
        self.assertNotIn("c4", note)


class TestCallersNotUpdated(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name) / "repo"
        shutil.copytree(FIX / "android_sample", self.root)
        self.old = (self.root / REPO_KT).read_text()

    def tearDown(self):
        self.td.cleanup()

    def _edit(self, old, new):
        (self.root / REPO_KT).write_text(self.old.replace(old, new))

    def test_signature_change_reports_untouched_caller(self):
        self._edit("suspend fun load(id: String): User",
                   "suspend fun load(id: String, force: Boolean): User")
        missed = callers_not_updated(self.root, {REPO_KT: self.old})
        self.assertEqual([(f, c) for f, c, _ in missed], [("UserRepository.load", "MainViewModel.refresh")])

    def test_body_only_change_reports_nothing(self):
        self._edit("val cached = cache.get(id)", "val cached = cache.get(id.trim())")
        self.assertEqual(callers_not_updated(self.root, {REPO_KT: self.old}), [])

    def test_caller_file_in_the_same_edit_is_not_reported(self):
        self._edit("suspend fun load(id: String): User", "suspend fun load(id: String, f: Boolean): User")
        vm = (self.root / VM_KT).read_text()
        missed = callers_not_updated(self.root, {REPO_KT: self.old, VM_KT: vm})
        self.assertEqual(missed, [])

    def test_new_files_and_unparsed_languages_are_skipped(self):
        (self.root / "x.py").write_text("def f(a, b):\n    pass\n")
        self.assertEqual(callers_not_updated(self.root, {"x.py": "def f(a):\n    pass\n",
                                                         "new.kt": None}), [])


class TestCoderPrintsRiskNote(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "calc.py").write_text("x = 1\n")

    def tearDown(self):
        self.server.stop()
        self.td.cleanup()

    def _send(self, reply):
        self.server = FakeServer([reply])
        self.server.start()
        cfg = Config(base_url=self.server.base_url, workdir=self.root, auto_approve=True,
                     stream=False, use_map=False)
        coder = Coder(cfg, mode="code")
        coder.add(["calc.py"])
        buf = io.StringIO()
        with mock.patch("agent_cli.coder.console", Console(file=buf, width=300)), \
                mock.patch("agent_cli.agent.console", Console(file=buf, width=300)), \
                mock.patch("agent_cli.coder.Confirm.ask", return_value=False):
            coder.send("set x to 2")
        return coder, buf.getvalue()

    def test_edit_reply_ends_with_risk_note(self):
        reply = ("Setting x.\n\ncalc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n"
                 ">>>>>>> REPLACE\n```\nRisk: anything reading x sees 2")
        coder, out = self._send(reply)
        lines = [ln for ln in out.splitlines() if ln.startswith("Risk: ")]
        self.assertEqual(lines, ["Risk: lint ok, not compiled · tests not run · "
                                 "model: anything reading x sees 2"])
        self.assertTrue(coder.last_checks.lint)

    def test_markup_in_model_line_is_shown_verbatim(self):
        reply = ("calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n"
                 ">>>>>>> REPLACE\n```\nRisk: [bold]x[/bold] may be [/] stale")
        _, out = self._send(reply)
        self.assertIn("model: [bold]x[/bold] may be [/] stale", out)

    def test_lint_failure_shows_in_note(self):
        reply = ("calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = (\n"
                 ">>>>>>> REPLACE\n```")
        coder, out = self._send(reply)
        self.assertIn("Risk: lint FAILED", out)
        self.assertIn("model: (no risk line)", out)

    def test_no_note_without_edits(self):
        _, out = self._send("Which x do you mean?")
        self.assertNotIn("Risk: ", out)


class TestReviewFixes(unittest.TestCase):
    """Regressions from the code review."""

    def test_risk_lines_inside_code_are_ignored(self):
        reply = ("Done.\n\nf.kt\n```kotlin\n<<<<<<< SEARCH\nfun f(\n    risk: Double,\n=======\n"
                 "fun f(\n    risk: Double, x: Int,\n>>>>>>> REPLACE\n```\n")
        self.assertEqual(model_risk_line(reply), "")
        self.assertIn("    risk: Double, x: Int,", strip_risk_line(reply))
        self.assertEqual(model_risk_line(reply + "Risk: x is new"), "x is new")
        fenced = "```yaml\nrisk: high\n```\nRisk: low"
        self.assertEqual(model_risk_line(fenced), "low")
        self.assertIn("risk: high", strip_risk_line(fenced))

    def test_non_graph_files_skip_the_graph(self):
        with mock.patch("agent_cli.graph.load_graph", side_effect=AssertionError("loaded")):
            self.assertEqual(callers_not_updated(Path("."), {"a.py": "x"}), [])


class TestRenamedFunction(TestCallersNotUpdated):
    def test_renamed_function_reports_old_callers(self):
        self._edit("suspend fun load(id: String): User", "suspend fun fetchUser(id: String): User")
        missed = callers_not_updated(self.root, {REPO_KT: self.old})
        self.assertEqual([(c, loc) for _f, c, loc in missed],
                         [("call to removed load", f"{VM_KT}:20")])

    def test_removed_name_still_defined_elsewhere_is_not_reported(self):
        # `clear` also exists on UserCache, so its removal from Repository can't be judged by name
        self._edit("    override fun clear() {\n        cache.clear()\n    }\n", "")
        missed = callers_not_updated(self.root, {REPO_KT: self.old})
        self.assertFalse(any("removed clear" in c for _f, c, _l in missed))


class TestCoderRiskRegressions(TestCoderPrintsRiskNote):
    def test_note_failure_keeps_the_edit_undoable(self):
        reply = "calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```"
        with mock.patch("agent_cli.coder.callers_not_updated", side_effect=RuntimeError("boom")):
            coder, out = self._send(reply)
        self.assertEqual(len(coder.undo_stack), 1)
        self.assertIn("Risk: (note unavailable: boom)", out)

    def test_latest_fix_round_risk_line_wins(self):
        self.server = FakeServer([
            "calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = (\n>>>>>>> REPLACE\n```\nRisk: low",
            "calc.py\n```python\n<<<<<<< SEARCH\nx = (\n=======\nx = 2\n>>>>>>> REPLACE\n```\nRisk: x type changed",
        ])
        self.server.start()
        cfg = Config(base_url=self.server.base_url, workdir=self.root, auto_approve=True,
                     stream=False, use_map=False)
        coder = Coder(cfg, mode="code")
        coder.add(["calc.py"])
        buf = io.StringIO()
        with mock.patch("agent_cli.coder.console", Console(file=buf, width=300)), \
                mock.patch("agent_cli.agent.console", Console(file=buf, width=300)), \
                mock.patch("agent_cli.coder.Confirm.ask", return_value=False):
            coder.send("set x")
        out = buf.getvalue()
        self.assertIn("Risk: lint ok, not compiled · tests not run · model: x type changed", out)
        self.assertNotIn("\nRisk: x type changed\n", out)  # not echoed raw in the reply


if __name__ == "__main__":
    unittest.main()
