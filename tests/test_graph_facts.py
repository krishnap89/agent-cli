"""Graph facts given to the micro Editor: callers, affected tests, reuse candidates."""
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli.coder import Coder
from agent_cli.config import Config
from agent_cli.graph import load_graph
from agent_cli.graph.facts import changed_functions, edit_facts, reuse_candidates
from agent_cli.micro import CheckResult, MicroOrchestrator
from tests.fake_server import FakeServer

FIX = Path(__file__).parent / "fixtures"
REPO_KT = "core/data/src/main/java/com/example/data/UserRepository.kt"
VM_KT = "app/src/main/java/com/example/app/MainViewModel.kt"


def _copy(name: str) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    shutil.copytree(FIX / name, Path(td.name) / "repo")
    return td


class TestEditFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = _copy("android_sample")
        cls.root = Path(cls.td.name) / "repo"
        cls.g, _ = load_graph(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_callers_outside_the_step_files(self):
        text = edit_facts(self.g, [REPO_KT], "make load accept a forceRefresh flag")
        self.assertIn("UserRepository.load: MainViewModel.refresh", text)
        self.assertIn(f"{VM_KT}:20", text)

    def test_named_function_narrows_the_changed_set(self):
        named = changed_functions(self.g, [REPO_KT], "change load")
        self.assertEqual([self.g.nodes[n].name for n in named], ["load"])
        everything = changed_functions(self.g, [REPO_KT], "tidy this file")
        self.assertGreater(len(everything), 1)

    def test_affected_tests_from_impact(self):
        text = edit_facts(self.g, [REPO_KT], "change load")
        self.assertIn("Affected tests (graph impact): "
                      "app/src/test/java/com/example/app/MainViewModelTest.kt", text)

    def test_callers_in_the_step_files_are_not_listed(self):
        text = edit_facts(self.g, [REPO_KT, VM_KT], "change load")
        self.assertNotIn("UserRepository.load: MainViewModel.refresh", text)

    def test_reuse_candidates_are_capped_and_exclude_step_files(self):
        hits = reuse_candidates(self.g, "add a clear cache invalidate helper", {VM_KT}, limit=5)
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 5)
        files = {self.g.nodes[nid].file for _s, nid in hits}
        self.assertNotIn(VM_KT, files)
        names = {self.g.nodes[nid].name for _s, nid in hits}
        self.assertTrue({"clear", "invalidate"} & names, names)

    def test_reuse_candidates_match_identifier_spelling(self):
        hits = reuse_candidates(self.g, "write a fetchUsers() wrapper", {VM_KT})
        names = [self.g.nodes[nid].name for _s, nid in hits]
        self.assertIn("fetch", names)

    def test_tests_are_never_reuse_candidates(self):
        hits = reuse_candidates(self.g, "refreshLoads", set())
        self.assertFalse(any("Test" in self.g.nodes[nid].file for _s, nid in hits))

    def test_unknown_files_and_no_matches_give_empty_text(self):
        self.assertEqual(edit_facts(self.g, ["nope.py"], "xyzzy"), "")

    def test_budget_caps_the_text(self):
        text = edit_facts(self.g, [REPO_KT], "clear cache invalidate", budget=120)
        self.assertLessEqual(len(text), 124)


class TestMicroEditorGetsFacts(unittest.TestCase):
    def setUp(self):
        self.td = _copy("android_sample")
        self.root = Path(self.td.name) / "repo"

    def tearDown(self):
        self.server.stop()
        self.td.cleanup()

    def _run(self, env):
        edit = (f"{REPO_KT}\n```\n<<<<<<< SEARCH\n    fun all(): List<User> = api.fetchAll()\n"
                "=======\n    fun all(): List<User> = api.fetchAll().sortedBy { it.id }\n"
                ">>>>>>> REPLACE\n```\nRisk: order changes for callers")
        self.server = FakeServer([edit, "VERDICT: OK", "Sorted all() by id."])
        self.server.start()
        cfg = Config(base_url=self.server.base_url, workdir=self.root, auto_approve=True,
                     stream=False, use_map=False, android="off")
        coder = Coder(cfg, mode="code")
        coder.add([REPO_KT])
        buf = io.StringIO()
        checker = mock.Mock()
        checker.check.return_value = CheckResult(ok=True, lint=True)
        with mock.patch.dict(os.environ, env), \
                mock.patch("agent_cli.micro.loop.console", Console(file=buf, width=300)):
            MicroOrchestrator(coder, checker=checker).run("In UserRepository.kt make load cache misses")
        editor_user = self.server.requests[0]["messages"][1]["content"]
        return editor_user, buf.getvalue()

    def test_facts_reach_the_editor(self):
        user, _ = self._run({"AGENT_GRAPH_FACTS": "1"})
        self.assertIn("Code graph facts:", user)
        self.assertIn("MainViewModel.refresh", user)

    def test_facts_can_be_turned_off(self):
        user, _ = self._run({"AGENT_GRAPH_FACTS": "0"})
        self.assertNotIn("Code graph facts:", user)

    def test_micro_run_ends_with_risk_note(self):
        _, out = self._run({"AGENT_GRAPH_FACTS": "0"})
        risk = [ln for ln in out.splitlines() if ln.startswith("Risk: ")]
        self.assertEqual(len(risk), 1, out)
        self.assertIn("lint ok, not compiled", risk[0])
        self.assertIn("tests not run", risk[0])
        self.assertIn("model: order changes for callers", risk[0])


class TestFactsSkipNonGraphSteps(unittest.TestCase):
    def test_python_step_does_not_load_the_graph(self):
        from agent_cli.micro import PlanStep
        with tempfile.TemporaryDirectory() as td:
            coder = Coder(Config(workdir=Path(td), use_map=False), mode="code")
            orch = MicroOrchestrator(coder, call=lambda m: None)
            with mock.patch("agent_cli.graph.load_graph", side_effect=AssertionError("loaded")):
                self.assertEqual(orch._graph_facts(PlanStep("s", ["a.py"])), "")


if __name__ == "__main__":
    unittest.main()
