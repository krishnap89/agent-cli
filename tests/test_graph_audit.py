"""`agent graph audit`: dead code, near-duplicates, untested risky files."""
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_cli.config import Config
from agent_cli.graph import load_graph
from agent_cli.graph import audit
from agent_cli.graph.cli import run
from agent_cli.llm import LLMError
from tests.fake_server import FakeServer

FIX = Path(__file__).parent / "fixtures"

DUP_TS = """export function totalPrice(items: Item[]): number {
  let sum = 0
  for (const it of items) {
    if (it.qty > 0) {
      sum += it.price * it.qty
    }
  }
  return Math.round(sum * 100) / 100
}

export function totalCost(lines: Item[]): number {
  let sum = 0
  for (const it of lines) {
    if (it.qty > 0) {
      sum += it.price * it.qty
    }
  }
  return Math.round(sum * 100) / 100
}

export function unrelated(a: number): string {
  const parts = [String(a), 'x', 'y']
  return parts.join('-')
    .toUpperCase()
    .trim()
}
"""


class _Copy(unittest.TestCase):
    fixture = "android_sample"

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name) / "repo"
        shutil.copytree(FIX / cls.fixture, cls.root)
        cls.prepare(cls.root)
        cls.g, _ = load_graph(cls.root)

    @classmethod
    def prepare(cls, root):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()


class TestDeadCode(_Copy):
    def test_uncalled_function_is_dead(self):
        names = {f.title for f in audit.dead_code(self.g)}
        self.assertIn("UserRepository.all", names)
        self.assertIn("Outer.lambdas", names)

    def test_called_override_and_lifecycle_functions_are_not_dead(self):
        names = {f.title for f in audit.dead_code(self.g)}
        for live in ("UserRepository.load", "MainViewModel.refresh", "MainViewModel.onCleared",
                     "MainActivity.onCreate", "UserRepository.clear", "Repository.clear"):
            self.assertNotIn(live, names)

    def test_tests_are_never_dead(self):
        self.assertFalse(any("Test" in f.loc for f in audit.dead_code(self.g)))

    def test_score_is_line_count(self):
        lam = next(f for f in audit.dead_code(self.g) if f.title == "Outer.lambdas")
        self.assertEqual(lam.score, 10.0)


class TestUntested(_Copy):
    def test_files_reached_by_tests_are_excluded(self):
        files = {f.title for f in audit.untested_risky(self.g, audit._Sources(self.root))}
        self.assertNotIn("app/src/main/java/com/example/app/MainViewModel.kt", files)
        self.assertNotIn("core/data/src/main/java/com/example/data/UserRepository.kt", files)
        self.assertIn("app/src/main/java/com/example/app/MainActivity.kt", files)

    def test_resources_are_not_modules(self):
        files = {f.title for f in audit.untested_risky(self.g, audit._Sources(self.root))}
        self.assertFalse(any(f.endswith((".xml", ".json")) for f in files), files)


class TestNearDuplicates(_Copy):
    fixture = "ts_sample"

    @classmethod
    def prepare(cls, root):
        (root / "apps/web/src/price.ts").write_text(DUP_TS)

    def test_similar_bodies_pair_up(self):
        dups = audit.near_duplicates(self.g, audit._Sources(self.root))
        self.assertEqual(len(dups), 1)
        self.assertIn("totalPrice", dups[0].title)
        self.assertIn("totalCost", dups[0].title)
        self.assertGreaterEqual(float(dups[0].detail.split("%")[0]), 85)

    def test_threshold(self):
        self.assertEqual(audit.near_duplicates(self.g, audit._Sources(self.root), min_ratio=1.01), [])

    def test_audit_ranks_by_score(self):
        found = audit.audit(self.g, self.root)
        scores = [f.score for f in found]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual({f.kind for f in found}, {audit.DEAD, audit.DUP, audit.UNTESTED})


class TestAuditCli(_Copy):
    def test_no_explain_lists_ranked_findings(self):
        out = io.StringIO()
        self.assertEqual(run(["audit", "--no-explain", "-d", str(self.root), "--limit", "3"], out), 0)
        text = out.getvalue()
        self.assertIn("audit: ", text)
        self.assertIn("  1. [", text)
        self.assertNotIn("  4. [", text)
        self.assertNotIn("Model's take", text)

    def test_model_explains_only_the_top_five(self):
        server = FakeServer(["1. real, delete it."])
        server.start()
        real = audit.explain

        def explain(findings, g, root, cfg=None):
            cfg = Config(base_url=server.base_url, stream=False, workdir=root)
            return real(findings, g, root, cfg)
        try:
            with mock.patch("agent_cli.graph.audit.explain", side_effect=explain):
                out = io.StringIO()
                run(["audit", "-d", str(self.root)], out)
        finally:
            server.stop()
        self.assertEqual(len(server.requests), 1)
        prompt = server.requests[0]["messages"][1]["content"]
        self.assertIn("\n\n5. [", prompt)
        self.assertNotIn("\n\n6. [", prompt)
        self.assertIn("```", prompt)  # code is shown
        self.assertIn("Model's take on the top 5:\n1. real, delete it.", out.getvalue())

    def test_model_unavailable_still_prints_findings(self):
        out = io.StringIO()
        with mock.patch("agent_cli.graph.audit.explain", side_effect=LLMError("down\nmore")):
            run(["audit", "-d", str(self.root)], out)
        self.assertIn("  1. [", out.getvalue())
        self.assertIn("(model unavailable: down; use --no-explain)", out.getvalue())

    def test_repo_without_graph_code(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.py").write_text("def f():\n    pass\n")
            out = io.StringIO()
            run(["audit", "-d", td], out)
        self.assertIn("nothing to audit", out.getvalue())


class TestAuditSharesTestReach(_Copy):
    def test_cli_walks_test_reachability_once(self):
        real = audit._tested_files
        with mock.patch("agent_cli.graph.audit._tested_files", side_effect=real) as walk:
            run(["audit", "--no-explain", "-d", str(self.root)], io.StringIO())
        self.assertEqual(walk.call_count, 1)


if __name__ == "__main__":
    unittest.main()
