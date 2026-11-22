"""Diff-size eval harness (mechanics only; real numbers need a model server)."""
import json
import tempfile
import unittest
from pathlib import Path

from agent_cli import evaldiff
from tests.fake_server import FakeServer

BIG = ("calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nimport os\n\n\ndef get_x():\n"
       "    return int(os.getenv('X', '2'))\n\n\nx = get_x()\n>>>>>>> REPLACE\n```")
SMALL = "calc.py\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```"


class TestDiffSize(unittest.TestCase):
    def test_counts_plus_minus_without_headers(self):
        d = "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x = 1\n+x = 2\n+y = 3\n ctx\n"
        self.assertEqual(evaldiff.diff_size([d, d]), 6)

    def test_task_set_is_valid(self):
        tasks = json.loads(evaldiff.DEFAULT_TASKS.read_text())
        self.assertGreaterEqual(len(tasks), 5)
        for t in tasks:
            fx = evaldiff.REPO / t["fixture"]
            for f in t["files"]:
                self.assertTrue((fx / f).is_file(), f"{t['id']}: {f}")


class TestRunEval(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.base = Path(self.td.name)
        (self.base / "fx").mkdir()
        (self.base / "fx" / "calc.py").write_text("x = 1\n")
        # The fake model writes a big change without the style rules, a small one with them.
        self.server = FakeServer([lambda m: SMALL if "smallest change" in m[0]["content"] else BIG] * 4)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.td.cleanup()

    def test_off_vs_full(self):
        tasks = [{"id": "t1", "fixture": "fx", "files": ["calc.py"], "request": "set x to 2"}]
        res = evaldiff.run_eval(tasks, ["off", "full"], base=self.base,
                                cfg_overrides={"base_url": self.server.base_url, "stream": False,
                                               "use_map": False, "lint_enabled": False})
        self.assertEqual(res, {"t1": {"off": [9], "full": [2]}})
        self.assertEqual((self.base / "fx" / "calc.py").read_text(), "x = 1\n")  # fixture untouched
        text = evaldiff.report(res, ["off", "full"])
        self.assertIn("full vs off: -77.8% changed lines", text)

    def test_report_marks_tasks_without_edits(self):
        text = evaldiff.report({"a": {"off": [4], "full": [-1]}, "b": {"off": [6], "full": [3]}},
                               ["off", "full"])
        self.assertIn("(1/2 tasks edited under every style)", text)
        self.assertIn("full vs off: -50.0%", text)


if __name__ == "__main__":
    unittest.main()
