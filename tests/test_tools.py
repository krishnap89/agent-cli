"""Tests for apply_patch (atomic multi-edit) and the todo checklist tool."""
import tempfile
import unittest
from pathlib import Path

from agent_cli.config import Config
from agent_cli.tools import Tools, ToolError


def make_tools() -> tuple[Tools, Path]:
    d = Path(tempfile.mkdtemp())
    cfg = Config(workdir=d, auto_approve=True)
    return Tools(cfg), d


class ApplyPatchTest(unittest.TestCase):
    def test_multi_file_edit_and_create(self):
        t, d = make_tools()
        (d / "a.py").write_text("x = 1\ny = 2\n")
        out = t.run("apply_patch", {"edits": [
            {"path": "a.py", "old_str": "x = 1", "new_str": "x = 42"},
            {"path": "b.py", "old_str": "", "new_str": "print(1)\n"},
        ]})
        self.assertIn("Changed: a.py", out)
        self.assertIn("Created: b.py", out)
        self.assertEqual((d / "a.py").read_text(), "x = 42\ny = 2\n")
        self.assertEqual((d / "b.py").read_text(), "print(1)\n")

    def test_atomic_nothing_written_on_failure(self):
        t, d = make_tools()
        (d / "a.py").write_text("x = 1\ny = 2\n")
        with self.assertRaises(ToolError):
            t.run("apply_patch", {"edits": [
                {"path": "a.py", "old_str": "y = 2", "new_str": "y = 99"},
                {"path": "a.py", "old_str": "NOPE", "new_str": "q"},
            ]})
        # First edit must not have been applied.
        self.assertEqual((d / "a.py").read_text(), "x = 1\ny = 2\n")

    def test_sequential_edits_to_same_file(self):
        t, d = make_tools()
        (d / "a.py").write_text("x = 1\n")
        t.run("apply_patch", {"edits": [
            {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
            {"path": "a.py", "old_str": "x = 2", "new_str": "x = 3"},
        ]})
        self.assertEqual((d / "a.py").read_text(), "x = 3\n")

    def test_non_unique_old_str_rejected(self):
        t, d = make_tools()
        (d / "a.py").write_text("v = 1\nv = 1\n")
        with self.assertRaises(ToolError):
            t.run("apply_patch", {"edits": [
                {"path": "a.py", "old_str": "v = 1", "new_str": "v = 2"},
            ]})

    def test_create_over_existing_file_rejected(self):
        t, d = make_tools()
        (d / "a.py").write_text("x = 1\n")
        with self.assertRaises(ToolError):
            t.run("apply_patch", {"edits": [
                {"path": "a.py", "old_str": "", "new_str": "nope"},
            ]})

    def test_empty_edits_rejected(self):
        t, _ = make_tools()
        with self.assertRaises(ToolError):
            t.run("apply_patch", {"edits": []})

    def test_denied_patch_writes_nothing(self):
        t, d = make_tools()
        t.cfg.auto_approve = False
        t._confirm = lambda action: False
        (d / "a.py").write_text("x = 1\n")
        out = t.run("apply_patch", {"edits": [
            {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
        ]})
        self.assertIn("denied", out)
        self.assertEqual((d / "a.py").read_text(), "x = 1\n")


class DiffPreviewTest(unittest.TestCase):
    """_show_diff runs before a write confirm and must not alter behaviour."""

    def _capture(self, t: Tools):
        calls = []
        t._show_diff = lambda rel, old, new: calls.append((rel, old, new))
        return calls

    def test_edit_shows_diff_of_the_change(self):
        t, d = make_tools()
        t.cfg.auto_approve = False
        t._confirm = lambda action: True
        calls = self._capture(t)
        (d / "a.py").write_text("x = 1\n")
        t.run("edit_file", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})
        self.assertEqual(calls, [("a.py", "x = 1\n", "x = 2\n")])

    def test_write_shows_old_vs_new(self):
        t, d = make_tools()
        t.cfg.auto_approve = False
        t._confirm = lambda action: True
        calls = self._capture(t)
        (d / "a.py").write_text("old\n")
        t.run("write_file", {"path": "a.py", "content": "new\n"})
        self.assertEqual(calls, [("a.py", "old\n", "new\n")])

    def test_apply_patch_shows_one_diff_per_file(self):
        t, d = make_tools()
        t.cfg.auto_approve = False
        t._confirm = lambda action: True
        calls = self._capture(t)
        (d / "a.py").write_text("x = 1\n")
        t.run("apply_patch", {"edits": [
            {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
            {"path": "b.py", "old_str": "", "new_str": "new\n"},
        ]})
        rels = sorted(c[0] for c in calls)
        self.assertEqual(rels, ["a.py", "b.py"])
        # the new file is shown as old="" -> content
        b = next(c for c in calls if c[0] == "b.py")
        self.assertEqual(b[1], "")


class SummarizeTest(unittest.TestCase):
    def _s(self, name, output):
        from agent_cli.agent import Agent
        return Agent._summarize(name, output)

    def test_search_uses_header_line(self):
        self.assertEqual(self._s("search", "12 matches in 4 files\nfoo.py\n  3: x"),
                         "12 matches in 4 files")

    def test_read_file_counts_lines(self):
        self.assertEqual(self._s("read_file", "a\nb\nc"), "3 lines")

    def test_list_dir_singular_plural(self):
        self.assertEqual(self._s("list_dir", "only"), "1 entry")
        self.assertEqual(self._s("list_dir", "a\nb"), "2 entries")

    def test_empty_output(self):
        self.assertEqual(self._s("read_file", "   "), "(no output)")

    def test_write_uses_first_line(self):
        self.assertEqual(self._s("write_file", "Wrote a.py"), "Wrote a.py")


class TodoTest(unittest.TestCase):
    def test_set_and_list(self):
        t, _ = make_tools()
        out = t.run("todo", {"action": "set", "items": [
            {"text": "plan", "status": "done"},
            {"text": "build", "status": "in_progress"},
            {"text": "test"},
        ]})
        self.assertIn("1/3 done", out)
        self.assertIn("[x] plan", out)
        self.assertIn("[~] build", out)
        self.assertIn("[ ] test", out)
        self.assertEqual(t.run("todo", {"action": "list"}), out)

    def test_set_replaces_list(self):
        t, _ = make_tools()
        t.run("todo", {"action": "set", "items": [{"text": "old"}]})
        out = t.run("todo", {"action": "set", "items": [{"text": "new"}]})
        self.assertIn("new", out)
        self.assertNotIn("old", out)

    def test_empty_list(self):
        t, _ = make_tools()
        self.assertIn("empty", t.run("todo", {"action": "list"}))

    def test_invalid_status_rejected(self):
        t, _ = make_tools()
        with self.assertRaises(ToolError):
            t.run("todo", {"action": "set", "items": [{"text": "x", "status": "maybe"}]})

    def test_item_without_text_rejected(self):
        t, _ = make_tools()
        with self.assertRaises(ToolError):
            t.run("todo", {"action": "set", "items": [{"status": "done"}]})


if __name__ == "__main__":
    unittest.main()
