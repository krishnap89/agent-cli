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
