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


class SearchFindsFilesTest(unittest.TestCase):
    """search must surface files whose *name* matches, not just contents."""

    def _repo(self):
        import subprocess
        t, d = make_tools()
        (d / "pkg").mkdir()
        (d / "pkg" / "agent.py").write_text("def run(): pass\n")
        (d / "README.md").write_text("The agent.py module does X.\n")
        (d / "other.py").write_text("import something\n")
        subprocess.run(["git", "init", "-q"], cwd=d)
        subprocess.run(["git", "add", "-A"], cwd=d)
        return t, d

    def test_filename_only_match_when_no_content(self):
        t, d = self._repo()
        # 'pkg/agent.py' is a filename but no file *content* is literally that string
        out = t.run("search", {"pattern": "pkg/agent.py"})
        self.assertIn("looks like a filename", out)
        self.assertIn("pkg/agent.py", out)

    def test_filename_appended_to_content_hits(self):
        t, d = self._repo()
        out = t.run("search", {"pattern": "agent.py"})
        # README mentions it (content) AND pkg/agent.py is a name match
        self.assertIn("README.md", out)
        self.assertIn("Files whose name also matches", out)
        self.assertIn("pkg/agent.py", out)

    def test_regex_mode_skips_filename_matching(self):
        t, d = self._repo()
        out = t.run("search", {"pattern": "nonexistent.py", "regex": True})
        self.assertIn("No matches", out)
        self.assertNotIn("filename", out)


class FooterTest(unittest.TestCase):
    def _footer(self, started, n):
        import io
        from contextlib import redirect_stdout
        from agent_cli import agent
        buf = io.StringIO()
        # agent.console writes to its own stdout; capture via a temp Console
        from rich.console import Console
        old = agent.console
        agent.console = Console(file=buf, force_terminal=False, width=80)
        try:
            agent.Agent._print_footer(started, n)
        finally:
            agent.console = old
        return buf.getvalue().strip()

    def test_seconds_and_plural(self):
        import time
        self.assertIn("3 tool calls", self._footer(time.monotonic() - 4.2, 3))

    def test_singular_tool(self):
        import time
        self.assertIn("1 tool call", self._footer(time.monotonic() - 1, 1))
        self.assertNotIn("calls", self._footer(time.monotonic() - 1, 1))

    def test_no_tools_no_count(self):
        import time
        out = self._footer(time.monotonic() - 1, 0)
        self.assertIn("done in", out)
        self.assertNotIn("tool", out)

    def test_minutes_rollover(self):
        import time
        self.assertIn("m", self._footer(time.monotonic() - 120, 0))


class ToolColorTest(unittest.TestCase):
    def test_writes_and_commands_differ_from_reads(self):
        from agent_cli.agent import Agent
        c = Agent._TOOL_COLORS
        self.assertEqual(c.get("write_file"), "yellow")
        self.assertEqual(c.get("run_command"), "magenta")
        # reads fall through to the default (not in the map)
        self.assertNotIn("search", c)
        self.assertNotIn("read_file", c)


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
