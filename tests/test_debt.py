"""/debt: `shortcut: <limit>, <when to upgrade>` comments listed by file."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agent_cli.debt import find_shortcuts, format_debt, main, parse_marker


class TestParseMarker(unittest.TestCase):
    def test_comment_styles(self):
        for line in ("# shortcut: in-memory only, when we add a second worker",
                     "    // shortcut: in-memory only, when we add a second worker",
                     "/* shortcut: in-memory only, when we add a second worker */",
                     " * Shortcut: in-memory only, when we add a second worker",
                     "-- shortcut: in-memory only, when we add a second worker",
                     "<!-- shortcut: in-memory only, when we add a second worker -->"):
            self.assertEqual(parse_marker(line), ("in-memory only", "when we add a second worker"), line)

    def test_semicolon_separator_wins_over_comma(self):
        self.assertEqual(parse_marker("# shortcut: no retry, no backoff; when the API rate-limits"),
                         ("no retry, no backoff", "when the API rate-limits"))

    def test_missing_upgrade_part(self):
        self.assertEqual(parse_marker("x = 1  # shortcut: hard-coded locale"), ("hard-coded locale", ""))

    def test_not_a_marker(self):
        self.assertIsNone(parse_marker("shortcut: not in a comment"))
        self.assertIsNone(parse_marker("# keyboard shortcuts: ctrl+s"))


class TestFindShortcuts(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text(
            "def f():\n    # shortcut: O(n^2), when lists pass 1k items\n    pass\n")
        (self.root / "src" / "B.kt").write_text(
            "// shortcut: no paging, when the API returns >100 users\n"
            "fun g() = 1 // shortcut: fixed timeout\n")
        (self.root / "notes.txt").write_text("nothing here\n")
        (self.root / "blob.bin").write_bytes(b"\x00\x01# shortcut: binary, never")
        (self.root / "server.pem").write_text("# shortcut: blocked file, never\n")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "x.js").write_text("// shortcut: vendored, never\n")

    def tearDown(self):
        self.td.cleanup()

    def test_found_by_file_and_line(self):
        found = sorted((s.file, s.line, s.limit, s.upgrade) for s in find_shortcuts(self.root))
        self.assertEqual(found, [
            ("src/B.kt", 1, "no paging", "when the API returns >100 users"),
            ("src/B.kt", 2, "fixed timeout", ""),
            ("src/a.py", 2, "O(n^2)", "when lists pass 1k items"),
        ])

    def test_format_groups_by_file(self):
        text = format_debt(find_shortcuts(self.root))
        self.assertTrue(text.startswith("3 shortcuts in 2 files:"))
        self.assertLess(text.index("src/B.kt"), text.index("src/a.py"))
        self.assertIn("2  O(n^2)  ->  upgrade: when lists pass 1k items", text)
        self.assertIn("fixed timeout  ->  upgrade: (no upgrade trigger given)", text)

    def test_empty_project_explains_the_marker(self):
        self.assertIn("# shortcut: <limit>, <when to upgrade>", format_debt([]))

    def test_cli_main(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main([str(self.root)]), 0)
        self.assertIn("3 shortcuts in 2 files", buf.getvalue())


class TestDebtRegressions(unittest.TestCase):
    def test_markers_inside_strings_are_ignored(self):
        self.assertIsNone(parse_marker('    "# shortcut: in a string, never"'))
        self.assertIsNone(parse_marker("x.write_text('// shortcut: vendored, never\\n')"))
        self.assertEqual(parse_marker('s = "it\'s fine"  # shortcut: real, when x'), ("real", "when x"))

    def test_block_comment_close_mid_line(self):
        self.assertEqual(parse_marker("/* shortcut: fixed size, when >1k */ int n = 10;"),
                         ("fixed size", "when >1k"))

    def test_triple_quoted_docstrings_are_skipped(self):
        from agent_cli.debt import scan_text
        text = 'def f():\n    """Doc.\n\n    # shortcut: example, never\n    """\n    # shortcut: real, when y\n'
        self.assertEqual(scan_text(text), [(6, "real", "when y")])

    def test_this_repo_has_no_false_shortcuts(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual([(s.file, s.line) for s in find_shortcuts(root)], [])

    def test_no_secrets_warning_for_sensitive_files(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / ".env.example").write_text("# shortcut: x, y\n")
            (Path(td) / "a.py").write_text("# shortcut: real, later\n")
            with mock.patch("agent_cli.redact._report") as report:
                found = find_shortcuts(Path(td))
        report.assert_not_called()
        self.assertEqual([s.file for s in found], ["a.py"])


if __name__ == "__main__":
    unittest.main()
