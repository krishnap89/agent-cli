"""Tests for @path mention expansion."""
import tempfile
import unittest
from pathlib import Path

from agent_cli.mentions import expand, find_mentions, MAX_FILE_CHARS


class TestFindMentions(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(find_mentions("explain @agent_cli/main.py"), ["agent_cli/main.py"])

    def test_at_start_of_line(self):
        self.assertEqual(find_mentions("@foo.py what is this"), ["foo.py"])

    def test_dedup_preserves_order(self):
        self.assertEqual(find_mentions("@a.py and @b.py and @a.py"), ["a.py", "b.py"])

    def test_email_is_not_a_mention(self):
        self.assertEqual(find_mentions("mail me at a@b.com"), [])

    def test_no_mentions(self):
        self.assertEqual(find_mentions("just a plain message"), [])


class TestExpand(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_no_mentions_returns_text_unchanged(self):
        out, notes = expand("plain message", self.root)
        self.assertEqual(out, "plain message")
        self.assertEqual(notes, [])

    def test_inlines_file_contents(self):
        self._write("foo.py", "print('hi')\n")
        out, notes = expand("explain @foo.py", self.root)
        self.assertIn("explain @foo.py", out)          # original kept
        self.assertIn("--- foo.py ---", out)            # block header
        self.assertIn("print('hi')", out)               # contents inlined
        self.assertEqual(notes, ["attached foo.py (12 chars)"])

    def test_empty_file_not_attached(self):
        self._write("agent_cli/__init__.py", "")
        out, notes = expand("explain @agent_cli/__init__.py", self.root)
        self.assertEqual(out, "explain @agent_cli/__init__.py")  # unchanged
        self.assertEqual(notes, ["agent_cli/__init__.py: empty, not attached"])

    def test_missing_file_not_attached(self):
        out, notes = expand("see @nope.py", self.root)
        self.assertEqual(out, "see @nope.py")
        self.assertEqual(notes, ["nope.py: not found, not attached"])

    def test_directory_not_attached(self):
        (self.root / "pkg").mkdir()
        _, notes = expand("see @pkg", self.root)
        self.assertEqual(notes, ["pkg: is a directory, not attached"])

    def test_path_escape_refused(self):
        _, notes = expand("see @../secret.txt", self.root)
        self.assertEqual(len(notes), 1)
        self.assertIn("outside", notes[0])

    def test_large_file_truncated(self):
        self._write("big.dat", "x" * (MAX_FILE_CHARS + 500))
        out, notes = expand("@big.dat", self.root)
        self.assertIn("truncated", notes[0])
        # only MAX_FILE_CHARS of the file's 'x' content survive (the longest
        # unbroken run of x in the output is exactly the capped content).
        longest = max((len(r) for r in out.split() if set(r) == {"x"}), default=0)
        self.assertEqual(longest, MAX_FILE_CHARS)

    def test_two_files_both_attached(self):
        self._write("a.py", "AAA")
        self._write("b.py", "BBB")
        out, notes = expand("@a.py @b.py", self.root)
        self.assertIn("AAA", out)
        self.assertIn("BBB", out)
        self.assertEqual(len(notes), 2)


if __name__ == "__main__":
    unittest.main()
