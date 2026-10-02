"""Tests for Feature 3: /review command."""
import os
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

from rich.prompt import Confirm

from agent_cli.config import Config
from agent_cli.coder import Coder
from agent_cli.review import (
    collect_material, split_material_by_file, build_review_messages,
    build_merge_messages, needs_split, _extract_files_from_diff,
)
from tests.fake_server import FakeServer


def _git_init(root: Path) -> None:
    """Initialize a git repo with an initial commit."""
    os.system(f"cd {root} && git init -q && git config user.email test@test "
              f"&& git config user.name test && git commit --allow-empty -m init -q")


class TestCollectMaterial(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_uncommitted_tracked_and_untracked(self):
        """Default /review collects both modified tracked and new untracked files."""
        _git_init(self.root)
        # Create and commit a file
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        # Modify it
        (self.root / "a.py").write_text("x = 2\n")
        # Add an untracked file
        (self.root / "b.py").write_text("y = 1\n")

        material, files, desc = collect_material(self.root)
        self.assertIn("a.py", files)
        self.assertIn("b.py", files)
        self.assertIn("x = 2", material)
        self.assertIn("new file: b.py", material)
        self.assertIn("uncommitted", desc)

    def test_staged(self):
        _git_init(self.root)
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")
        os.system(f"cd {self.root} && git add a.py")

        material, files, desc = collect_material(self.root, "staged")
        self.assertIn("a.py", files)
        self.assertIn("staged", desc)

    def test_branch(self):
        _git_init(self.root)
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        os.system(f"cd {self.root} && git checkout -q -b feature")
        (self.root / "a.py").write_text("x = 2\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'change a'")

        material, files, desc = collect_material(self.root, "main")
        self.assertIn("a.py", files)
        self.assertIn("main", desc)

    def test_file_targets(self):
        """Review specific files without git."""
        (self.root / "app.py").write_text("print('hello')\n")
        material, files, desc = collect_material(self.root, "app.py")
        self.assertEqual(files, ["app.py"])
        self.assertIn("print('hello')", material)

    def test_glob_targets(self):
        (self.root / "a.py").write_text("x = 1\n")
        (self.root / "b.py").write_text("y = 2\n")
        material, files, desc = collect_material(self.root, "*.py")
        self.assertIn("a.py", files)
        self.assertIn("b.py", files)

    def test_no_changes(self):
        _git_init(self.root)
        with self.assertRaises(ValueError) as ctx:
            collect_material(self.root)
        self.assertIn("No changes", str(ctx.exception))

    def test_not_git_repo(self):
        with self.assertRaises(ValueError) as ctx:
            collect_material(self.root)
        self.assertIn("Not a git repository", str(ctx.exception))

    def test_unknown_target(self):
        _git_init(self.root)
        with self.assertRaises(ValueError) as ctx:
            collect_material(self.root, "nonexistent_branch_xyz")
        self.assertIn("Unknown review target", str(ctx.exception))


class TestSplitMaterial(unittest.TestCase):
    def test_split_diff(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-y = 1\n"
            "+y = 2\n"
        )
        chunks = split_material_by_file(diff)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], "a.py")
        self.assertEqual(chunks[1][0], "b.py")

    def test_split_with_new_file(self):
        material = (
            "diff --git a/a.py b/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
            "\n"
            "new file: b.py\n"
            "1 y = 1\n"
        )
        chunks = split_material_by_file(material)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1][0], "b.py")


class TestNeedsSplit(unittest.TestCase):
    def test_small_fits(self):
        self.assertFalse(needs_split("x" * 100, 1000))

    def test_large_splits(self):
        self.assertTrue(needs_split("x" * 600, 1000))


class TestBuildMessages(unittest.TestCase):
    def test_review_messages(self):
        msgs = build_review_messages("diff content", "test desc")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("expert code reviewer", msgs[0]["content"])
        self.assertIn("diff content", msgs[1]["content"])

    def test_merge_messages(self):
        msgs = build_merge_messages(["finding 1", "finding 2"])
        self.assertEqual(len(msgs), 2)
        self.assertIn("finding 1", msgs[1]["content"])
        self.assertIn("finding 2", msgs[1]["content"])


class TestCoderReviewIntegration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _git_init(self.root)

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def _make_coder(self, script):
        # type: (List[str]) -> Coder
        self.server = FakeServer(script)
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        return Coder(cfg, mode="code")

    def test_review_sends_diff_and_untracked(self):
        """Review of uncommitted changes sends both tracked diff and untracked file."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")
        (self.root / "b.py").write_text("y = 1\n")

        review_reply = "1. [low] a.py:1 - Variable change\n   Problem: Minor.\n   Fix: Ok.\n\nLooks good."
        coder = self._make_coder([review_reply])

        with patch.object(Confirm, "ask", return_value=False):
            coder.review()

        # Check that the model received a review request
        self.assertEqual(len(self.server.requests), 1)
        sent = self.server.requests[0]["messages"]
        # System prompt should be the review prompt
        self.assertIn("expert code reviewer", sent[0]["content"])
        # User message should contain both files
        user_content = sent[1]["content"]
        self.assertIn("a.py", user_content)
        self.assertIn("b.py", user_content)

    def test_review_no_edits_even_if_model_returns_blocks(self):
        """Review never applies edits, even if the model returns SEARCH/REPLACE."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")

        # Model returns a SEARCH/REPLACE block in the review
        bad_reply = ("a.py\n```python\n<<<<<<< SEARCH\nx = 2\n=======\n"
                     "x = 3\n>>>>>>> REPLACE\n```")
        coder = self._make_coder([bad_reply])

        with patch.object(Confirm, "ask", return_value=False):
            coder.review()

        # File should NOT have been changed
        self.assertEqual((self.root / "a.py").read_text(), "x = 2\n")

    def test_review_restores_mode(self):
        """Review restores the previous mode afterwards."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")

        coder = self._make_coder(["No issues found."])
        coder.mode = "code"

        with patch.object(Confirm, "ask", return_value=False):
            coder.review()

        self.assertEqual(coder.mode, "code")

    def test_review_stored_in_history(self):
        """After review, findings are in history for follow-up."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")

        review_text = "1. [high] a.py:1 - Bug\n   Problem: Logic error.\n   Fix: Change x."
        coder = self._make_coder([review_text])

        with patch.object(Confirm, "ask", return_value=False):
            coder.review()

        # History should contain the review
        has_review = any("Review of" in m["content"] for m in coder.history if m["role"] == "user")
        has_findings = any("Bug" in m["content"] for m in coder.history if m["role"] == "assistant")
        self.assertTrue(has_review)
        self.assertTrue(has_findings)

    def test_review_offers_to_add_files(self):
        """After review, offer to add reviewed files not yet in chat."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")

        coder = self._make_coder(["No issues found."])

        with patch.object(Confirm, "ask", return_value=True) as mock_ask:
            coder.review()

        # a.py should have been added
        self.assertIn("a.py", coder.editable)

    def test_review_staged(self):
        """Review staged changes uses git diff --cached."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        (self.root / "a.py").write_text("x = 2\n")
        os.system(f"cd {self.root} && git add a.py")

        coder = self._make_coder(["No issues found."])
        with patch.object(Confirm, "ask", return_value=False):
            coder.review("staged")

        self.assertEqual(len(self.server.requests), 1)
        self.assertIn("staged", self.server.requests[0]["messages"][1]["content"])

    def test_review_per_file_split(self):
        """Material larger than budget triggers per-file review + merge."""
        (self.root / "a.py").write_text("x = 1\n")
        os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
        # Make a.py have a big diff
        (self.root / "a.py").write_text("x = 2\n" + "# line\n" * 500)
        # Add another file
        (self.root / "b.py").write_text("y = 1\n" + "# line\n" * 500)
        os.system(f"cd {self.root} && git add b.py && git commit -q -m 'add b'")
        (self.root / "b.py").write_text("y = 2\n" + "# line\n" * 500)

        coder = self._make_coder([
            "1. [low] a.py:1 - Minor",  # per-file review a.py
            "1. [low] b.py:1 - Minor",  # per-file review b.py
            "1. [low] a.py:1 - Minor\n2. [low] b.py:1 - Minor",  # merged
        ])
        # Set a tiny budget to force splitting
        coder.cfg.context_chars = 200

        with patch.object(Confirm, "ask", return_value=False):
            coder.review()

        # Should have 3 requests: a.py review, b.py review, merge
        self.assertEqual(len(self.server.requests), 3)

    def test_review_files_without_git(self):
        """Review specific files works without git."""
        td2 = tempfile.TemporaryDirectory()
        root2 = Path(td2.name)
        (root2 / "app.py").write_text("print('hello')\n")

        self.server = FakeServer(["No issues found."])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=root2,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        coder = Coder(cfg, mode="code")
        with patch.object(Confirm, "ask", return_value=False):
            coder.review("app.py")

        self.assertEqual(len(self.server.requests), 1)
        self.assertIn("print('hello')", self.server.requests[0]["messages"][1]["content"])
        td2.cleanup()

    def test_review_nothing_to_review(self):
        """Clear message when there's nothing to review."""
        coder = self._make_coder([])
        # No changes in this fresh repo
        coder.review()
        # No requests should have been sent
        self.assertEqual(len(self.server.requests), 0)

    def test_review_not_git_repo_diff_target(self):
        """Clear error for diff target in non-git dir."""
        td2 = tempfile.TemporaryDirectory()
        root2 = Path(td2.name)
        self.server = FakeServer([])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=root2,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        coder = Coder(cfg, mode="code")
        coder.review("staged")
        self.assertEqual(len(self.server.requests), 0)
        td2.cleanup()


class TestExtractFilesFromDiff(unittest.TestCase):
    def test_extracts_files(self):
        diff = "+++ b/a.py\n+++ b/src/b.py\n"
        files = _extract_files_from_diff(diff)
        self.assertEqual(files, ["a.py", "src/b.py"])


if __name__ == "__main__":
    unittest.main()
