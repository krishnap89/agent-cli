"""Reviewer: new branching logic without a test change, /test-gen offer, checklist."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli.coder import Coder
from agent_cli.config import Config
from agent_cli.review import (REVIEW_SYSTEM, branching_note, branching_without_tests,
                              build_review_messages)


def _diff(path, added):
    body = "\n".join("+" + ln for ln in added)
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,1 +1,{len(added) + 1} @@\n context\n{body}\n")


class TestBranchingWithoutTests(unittest.TestCase):
    def test_counts_new_branches(self):
        d = _diff("app/Foo.kt", ["if (x > 0) {", "} else if (z) {", "when (s) {",
                                 "val q = c ? 1 : 2", "// if this is a comment"])
        self.assertEqual(branching_without_tests(d), [("app/Foo.kt", 4)])

    def test_python_and_swift_keywords(self):
        d = _diff("a.py", ["    if a:", "    elif b:", "    except ValueError:"])
        d += _diff("B.swift", ["guard let x else { return }", "switch v {", "case .a:"])
        self.assertEqual(branching_without_tests(d), [("a.py", 3), ("B.swift", 3)])

    def test_kotlin_nullable_types_are_not_ternaries(self):
        d = _diff("a.kt", ["fun f(a: String?, b: Int): Int? = null", "val m: Map<String, Int?> = x"])
        self.assertEqual(branching_without_tests(d), [])

    def test_a_test_change_silences_the_flag(self):
        d = _diff("app/Foo.kt", ["if (x) y()"]) + _diff("app/src/test/FooTest.kt", ["@Test fun t() {}"])
        self.assertEqual(branching_without_tests(d), [])
        d = _diff("a.py", ["if x:"]) + _diff("tests/test_a.py", ["def test_x(): pass"])
        self.assertEqual(branching_without_tests(d), [])

    def test_untracked_new_file_is_checked(self):
        material = "new file: svc.py\n1 def f(x):\n2     if x:\n3         return 1\n"
        self.assertEqual(branching_without_tests(material), [("svc.py", 1)])

    def test_non_source_files_and_whole_file_reviews_are_ignored(self):
        self.assertEqual(branching_without_tests(_diff("README.md", ["if you want"])), [])
        self.assertEqual(branching_without_tests("app.py\n1 if x:\n2   pass"), [])

    def test_removed_lines_do_not_count(self):
        d = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,1 @@\n-if x:\n-    y()\n+y()\n"
        self.assertEqual(branching_without_tests(d), [])

    def test_note_and_prompt_fact(self):
        d = _diff("a.py", ["if a:", "if b:"])
        note = branching_note(branching_without_tests(d))
        self.assertEqual(note, "New branching logic without a test change: a.py (+2 branches).")
        user = build_review_messages(d, "x")[1]["content"]
        self.assertIn("Fact (from the diff): " + note, user)
        self.assertNotIn("Fact", build_review_messages(_diff("a.py", ["y = 1"]), "x")[1]["content"])

    def test_checklist_has_unnecessary_code(self):
        self.assertIn("Unnecessary code: anything that could be deleted", REVIEW_SYSTEM)


class TestOfferTestGen(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.coder = Coder(Config(workdir=Path(self.td.name), use_map=False), mode="code")
        self.buf = io.StringIO()
        self.patch = mock.patch("agent_cli.coder.console", Console(file=self.buf, width=300))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.td.cleanup()

    def test_nothing_flagged_prints_nothing(self):
        self.coder._offer_test_gen([])
        self.assertEqual(self.buf.getvalue(), "")

    def test_non_android_gets_a_hint_only(self):
        with mock.patch.object(self.coder, "test_gen") as tg:
            self.coder._offer_test_gen([("a.py", 2)])
        tg.assert_not_called()
        self.assertIn("New branching logic without a test change", self.buf.getvalue())
        self.assertIn("Consider adding tests", self.buf.getvalue())

    def test_android_kotlin_offers_and_runs_test_gen(self):
        self.coder.android_info = {"modules": {}}
        with mock.patch.object(self.coder, "test_gen") as tg, \
                mock.patch("agent_cli.coder.Confirm.ask", return_value=True) as ask:
            self.coder._offer_test_gen([("app/src/main/java/LoginViewModel.kt", 3)])
        self.assertIn("/test-gen LoginViewModel", ask.call_args[0][0])
        tg.assert_called_once_with("LoginViewModel")

    def test_declined_offer_does_not_run(self):
        self.coder.android_info = {"modules": {}}
        with mock.patch.object(self.coder, "test_gen") as tg, \
                mock.patch("agent_cli.coder.Confirm.ask", return_value=False):
            self.coder._offer_test_gen([("Foo.kt", 1)])
        tg.assert_not_called()

    def test_auto_approve_only_suggests(self):
        self.coder.android_info = {"modules": {}}
        self.coder.cfg.auto_approve = True
        with mock.patch.object(self.coder, "test_gen") as tg:
            self.coder._offer_test_gen([("Foo.kt", 1)])
        tg.assert_not_called()
        self.assertIn("Generate tests with: /test-gen Foo", self.buf.getvalue())


class TestReviewRegressions(unittest.TestCase):
    def test_soft_keywords_used_as_names_are_not_branches(self):
        d = _diff("a.py", ["match = RE.search(t)", "case = 1", "when.x = 2", "matches += 1"])
        self.assertEqual(branching_without_tests(d), [])

    def test_soft_keywords_as_statements_still_count(self):
        d = _diff("a.py", ["match command:", "    case 'go':"])
        d += _diff("A.kt", ["when (x) {", "when {"])
        d += _diff("B.swift", ["case .a:"])
        self.assertEqual(branching_without_tests(d), [("a.py", 2), ("A.kt", 2), ("B.swift", 1)])

    def test_split_chunk_uses_the_whole_diff_note(self):
        user = build_review_messages(_diff("Foo.kt", ["if (x) y()"]), "Foo.kt", branch_note="")
        self.assertNotIn("Fact", user[1]["content"])

    def test_coder_split_review_sees_tests_in_other_chunks(self):
        material = _diff("app/Foo.kt", ["if (x) y()"]) + _diff("app/src/test/FooTest.kt", ["@Test fun t() {}"])
        with tempfile.TemporaryDirectory() as td:
            coder = Coder(Config(workdir=Path(td), use_map=False), mode="code")
            sent = []
            with mock.patch.object(coder, "_call_model", side_effect=lambda m: sent.append(m) or "ok"):
                coder._review_split(material, "x")
        self.assertEqual(len(sent), 3)  # two chunks + merge
        self.assertFalse(any("Fact (from the diff)" in m[1]["content"] for m in sent[:2]))


if __name__ == "__main__":
    unittest.main()
