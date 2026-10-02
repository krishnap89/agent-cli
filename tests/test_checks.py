"""Tests for Feature 1: Automatic checks after edits (lint + test loop)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_cli.checks import builtin_lint, custom_lint, run_tests, truncate_output
from agent_cli.config import Config
from agent_cli.coder import Coder
from tests.fake_server import FakeServer


class TestBuiltinLint(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_valid_python(self):
        f = self.root / "ok.py"
        f.write_text("x = 1\n")
        errs = builtin_lint(self.root, ["ok.py"])
        self.assertEqual(errs, [])

    def test_invalid_python(self):
        f = self.root / "bad.py"
        f.write_text("def foo(\n")
        errs = builtin_lint(self.root, ["bad.py"])
        self.assertEqual(len(errs), 1)
        self.assertIn("SyntaxError", errs[0])
        self.assertIn("bad.py", errs[0])

    def test_valid_json(self):
        f = self.root / "ok.json"
        f.write_text('{"a": 1}\n')
        errs = builtin_lint(self.root, ["ok.json"])
        self.assertEqual(errs, [])

    def test_invalid_json(self):
        f = self.root / "bad.json"
        f.write_text('{bad}\n')
        errs = builtin_lint(self.root, ["bad.json"])
        self.assertEqual(len(errs), 1)
        self.assertIn("JSONDecodeError", errs[0])

    def test_unknown_extension_no_check(self):
        f = self.root / "file.txt"
        f.write_text("anything\n")
        errs = builtin_lint(self.root, ["file.txt"])
        self.assertEqual(errs, [])


class TestCustomLint(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_success(self):
        result = custom_lint(self.root, ["a.py"], "true")
        self.assertIsNone(result)

    def test_failure(self):
        result = custom_lint(self.root, ["a.py"], "echo 'error in a.py' && exit 1")
        self.assertIsNotNone(result)
        self.assertIn("error in a.py", result)

    def test_files_placeholder(self):
        (self.root / "a.py").write_text("")
        (self.root / "b.py").write_text("")
        result = custom_lint(self.root, ["a.py", "b.py"], "echo {files}")
        # exit 0, so None
        self.assertIsNone(result)


class TestRunTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_passing(self):
        passed, output = run_tests(self.root, "true")
        self.assertTrue(passed)

    def test_failing(self):
        passed, output = run_tests(self.root, "echo FAIL && exit 1")
        self.assertFalse(passed)
        self.assertIn("FAIL", output)


class TestTruncateOutput(unittest.TestCase):
    def test_short(self):
        self.assertEqual(truncate_output("hello", 100), "hello")

    def test_long(self):
        text = "A" * 100
        result = truncate_output(text, 50)
        self.assertTrue(result.startswith("...(truncated)..."))
        self.assertTrue(result.endswith("A" * 50))


class TestCoderLintLoop(unittest.TestCase):
    """Integration tests for the lint+fix loop in Coder._send()."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        # Init git so all_files works
        os.system(f"cd {self.root} && git init -q && git config user.email test@test && git config user.name test")

    def tearDown(self):
        self.td.cleanup()

    def _make_coder(self, script):
        self.server = FakeServer(script)
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        coder = Coder(cfg, mode="code")
        return coder

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def test_syntax_error_triggers_fix(self):
        """An edit that introduces a syntax error triggers a fix attempt."""
        # Write a valid file first
        (self.root / "calc.py").write_text("x = 1\n")

        # First reply: introduce a syntax error
        bad_edit = """I'll fix calc.py.

calc.py
```python
<<<<<<< SEARCH
x = 1
=======
def foo(
>>>>>>> REPLACE
```"""
        # Second reply (fix): correct the syntax
        good_edit = """Let me fix that.

calc.py
```python
<<<<<<< SEARCH
def foo(
=======
def foo():
    pass
>>>>>>> REPLACE
```"""
        coder = self._make_coder([bad_edit, good_edit])
        coder.add(["calc.py"])
        coder.send("break it then fix it")

        # The file should have the corrected content
        content = (self.root / "calc.py").read_text()
        self.assertIn("def foo():", content)
        # Both requests should have been made
        self.assertEqual(len(self.server.requests), 2)

    def test_fix_attempt_limit(self):
        """When the model keeps breaking code, the loop stops after max_fix_attempts."""
        (self.root / "calc.py").write_text("x = 1\n")

        bad_edit = """calc.py
```python
<<<<<<< SEARCH
x = 1
=======
def foo(
>>>>>>> REPLACE
```"""
        # Model keeps returning bad code
        still_bad = """calc.py
```python
<<<<<<< SEARCH
def foo(
=======
def bar(
>>>>>>> REPLACE
```"""
        coder = self._make_coder([bad_edit, still_bad, still_bad])
        coder.cfg.max_fix_attempts = 2
        coder.add(["calc.py"])
        coder.send("break it")

        # Should have made 3 requests: original + 2 fix attempts
        self.assertEqual(len(self.server.requests), 3)

    def test_single_undo_reverts_all(self):
        """One /undo reverts both the original edit and fix edits."""
        (self.root / "calc.py").write_text("x = 1\n")

        bad_edit = """calc.py
```python
<<<<<<< SEARCH
x = 1
=======
def foo(
>>>>>>> REPLACE
```"""
        good_edit = """calc.py
```python
<<<<<<< SEARCH
def foo(
=======
def foo():
    pass
>>>>>>> REPLACE
```"""
        coder = self._make_coder([bad_edit, good_edit])
        coder.add(["calc.py"])
        coder.send("edit it")
        coder.undo()

        content = (self.root / "calc.py").read_text()
        self.assertEqual(content, "x = 1\n")

    def test_lint_disabled(self):
        """AGENT_LINT=0 disables all checks."""
        (self.root / "calc.py").write_text("x = 1\n")

        bad_edit = """calc.py
```python
<<<<<<< SEARCH
x = 1
=======
def foo(
>>>>>>> REPLACE
```"""
        coder = self._make_coder([bad_edit])
        coder.cfg.lint_enabled = False
        coder.add(["calc.py"])
        coder.send("break it")

        # No fix attempt — only 1 request
        self.assertEqual(len(self.server.requests), 1)
        content = (self.root / "calc.py").read_text()
        self.assertIn("def foo(", content)

    def test_ask_mode_no_checks(self):
        """In ask mode, no checks run."""
        (self.root / "calc.py").write_text("x = 1\n")
        coder = self._make_coder(["The answer is 42."])
        coder.add(["calc.py"])
        coder.send("what is x?", mode="ask")
        self.assertEqual(len(self.server.requests), 1)

    def test_custom_lint_cmd_receives_files(self):
        """AGENT_LINT_CMD with {files} receives the changed file paths."""
        (self.root / "calc.py").write_text("x = 1\n")
        (self.root / "check.sh").write_text("#!/bin/sh\necho \"checked: $@\"\nexit 0\n")
        os.chmod(self.root / "check.sh", 0o755)

        edit = """calc.py
```python
<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE
```"""
        coder = self._make_coder([edit])
        coder.cfg.lint_cmd = "echo {files}"
        coder.add(["calc.py"])
        coder.send("change x")
        # Should succeed with no fix attempts
        self.assertEqual(len(self.server.requests), 1)

    def test_auto_test_sends_failure_to_model(self):
        """With AGENT_AUTO_TEST=1, a failing test sends output to the model."""
        (self.root / "calc.py").write_text("x = 1\n")

        edit = """calc.py
```python
<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE
```"""
        fix_edit = """calc.py
```python
<<<<<<< SEARCH
x = 2
=======
x = 3
>>>>>>> REPLACE
```"""
        coder = self._make_coder([edit, fix_edit])
        coder.cfg.auto_test = True
        coder.cfg.test_cmd = "echo 'FAILED test_calc' && exit 1"
        coder.cfg.max_fix_attempts = 1
        coder.add(["calc.py"])
        coder.send("change x")

        # Should have 2 requests: original + 1 fix attempt from test failure
        self.assertEqual(len(self.server.requests), 2)
        # The fix feedback should contain the test output
        last_msgs = self.server.requests[1]["messages"]
        feedback = [m for m in last_msgs if m["role"] == "user" and "the tests" in m.get("content", "")]
        self.assertTrue(len(feedback) > 0)


if __name__ == "__main__":
    unittest.main()
