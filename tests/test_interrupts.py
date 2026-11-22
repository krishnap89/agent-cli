"""An interrupted command can still be undone, and Ctrl-D at a question means no."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli import coder, input_reader, intent
from agent_cli.config import Config


def edit(path, old, new):
    return (f"{path}\n```\n<<<<<<< SEARCH\n{old}=======\n{new}>>>>>>> REPLACE\n```\n")


class InterruptCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.out = io.StringIO()
        self.console = Console(file=self.out, width=300, color_system=None)
        self.patch(coder, "console", self.console)
        self.patch(input_reader, "console", self.console)
        self.patch(coder, "show", lambda *a, **k: None)

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, rel, text="x = 1\n"):
        (self.root / rel).write_text(text)
        return self.root / rel

    def coder(self, *replies, files=(), **settings):
        values = dict(notes_enabled=False, use_map=False, auto_approve=True, android="off",
                      lint_enabled=True, lint_cmd="", auto_test=False, test_cmd="",
                      auto_model=False)
        values.update(settings)
        made = coder.Coder(Config(workdir=self.root, **values))
        made.editable.update(dict.fromkeys(files))
        made._call_model = mock.Mock(side_effect=list(replies))
        return made


class UndoTest(InterruptCase):
    def interrupted(self, action, *args):
        try:
            action(*args)
        except KeyboardInterrupt:
            return
        self.fail("not interrupted")

    def test_ctrl_c_during_lint(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), files=["f.py"])
        c.run_lint = mock.Mock(side_effect=KeyboardInterrupt)
        self.interrupted(c.send, "change x")
        self.assertEqual(path.read_text(), "x = 2\n")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_ctrl_c_during_tests(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), files=["f.py"],
                       auto_test=True, test_cmd="tests")
        with mock.patch.object(coder, "run_tests", side_effect=KeyboardInterrupt):
            self.interrupted(c.send, "change x")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_ctrl_c_after_a_fix_restores_the_original(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), edit("f.py", "x = 2\n", "x = 3\n"),
                       files=["f.py"])
        c.run_lint = mock.Mock(side_effect=[["bad"], KeyboardInterrupt])
        self.interrupted(c.send, "change x")
        self.assertEqual(len(c.undo_stack), 1)
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_ctrl_c_while_the_model_fixes_a_failed_block(self):
        path = self.write("f.py")
        reply = edit("f.py", "x = 1\n", "x = 2\n") + edit("f.py", "missing\n", "y\n")
        c = self.coder(reply, KeyboardInterrupt, files=["f.py"])
        self.interrupted(c.send, "change x")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_failed_request_after_a_partial_edit(self):
        for failure in (None, ""):
            path = self.write("f.py")
            reply = edit("f.py", "x = 1\n", "x = 2\n") + edit("f.py", "missing\n", "y\n")
            c = self.coder(reply, failure, files=["f.py"])
            c.send("change x")
            c.undo()
            self.assertEqual(path.read_text(), "x = 1\n", repr(failure))

    def test_ctrl_c_at_the_question_for_a_second_file(self):
        path = self.write("f.py")
        self.write("g.py")
        reply = edit("f.py", "x = 1\n", "x = 2\n") + edit("g.py", "x = 1\n", "x = 2\n")
        c = self.coder(reply, files=["f.py"], auto_approve=False)
        with mock.patch.object(coder.Confirm, "ask", side_effect=KeyboardInterrupt):
            self.interrupted(c.send, "change both")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_ctrl_c_after_creating_a_file(self):
        c = self.coder(edit("new.py", "", "x = 1\n"))
        c.run_lint = mock.Mock(side_effect=KeyboardInterrupt)
        self.interrupted(c.send, "create it")
        c.undo()
        self.assertFalse((self.root / "new.py").exists())

    def test_each_request_is_its_own_step(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), edit("f.py", "x = 2\n", "x = 3\n"),
                       files=["f.py"])
        c.send("one")
        c.run_lint = mock.Mock(side_effect=KeyboardInterrupt)
        self.interrupted(c.send, "two")
        c.undo()
        self.assertEqual(path.read_text(), "x = 2\n")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_ctrl_c_before_an_edit_keeps_the_previous_step(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), KeyboardInterrupt, files=["f.py"])
        c.send("one")
        self.interrupted(c.send, "two")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_micro_interrupted_in_its_checks(self):
        path = self.write("f.py")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), files=["f.py"])
        with mock.patch("agent_cli.micro.loop.Checker.check", side_effect=KeyboardInterrupt):
            self.interrupted(c.micro, "change x in f.py")
        self.assertEqual(path.read_text(), "x = 2\n")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")

    def test_test_gen_interrupted_in_compile(self):
        self.write("A.kt", "class A\n")
        c = self.coder("plan", edit("ATest.kt", "", "class ATest\n"), testgen_plan=False)
        c.android_info = {"modules": {":app": {"path": ".", "android": True}}}
        stubs = dict(resolve_target=("A.kt", "A", "", ":app"), classify_class="plain",
                     detect_test_setup={"libs": set(), "dsl": "kts"}, check_required_libs=[],
                     test_file_path="ATest.kt", find_test_examples=[], find_test_helpers=[],
                     collect_context=("context", []), compile_test_task="compileTests")
        for name, value in stubs.items():
            self.patch(coder, name, mock.Mock(return_value=value))
        self.patch(coder, "run_gradle", mock.Mock(side_effect=KeyboardInterrupt))
        self.interrupted(c.test_gen, "A")
        self.assertTrue((self.root / "ATest.kt").exists())
        c.undo()
        self.assertFalse((self.root / "ATest.kt").exists())

    def test_doc_interrupted_after_the_first_file(self):
        whole = "```kotlin\n/** Doc. */\nclass A\n```"
        blocks = edit("src/A.kt", "class A\n", "/** Doc. */\nclass A\n")
        (self.root / "src").mkdir()
        for reply in (whole, blocks):
            first = self.write("src/A.kt", "class A\n")
            self.write("src/B.kt", "class B\n")
            c = self.coder(reply, KeyboardInterrupt)
            self.interrupted(c.doc_gen, "src")
            self.assertIn("Doc.", first.read_text())
            c.undo()
            self.assertEqual(first.read_text(), "class A\n", reply)

    def test_doc_interrupted_before_writing_keeps_the_previous_step(self):
        path = self.write("f.py")
        self.write("A.kt", "class A\n")
        c = self.coder(edit("f.py", "x = 1\n", "x = 2\n"), KeyboardInterrupt, files=["f.py"])
        c.send("one")
        self.interrupted(c.doc_gen, "A.kt")
        c.undo()
        self.assertEqual(path.read_text(), "x = 1\n")


class CtrlDTest(InterruptCase):
    def setUp(self):
        super().setUp()
        self.ask = mock.patch("rich.console.Console.input", side_effect=EOFError).start()
        self.addCleanup(mock.patch.stopall)
        self.tty = mock.patch("sys.stdin.isatty", return_value=True).start()

    def test_create_question_is_no(self):
        self.coder().add(["new.py"])
        self.assertFalse((self.root / "new.py").exists())

    def test_default_yes_is_still_no(self):
        self.write("f.py")
        c = self.coder(auto_approve=False)
        blocks, _ = coder.parse_edit_blocks(edit("f.py", "x = 1\n", "x = 2\n"))
        self.assertEqual(len(c._apply(blocks, {})), 1)
        self.assertEqual((self.root / "f.py").read_text(), "x = 1\n")

    def test_run_output_question_is_no(self):
        c = self.coder()
        c.run("echo hi; false")
        self.assertEqual(c.history, [])

    def test_detected_command_question_is_no(self):
        pipeline = mock.Mock()
        found = intent.Intent(intent.DOC_GEN, "A")
        cfg = Config(workdir=self.root, intent_routing=True, auto_approve=False)
        with mock.patch.object(intent, "looks_like_candidate", return_value=True), \
                mock.patch.object(intent, "classify", return_value=found):
            self.assertFalse(intent.route(cfg, pipeline, "document A", self.console))
        pipeline.doc_gen.assert_not_called()

    def test_repl_goes_on(self):
        typed = mock.Mock(side_effect=["/run echo hi", "/run touch after", EOFError])
        self.patch(coder, "read_message", typed)
        self.patch(coder, "set_completer", mock.Mock())
        self.patch(coder, "Folders", mock.Mock(return_value=mock.Mock(
            label="here", is_missing=lambda: False)))
        self.patch(coder, "Config", lambda **given: Config(
            notes_enabled=False, use_map=False, android="off", **given))
        coder.main(files=None, read=None, message=None, micro=None, ask=False, yes=False,
                   model=None, base_url=None, workdir=self.root, init_notes=False, review=None,
                   test_gen=None, doc=None, doc_update=False, version=False)
        self.assertTrue((self.root / "after").exists())

    def test_piped_input_still_aborts(self):
        self.tty.return_value = False
        with self.assertRaises(EOFError):
            self.coder().run("echo hi")


if __name__ == "__main__":
    unittest.main()
