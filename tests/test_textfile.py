"""Edits keep a file's bytes outside the edit, and never touch non-UTF-8 files."""
import io
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli import coder, textfile, web
from agent_cli.config import Config
from agent_cli.docgen import reindent_kdoc_blocks
from agent_cli.micro import analysis, loop

LATIN1 = b"# caf\xe9\nx = 1\n"


def edit(path, old, new):
    return f"{path}\n```\n<<<<<<< SEARCH\n{old}=======\n{new}>>>>>>> REPLACE\n```\n"


class FileCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.out = io.StringIO()
        console = Console(file=self.out, width=300, color_system=None)
        for module in (coder, loop, analysis):
            self.patch(module, "console", console)
        self.patch(coder, "show", lambda *a, **k: None)

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, **settings):
        values = dict(notes_enabled=False, use_map=False, auto_approve=True, android="off",
                      lint_enabled=False, auto_test=False, test_cmd="", auto_model=False)
        values.update(settings)
        return Config(workdir=self.root, **values)

    def coder(self, *replies, files=(), **settings):
        made = coder.Coder(self.config(**settings))
        made.editable.update(dict.fromkeys(files))
        made._call_model = mock.Mock(side_effect=list(replies))
        return made

    def put(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


class KeepBytesTest(FileCase):
    def check_edit(self, before, after, old="x = 1\n", new="x = 9\n"):
        path = self.put("f.txt", before)
        c = self.coder(edit("f.txt", old, new), files=["f.txt"])
        c.send("change it")
        self.assertEqual(path.read_bytes(), after)
        c.undo()
        self.assertEqual(path.read_bytes(), before)

    def test_lf(self):
        self.check_edit(b"a\nx = 1\n", b"a\nx = 9\n")

    def test_crlf(self):
        self.check_edit(b"a\r\nx = 1\r\nb\r\n", b"a\r\nx = 9\r\nb\r\n")

    def test_crlf_added_lines(self):
        self.check_edit(b"a\r\nx = 1\r\n", b"a\r\nx = 9\r\ny = 2\r\n", new="x = 9\ny = 2\n")

    def test_crlf_removed_lines(self):
        self.check_edit(b"a\r\nx = 1\r\nb\r\n", b"a\r\nb\r\n", new="")

    def test_crlf_append(self):
        self.check_edit(b"a\r\n", b"a\r\nb\r\n", old="", new="b\n")

    def test_cr(self):
        self.check_edit(b"a\rx = 1\rb\r", b"a\rx = 9\rb\r")

    def test_mixed(self):
        self.check_edit(b"a\r\nx = 1\nb\r\nc\n", b"a\r\nx = 9\nb\r\nc\n")

    def test_mixed_with_an_added_line(self):
        self.check_edit(b"a\r\nb\nx = 1\r\nc\nd\r\n", b"a\r\nb\nx = 9\r\ny\r\nc\nd\r\n",
                        new="x = 9\ny\n")

    def test_no_final_newline(self):
        self.check_edit(b"x = 1\nlast", b"x = 9\nlast")

    def test_crlf_no_final_newline(self):
        self.check_edit(b"x = 1\r\nlast", b"x = 9\r\nlast")

    def test_append_without_final_newline(self):
        self.check_edit(b"a\r\nlast", b"a\r\nlast\r\nnext\r\n", old="", new="next\n")

    def test_bom(self):
        self.check_edit(b"\xef\xbb\xbfa\r\nx = 1\r\n", b"\xef\xbb\xbfa\r\nx = 9\r\n")

    def test_bom_when_the_first_line_is_replaced(self):
        self.check_edit(b"\xef\xbb\xbfx = 1\n", b"\xef\xbb\xbfx = 9\n", old="﻿x = 1\n")

    def test_utf8(self):
        self.check_edit("café ✓\r\nx = 1\r\n".encode(), "café ✓\r\nx = 9\r\n".encode())

    def test_loose_match_keeps_other_separators(self):
        for char in "\x0b\x0c\x1c\x85  ":
            self.check_edit(f"a{char}b\nx = 1 \n".encode(), f"a{char}b\nx = 9\n".encode())

    def test_separator_inside_the_edit(self):
        self.check_edit(b"x = 1\n", "x = 'a b'\n".encode(), new="x = 'a b'\n")

    def test_reply_with_crlf(self):
        path = self.put("f.txt", b"a\r\nx = 1\r\n")
        c = self.coder(edit("f.txt", "x = 1\n", "x = 9\n").replace("\n", "\r\n"), files=["f.txt"])
        c.send("change it")
        self.assertEqual(path.read_bytes(), b"a\r\nx = 9\r\n")

    def test_undo_restores_the_first_version(self):
        path = self.put("f.txt", b"a\r\nx = 1\nb\r\n")
        c = self.coder(edit("f.txt", "x = 1\n", "x = 2\n") + edit("f.txt", "a\nx = 2\n", "z\n"),
                       files=["f.txt"])
        c.send("twice")
        c.undo()
        self.assertEqual(path.read_bytes(), b"a\r\nx = 1\nb\r\n")

    def test_diff_shows_only_the_edit(self):
        self.put("f.txt", b"a\r\nx = 1\r\nb\r\n")
        c = self.coder(edit("f.txt", "x = 1\n", "x = 9\n"), files=["f.txt"])
        c.send("change it")
        lines = [l for l in c.last_diffs[0].splitlines()
                 if l[:1] in "+-" and not l.startswith(("---", "+++"))]
        self.assertEqual(lines, ["-x = 1", "+x = 9"])

    def test_random_edit_changes_one_line_only(self):
        rng = random.Random(1)
        for _ in range(3000):
            ends = rng.choice((["\n"], ["\r\n"], ["\r"], ["\n", "\r\n"], ["\n", "\r\n", "\r"]))
            words = [rng.choice(("same", "{", "}", "  indented", "é")) for _ in range(rng.randint(1, 12))]
            picked = [rng.choice(ends) for _ in words]
            if rng.random() < 0.3:
                picked[-1] = ""
            before = "".join(w + e for w, e in zip(words, picked)).encode()
            path = self.put("r.txt", before)
            self.assertEqual(textfile.encode(path, textfile.read(path)), before)
            i = rng.randrange(len(words))
            text = textfile.read(path).split("\n")
            text[i] = "changed"
            expected = "".join(("changed" if k == i else w) + e
                               for k, (w, e) in enumerate(zip(words, picked))).encode()
            self.assertEqual(textfile.encode(path, "\n".join(text)), expected, before)

    def test_whole_file_rewrite_of_mixed_endings(self):
        lines = [b"}\r\n" if i % 3 else b"}\n" for i in range(1500)]
        path = self.put("f.txt", b"first\n" + b"".join(lines) + b"last\r\n")
        text = textfile.read(path).replace("first\n", "first\nadded\n").replace("last", "changed")
        got = textfile.encode(path, text)
        self.assertTrue(got.endswith(b"".join(lines) + b"changed\r\n"))

    def test_large_mixed_file(self):
        before = b"}\r\n" * 10000 + b"x = 1\n" + b"}\r\n" * 10000
        self.check_edit(before, before.replace(b"x = 1", b"x = 9"))


class RefuseTest(FileCase):
    def test_add(self):
        for data in (LATIN1, "x = 1\n".encode("utf-16"), "x = 1\n".encode("utf-16-le"), b"\x89PNG\r\n"):
            self.put("f.txt", data)
            c = self.coder()
            c.add(["f.txt"])
            self.assertEqual(list(c.editable), [], data)
        self.assertIn("not UTF-8", self.out.getvalue())

    def test_read_only_still_works(self):
        self.put("f.txt", LATIN1)
        c = self.coder()
        c.add(["f.txt"], read_only=True)
        c.add(["f.txt"])
        self.assertEqual((list(c.editable), list(c.read_only)), ([], ["f.txt"]))

    def test_file_changed_after_add(self):
        path = self.put("f.txt", LATIN1)
        c = self.coder(edit("f.txt", "x = 1\n", "x = 9\n"), files=["f.txt"])
        c.send("change it")
        self.assertEqual(path.read_bytes(), LATIN1)
        self.assertEqual(c.undo_stack, [])
        self.assertEqual(c._call_model.call_count, 1, "asked the model to fix what can't be fixed")
        self.assertIn("not UTF-8", c.history[-2]["content"])

    def test_not_asked_about_a_file_outside_the_chat(self):
        path = self.put("f.txt", LATIN1)
        c = self.coder(edit("f.txt", "x = 1\n", "x = 9\n"), auto_approve=False)
        c._offer_mentioned_files = mock.Mock()
        with mock.patch.object(coder.Confirm, "ask") as ask:
            c.send("change it")
        ask.assert_not_called()
        self.assertEqual(path.read_bytes(), LATIN1)

    def test_other_files_in_the_reply_are_still_edited(self):
        bad = self.put("bad.txt", LATIN1)
        good = self.put("good.txt", b"x = 1\n")
        reply = edit("bad.txt", "x = 1\n", "x = 9\n") + edit("good.txt", "x = 1\n", "x = 9\n")
        c = self.coder(reply, files=["bad.txt", "good.txt"])
        c.send("change both")
        self.assertEqual((bad.read_bytes(), good.read_bytes()), (LATIN1, b"x = 9\n"))
        c.undo()
        self.assertEqual((bad.read_bytes(), good.read_bytes()), (LATIN1, b"x = 1\n"))

    def test_no_offer_to_resend_when_nothing_was_added(self):
        self.put("f.txt", LATIN1)
        c = self.coder("look at f.txt", auto_approve=False)
        c._all_files = ["f.txt"]
        with mock.patch.object(coder.Confirm, "ask", return_value=True) as ask:
            c.send("where")
        self.assertEqual(ask.call_count, 1)

    def test_unencodable_reply_leaves_the_file_whole(self):
        path = self.put("f.txt", b"x = 1\n")
        c = self.coder(edit("f.txt", "x = 1\n", "x = '\ud800'\n"), files=["f.txt"])
        c.send("change it")
        self.assertEqual(path.read_bytes(), b"x = 1\n")
        self.assertEqual(c.undo_stack, [])


class PipelineTest(FileCase):
    def test_micro_refuses_before_calling_the_model(self):
        path = self.put("f.txt", LATIN1)
        c = self.coder(files=["f.txt"])
        c.micro("change x in f.txt")
        c._call_model.assert_not_called()
        self.assertEqual(path.read_bytes(), LATIN1)

    def test_micro_refuses_a_located_file(self):
        path = self.put("f.txt", LATIN1)
        c = self.coder("f.txt")
        c.micro("change the value")
        self.assertEqual(c._call_model.call_count, 1)
        self.assertEqual(path.read_bytes(), LATIN1)

    def test_micro_keeps_crlf(self):
        path = self.put("f.txt", b"a\r\nx = 1\r\n")
        c = self.coder(edit("f.txt", "x = 1\n", "x = 9\n"), "VERDICT: OK", "Done.", files=["f.txt"])
        c.micro("change x in f.txt")
        self.assertEqual(path.read_bytes(), b"a\r\nx = 9\r\n")
        c.undo()
        self.assertEqual(path.read_bytes(), b"a\r\nx = 1\r\n")

    def test_doc_refuses(self):
        path = self.put("A.kt", b"// caf\xe9\nclass A\n")
        c = self.coder()
        c.doc_gen("A.kt")
        c._call_model.assert_not_called()
        self.assertEqual(path.read_bytes(), b"// caf\xe9\nclass A\n")

    def test_doc_keeps_crlf(self):
        before = b"package p\r\n\r\nclass A\r\n"
        after = b"package p\r\n\r\n/** Doc. */\r\nclass A\r\n"
        for reply in ("```kotlin\npackage p\n\n/** Doc. */\nclass A\n```",
                      edit("A.kt", "class A\n", "/** Doc. */\nclass A\n")):
            path = self.put("A.kt", before)
            c = self.coder(reply)
            c.doc_gen("A.kt")
            self.assertEqual(path.read_bytes(), after, reply)
            c.undo()
            self.assertEqual(path.read_bytes(), before, reply)

    def test_doc_revert_is_exact(self):
        before = b"class A\r\n// end\n"
        path = self.put("A.kt", before)
        self.coder("```kotlin\nclass B\n```").doc_gen("A.kt")
        self.assertEqual(path.read_bytes(), before)

    def test_reindent_keeps_other_separators(self):
        source = 'val s = "a b\x0c"\n/** Doc. */\nclass A\n'
        self.assertEqual(reindent_kdoc_blocks(source), source)

    def test_gen(self):
        for before, body, after in ((LATIN1, None, LATIN1),
                                    (b"class ATest\r\n", "class ATest2\n", b"class ATest2\r\n")):
            self.put("A.kt", b"class A\n")
            path = self.put("ATest.kt", before)
            replies = ["plan", edit("ATest.kt", "class ATest\n", body)] if body else []
            c = self.coder(*replies, testgen_plan=False)
            c.android_info = {"modules": {":app": {"path": ".", "android": True}}}
            stubs = dict(resolve_target=("A.kt", "A", "", ":app"), classify_class="plain",
                         detect_test_setup={"libs": set(), "dsl": "kts"}, check_required_libs=[],
                         test_file_path="ATest.kt", find_test_examples=[], find_test_helpers=[],
                         collect_context=("context", []), compile_test_task="t", run_test_task="t",
                         find_junit_xml=None, run_gradle=(True, ""), format_report="done")
            for name, value in stubs.items():
                self.patch(coder, name, mock.Mock(return_value=value))
            c.test_gen("A")
            self.assertEqual(path.read_bytes(), after)
            self.assertEqual(c._call_model.call_count, len(replies))

    def test_analysis_document(self):
        for before, after in ((b"old\r\n", b"new\r\n"), (LATIN1, LATIN1)):
            path = self.put("doc.md", before)
            c = self.coder()
            made = analysis.AnalysisOrchestrator(c)
            made._output_path = lambda request: "doc.md"
            made._write("explain", "new\n", ["a.py"])
            self.assertEqual(path.read_bytes(), after)
            c.undo()
            self.assertEqual(path.read_bytes(), before)


class WebTest(FileCase):
    def send(self, ui, reply):
        events = []
        with mock.patch.object(web, "open_stream"), \
                mock.patch.object(web, "iter_stream_events", return_value=[("content", reply)]):
            ui._do_send("change it", "code", events.append, lambda *a: None)
        return "".join(events)

    def test_keeps_crlf_and_undo_is_exact(self):
        before = b"a\r\nx = 1\r\n"
        path = self.put("f.txt", before)
        ui = web.WebCoder(self.config())
        self.send(ui, edit("f.txt", "x = 1\n", "x = 9\n"))
        self.assertEqual(path.read_bytes(), b"a\r\nx = 9\r\n")
        ui.undo()
        self.assertEqual(path.read_bytes(), before)

    def test_refuses_edit(self):
        path = self.put("f.txt", LATIN1)
        ui = web.WebCoder(self.config())
        self.assertIn("not UTF-8", self.send(ui, edit("f.txt", "x = 1\n", "x = 9\n")))
        self.assertEqual(path.read_bytes(), LATIN1)
        self.assertEqual(list(ui.coder.editable), [])

    def test_refuses_add_and_says_why(self):
        self.put("f.txt", LATIN1)
        ui = web.WebCoder(self.config())
        refused = []
        self.assertEqual(ui.add_files(["f.txt"], refused=refused), [])
        self.assertTrue(re.search("f.txt.*not UTF-8", refused[0]))
        self.assertEqual(ui.add_files(["f.txt"], read_only=True), ["f.txt"])


if __name__ == "__main__":
    unittest.main()
