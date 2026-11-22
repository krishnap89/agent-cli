"""Tests for /cd: folder sessions, the recent list, and both REPLs."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text
from rich.cells import cell_len
from rich.console import Console

from agent_cli import coder, folders, input_reader, main, redact, review, testgen
from agent_cli.completer import AgentCompleter, CoderCompleter
from agent_cli.config import Config
from agent_cli.llm import Reply
from agent_cli.term import safe_text
from agent_cli.turns import prompt_header


class FolderCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = folders.real_folder(Path(tmp.name))
        self.store = self.tmp / "home" / "folders.json"
        self.patch(folders, "store_path", lambda: self.store)
        self.out = io.StringIO()
        self.console = Console(file=self.out, width=300, color_system=None, emoji=False)
        self.patch(folders, "console", self.console)
        self.a = self.folder("alpha")
        self.b = self.folder("beta")
        redact.reset_sessions()
        self.addCleanup(redact.reset_sessions)

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def folder(self, rel):
        path = self.tmp / rel
        path.mkdir(parents=True)
        return path

    def git(self, root, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                        "-c", "commit.gpgsign=false", *args],
                       cwd=root, check=True, capture_output=True)


class RecentListTest(FolderCase):
    def test_missing_file_is_empty(self):
        self.assertEqual(folders.load_recent(), [])

    def test_broken_file_is_empty(self):
        self.store.parent.mkdir()
        for data in (b"{not json", b"\xff\xfe\x00", b'{"a": 1}', b"null"):
            self.store.write_bytes(data)
            self.assertEqual(folders.load_recent(), [], data)

    def test_bad_entries_are_skipped(self):
        self.store.parent.mkdir()
        self.store.write_text(json.dumps([str(self.a), 3, "relative", str(self.a), str(self.b)]))
        self.assertEqual(folders.load_recent(), [str(self.a), str(self.b)])

    def test_newest_first_without_duplicates(self):
        for path in (self.a, self.b, self.a):
            folders.save_recent(path)
        self.assertEqual(folders.load_recent(), [str(self.a), str(self.b)])

    def test_keeps_ten(self):
        for i in range(13):
            folders.save_recent(self.tmp / f"f{i}")
        saved = json.loads(self.store.read_text())
        self.assertEqual(len(saved), 10)
        self.assertEqual(saved[0], str(self.tmp / "f12"))

    def test_unwritable_store_only_warns(self):
        self.store.parent.write_text("a file where the folder should be")
        folders.save_recent(self.a)
        self.assertIn("Could not save", self.out.getvalue())


class CdTest(FolderCase):
    def setUp(self):
        super().setUp()
        self.created = []
        self.first = SimpleNamespace(cfg=Config(workdir=self.a))
        self.folders = folders.Folders(Config(workdir=self.a), self.first, self.create)

    def create(self, cfg):
        self.created.append(cfg.workdir)
        return SimpleNamespace(cfg=cfg)

    def cd(self, arg):
        self.folders.cd(str(arg))
        return self.folders.root

    def test_start_folder_is_saved(self):
        self.assertEqual(folders.load_recent(), [str(self.a)])

    def test_absolute_and_relative_paths(self):
        self.assertEqual(self.cd(self.b), self.b)
        self.assertEqual(self.cd("../alpha"), self.a)

    def test_home(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.b)}):
            self.assertEqual(self.cd("~"), self.b)

    def test_spaces_quotes_and_escapes(self):
        spaced = self.folder("my app")
        for arg in ("../my app", '"../my app"', "'../my app'", "../my\\ app"):
            self.cd(self.a)
            self.assertEqual(self.cd(arg), spaced, arg)

    def test_apostrophe(self):
        odd = self.folder("it's")
        self.assertEqual(self.cd("../it's"), odd)

    def test_one_session_per_folder(self):
        second = self.folders.cd(str(self.b))
        self.assertIs(self.folders.cd(str(self.a)), self.first)
        self.assertIs(self.folders.cd(str(self.b)), second)
        self.assertEqual(self.created, [self.b])

    def test_symlink_is_the_same_session(self):
        link = self.tmp / "link"
        link.symlink_to(self.a, target_is_directory=True)
        self.assertIsNone(self.folders.cd(str(link)))
        self.assertEqual(self.created, [])

    @unittest.skipUnless(sys.platform == "darwin", "needs F_GETPATH")
    def test_other_case_is_the_same_session(self):
        typed = self.tmp / "ALPHA"
        if not typed.is_dir():
            self.skipTest("case-sensitive volume")
        self.assertIsNone(self.folders.cd(str(typed)))
        self.assertEqual(self.cd(self.tmp / "BETA"), self.b)

    def test_no_args_lists_and_stays(self):
        self.cd(self.b)
        self.assertIsNone(self.folders.cd(""))
        shown = self.out.getvalue()
        self.assertIn("*  1. beta", shown)
        self.assertIn("   2. alpha", shown)
        self.assertEqual(self.folders.root, self.b)

    def test_list_starts_with_current_folder(self):
        self.cd("")
        self.assertTrue(self.out.getvalue().startswith(f"Current: {self.a}\n"))

    def test_current_shown_when_pushed_off_the_list(self):
        for i in range(10):
            folders.save_recent(self.folder(f"f{i}"))
        self.cd("")
        self.assertIn(f"Current: {self.a}\n", self.out.getvalue())
        self.assertNotIn("*", self.out.getvalue())

    def test_current_shown_without_a_list(self):
        self.store.unlink()
        self.cd("")
        self.assertIn(f"Current: {self.a}\n", self.out.getvalue())
        self.assertIn("No recent folders", self.out.getvalue())

    def test_current_marked_when_deleted(self):
        self.a.rmdir()
        self.cd("")
        self.assertIn(f"Current: {self.a} (missing)", self.out.getvalue())

    def test_interrupt_at_any_step_does_not_half_switch(self):
        for target, name in ((redact, "startup_notice"), (folders, "_warn_parent_rules"),
                             (folders, "save_recent")):
            with mock.patch.object(target, name, side_effect=KeyboardInterrupt):
                try:
                    self.cd(self.b)
                except KeyboardInterrupt:
                    pass
            self.assertEqual((self.folders.root, self.folders.active), (self.a, self.first), name)

    def test_number(self):
        self.cd(self.b)
        self.assertEqual(self.cd("2"), self.a)

    def test_numbers_follow_the_list_on_screen(self):
        c = self.folder("gamma")
        self.cd(self.b)
        self.cd(c)
        self.cd("")  # gamma, beta, alpha
        self.assertEqual(self.cd("3"), self.a)
        self.assertEqual(self.cd("2"), self.b)
        self.assertEqual(self.cd("1"), c)

    def test_number_not_in_list_is_a_path(self):
        year = self.folder("alpha/2024")
        self.assertEqual(self.cd("2024"), year)

    def test_number_comes_before_path(self):
        self.folder("beta/2")
        self.cd(self.b)
        self.assertEqual(self.cd("2"), self.a)

    def test_dot_slash_number_is_a_path(self):
        two = self.folder("alpha/1")
        self.assertEqual(self.cd("./1"), two)

    def test_number_of_deleted_folder(self):
        self.cd(self.b)
        self.cd(self.a)
        self.b.rmdir()
        self.assertEqual(self.cd("2"), self.a)
        self.assertIn("no longer exists", self.out.getvalue())

    def test_deleted_folder_is_marked(self):
        self.cd(self.b)
        self.cd(self.a)
        self.b.rmdir()
        self.cd("")
        self.assertIn("(missing)", self.out.getvalue())

    def test_name_from_saved_list(self):
        other = self.folder("elsewhere/billing")
        folders.save_recent(other)
        self.assertEqual(self.cd("billing"), other)

    def test_name_ignores_case(self):
        other = self.folder("elsewhere/Billing")
        folders.save_recent(other)
        self.assertEqual(self.cd("billing"), other)

    def test_name_of_session_that_left_the_list(self):
        self.cd(self.b)
        for i in range(11):
            folders.save_recent(self.folder(f"f{i}"))
        self.assertEqual(self.cd("alpha"), self.a)

    def test_path_comes_before_name(self):
        folders.save_recent(self.folder("elsewhere/src"))
        local = self.folder("alpha/src")
        self.assertEqual(self.cd("src"), local)

    def test_ambiguous_name_stays(self):
        for parent in ("one", "two"):
            folders.save_recent(self.folder(f"{parent}/api"))
        self.assertEqual(self.cd("api"), self.a)
        self.assertIn("Several folders", self.out.getvalue())

    def test_bad_targets_stay(self):
        (self.a / "file.txt").write_text("x")
        for arg in ("nope", "file.txt", "a\0b", "0", "'unclosed"):
            self.assertEqual(self.cd(arg), self.a, arg)
        self.assertEqual(self.created, [])

    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0, "needs permissions")
    def test_unreadable_folder_stays(self):
        self.b.chmod(0)
        self.addCleanup(self.b.chmod, 0o755)
        self.assertEqual(self.cd(self.b), self.a)
        self.assertIn("permission denied", self.out.getvalue())

    def test_failed_session_stays(self):
        self.folders._create = mock.Mock(side_effect=PermissionError("no"))
        self.assertEqual(self.cd(self.b), self.a)
        self.assertIs(self.folders.active, self.first)
        self.assertEqual(folders.load_recent(), [str(self.a)])

    def test_interrupted_session_stays(self):
        self.folders._create = mock.Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.cd(self.b)
        self.assertIs(self.folders.active, self.first)

    def test_model_and_autotest_follow(self):
        self.first.cfg.model, self.first.cfg.auto_test = "m2", True
        second = self.folders.cd(str(self.b))
        self.assertEqual((second.cfg.model, second.cfg.auto_test), ("m2", True))
        second.cfg.model = "m3"
        self.cd(self.a)
        self.assertEqual(self.first.cfg.model, "m3")

    def test_folder_setting_does_not_follow(self):
        self.first.cfg.compile_enabled = "on"
        second = self.folders.cd(str(self.b))
        self.assertEqual(second.cfg.compile_enabled, Config().compile_enabled)

    def test_strict_queue_is_per_folder(self):
        redact.note_untrusted("a" * 600)
        self.cd(self.b)
        self.assertEqual(redact.swap_pending([]), [])
        self.cd(self.a)
        self.assertEqual(redact.swap_pending([]), ["a" * 600])

    def test_warns_about_parent_rules(self):
        (self.a / ".agentignore").write_text("secrets/\n")
        self.folder("alpha/src")
        self.cd("src")
        self.assertIn("does not apply", self.out.getvalue())

    def test_is_missing(self):
        self.cd(self.b)
        self.assertFalse(self.folders.is_missing())
        self.b.rmdir()
        self.assertTrue(self.folders.is_missing())


class ReplCase(FolderCase):
    module = None

    def setUp(self):
        super().setUp()
        self.patch(self.module, "console", self.console)
        self.completers = []
        self.patch(self.module, "set_completer", self.completers.append)
        self.prompts = []
        self.sent = []

    def chat(self, cfg, messages, **kwargs):
        self.sent.append((cfg, [dict(m) for m in messages]))
        return Reply(content=f"answer {len(self.sent)}", reasoning="", finish_reason="stop")

    def read(self, lines):
        """A read_message that types `lines`; a callable runs, then the next is typed."""
        queue = list(lines)

        def read_message(prompt, erase=False):
            self.prompts.append(Console(file=io.StringIO()).render_str(prompt).plain)
            while queue:
                line = queue.pop(0)
                if not callable(line):
                    return line
                line()
            raise EOFError
        return read_message

    def cfg(self, **kwargs):
        defaults = dict(notes_enabled=False, intent_routing=False, use_map=True,
                        show_thinking=False, tool_format="xml", android="auto",
                        test_cmd="", auto_test=False, compile_enabled="")
        defaults.update(kwargs)
        return lambda **given: Config(**dict(defaults, **given))


class CoderReplTest(ReplCase):
    module = coder

    def repl(self, *lines, ask=False, message=None, **cfg):
        with mock.patch.object(coder, "Config", self.cfg(**cfg)), \
                mock.patch.object(coder, "read_message", self.read(lines)), \
                mock.patch.object(coder, "chat", self.chat), \
                mock.patch.object(coder, "show", lambda *a, **k: None), \
                mock.patch.object(coder.Confirm, "ask", return_value=False):
            coder.main(files=None, read=None, message=message, micro=None, ask=ask, yes=False,
                       model=None, base_url=None, workdir=self.a, init_notes=False, review=None,
                       test_gen=None, doc=None, doc_update=False, version=False)
        return self.out.getvalue()

    def test_prompt_shows_folder(self):
        self.repl(f"/cd {self.b}")
        self.assertEqual(self.prompts, ["code alpha (0 files) › ", "code beta (0 files) › "])

    def test_files_and_mode_are_per_folder(self):
        (self.a / "x.py").write_text("x = 1\n")
        self.repl("/add x.py", "/mode ask", f"/cd {self.b}", "/cd alpha")
        self.assertEqual(self.prompts[-2:], ["code beta (0 files) › ", "ask alpha (1 file) › "])

    def test_new_folder_starts_in_launch_mode(self):
        self.repl("/mode code", f"/cd {self.b}", ask=True)
        self.assertEqual(self.prompts[-1], "ask beta (0 files) › ")

    def test_history_is_per_folder(self):
        self.repl("first question", f"/cd {self.b}", "second question",
                  "/cd alpha", "third question")
        texts = [json.dumps(messages) for _, messages in self.sent]
        self.assertNotIn("first question", texts[1])
        self.assertIn("first question", texts[2])
        self.assertNotIn("second question", texts[2])
        self.assertEqual([c.workdir for c, _ in self.sent], [self.a, self.b, self.a])

    def test_repo_map_is_per_folder(self):
        (self.a / "a.py").write_text("def only_alpha():\n    pass\n")
        (self.b / "b.py").write_text("def only_beta():\n    pass\n")
        self.repl("question", f"/cd {self.b}", "question")
        first, second = [json.dumps(messages) for _, messages in self.sent]
        self.assertIn("only_alpha", first)
        self.assertIn("only_beta", second)
        self.assertNotIn("only_alpha", second)

    def test_undo_is_per_folder(self):
        target = self.a / "x.py"
        target.write_text("new\n")
        made = []
        init = coder.Coder.__init__

        def with_undo(instance, *args, **kwargs):
            init(instance, *args, **kwargs)
            if not made:  # only the first folder has an edit to undo
                instance.undo_stack.append({"x.py": b"old\n"})
            made.append(instance)

        seen = []
        with mock.patch.object(coder.Coder, "__init__", with_undo):
            self.repl(f"/cd {self.b}", "/undo", lambda: seen.append(target.read_text()),
                      "/cd alpha", "/undo")
        self.assertEqual(seen, ["new\n"])
        self.assertEqual(target.read_text(), "old\n")

    def test_run_uses_active_folder(self):
        self.repl(f"/cd {self.b}", "/run touch here", "/cd alpha", "/run touch there")
        self.assertTrue((self.b / "here").exists())
        self.assertTrue((self.a / "there").exists())

    def test_test_uses_active_folder(self):
        self.repl(f"/cd {self.b}", "/test", test_cmd="touch tested")
        self.assertTrue((self.b / "tested").exists())
        self.assertFalse((self.a / "tested").exists())

    def test_lint_uses_active_folder(self):
        (self.b / "bad.json").write_text("{")
        self.assertIn("bad.json", self.repl(f"/cd {self.b}", "/lint bad.json"))

    def test_review_uses_active_folder(self):
        self.git(self.b, "init", "-q")
        (self.b / "fresh.py").write_text("value = 'beta'\n")
        self.repl(f"/cd {self.b}", "/review")
        self.assertIn("fresh.py", json.dumps(self.sent[0][1]))

    def test_notes_and_map_use_active_folder(self):
        (self.b / "AGENT.md").write_text("beta notes")
        (self.b / "b.py").write_text("def only_beta():\n    pass\n")
        out = self.repl(f"/cd {self.b}", "/notes", "/map", notes_enabled=True)
        self.assertIn("beta notes", out)
        self.assertIn("only_beta", out)

    def test_skills_use_active_folder(self):
        skill = self.folder("beta/.agent/skills/deploy")
        (skill / "SKILL.md").write_text("---\nname: deploy-beta\n---\nSteps\n")
        self.assertIn("deploy-beta", self.repl(f"/cd {self.b}", "/skills"))

    def test_model_is_shared(self):
        with mock.patch("agent_cli.llm.list_models", return_value=[]):
            self.repl("/model other", f"/cd {self.b}", "question")
        self.assertEqual(self.sent[0][0].model, "other")

    def test_android_compile_does_not_leak(self):
        def detect(root, mode):
            return {"modules": {}} if root == self.a else None
        with mock.patch.object(coder, "detect_android", detect):
            self.repl(f"/cd {self.b}", "question")
        self.assertEqual(self.sent[0][0].compile_enabled, "")

    def test_test_gen_usage_keeps_brackets(self):
        self.assertIn("<class|file> [method]", self.repl("/test-gen"))

    def test_completer_follows(self):
        self.repl(f"/cd {self.b}")
        self.assertEqual([c.coder.root for c in self.completers], [self.a, self.b])

    def test_deleted_folder_can_be_left(self):
        out = self.repl(f"/cd {self.b}", self.b.rmdir, "/run ls", "/cd alpha", "/run touch back")
        self.assertIn("no longer exists", out)
        self.assertTrue((self.a / "back").exists())

    def test_bracket_folder_can_add(self):
        odd = self.folder("app [old]")
        (odd / "x.py").write_text("x = 1\n")
        self.repl(f"/cd {odd}", "/add x.py")
        self.assertEqual(self.prompts[-1], "code app [old] (1 file) › ")

    def test_one_shot_is_not_saved(self):
        self.repl(message="question")
        self.assertEqual(folders.load_recent(), [])

    def test_does_not_change_process_folder(self):
        before = os.getcwd()
        self.repl(f"/cd {self.b}")
        self.assertEqual(os.getcwd(), before)


class AgentReplTest(ReplCase):
    module = main

    def repl(self, *lines, chat_only=False, prompt=None, **cfg):
        with mock.patch.object(main, "Config", self.cfg(**cfg)), \
                mock.patch.object(main, "read_message", self.read(lines)), \
                mock.patch("agent_cli.agent.console", self.console), \
                mock.patch("agent_cli.agent.chat", self.chat), \
                mock.patch("agent_cli.llm.chat", self.chat):
            main.main(prompt=prompt, yes=False, model=None, base_url=None, workdir=self.a,
                      check=False, chat_only=chat_only, show_map=None, init_notes=False,
                      review=None, version=False)
        return self.out.getvalue()

    def test_prompt_shows_folder_and_mode_is_per_folder(self):
        self.repl("/chat", f"/cd {self.b}", "/cd alpha")
        self.assertEqual(self.prompts, ["agent alpha › ", "chat alpha › ",
                                        "agent beta › ", "chat alpha › "])

    def test_new_folder_starts_in_launch_mode(self):
        self.repl("/agent", f"/cd {self.b}", chat_only=True)
        self.assertEqual(self.prompts[-1], "chat beta › ")

    def test_history_is_per_folder(self):
        self.repl("first question", f"/cd {self.b}", "second question",
                  "/cd alpha", "third question")
        texts = [json.dumps(messages) for _, messages in self.sent]
        self.assertNotIn("first question", texts[1])
        self.assertIn("first question", texts[2])
        self.assertNotIn("second question", texts[2])

    def test_map_is_per_folder(self):
        (self.a / "a.py").write_text("def only_alpha():\n    pass\n")
        self.repl("/map", f"/cd {self.b}", "question", "/cd alpha", "question")
        first, second = [messages[0]["content"] for _, messages in self.sent]
        self.assertNotIn("only_alpha", first)
        self.assertIn("only_alpha", second)

    def test_cd_does_not_index(self):
        with mock.patch("agent_cli.repomap.RepoMap.build") as build:
            self.repl(f"/cd {self.b}")
        build.assert_not_called()

    def test_review_uses_active_folder(self):
        self.git(self.b, "init", "-q")
        (self.b / "fresh.py").write_text("value = 'beta'\n")
        self.repl(f"/cd {self.b}", "/review")
        self.assertIn("fresh.py", json.dumps(self.sent[0][1]))

    def test_notes_and_mentions_use_active_folder(self):
        (self.b / "AGENT.md").write_text("beta notes")
        (self.b / "data.txt").write_text("beta data")
        self.repl(f"/cd {self.b}", "read @data.txt", notes_enabled=True)
        sent = json.dumps(self.sent[0][1])
        self.assertIn("beta notes", sent)
        self.assertIn("beta data", sent)

    def test_pinned_skill_is_per_folder(self):
        skill = self.folder("alpha/.agent/skills/deploy")
        (skill / "SKILL.md").write_text("---\nname: deploy\n---\nalpha steps\n")
        self.repl("/skill deploy", f"/cd {self.b}", "question", "/cd alpha", "question")
        first, second = [json.dumps(messages) for _, messages in self.sent]
        self.assertNotIn("alpha steps", first)
        self.assertIn("alpha steps", second)

    def test_pipeline_coder_is_per_folder(self):
        roots = []
        with mock.patch.object(coder.Coder, "doc_gen",
                               lambda instance, *a, **k: roots.append(instance.root)):
            self.repl("/doc A", f"/cd {self.b}", "/doc B", "/cd alpha", "/doc C")
        self.assertEqual(roots, [self.a, self.b, self.a])

    def test_chat_mode_never_runs_pipelines(self):
        with mock.patch("agent_cli.intent.route") as route, \
                mock.patch.object(coder.Coder, "micro") as micro:
            self.repl("/chat", "write tests for Login", "/micro change it")
        route.assert_not_called()
        micro.assert_not_called()

    def test_completer_follows(self):
        (self.b / "b.py").write_text("")
        self.repl(f"/cd {self.b}")
        self.assertEqual(self.completers[-1]._files_list(), ["b.py"])

    def test_header_path_follows(self):
        with mock.patch.object(main, "prompt_header", return_value="") as header:
            self.repl(f"/cd {self.b}", "/skills", "/cd alpha", "/skills")
        self.assertEqual([Path(c.kwargs["path"]).name for c in header.call_args_list],
                         ["alpha", "beta", "beta", "alpha"])

    def test_startup_help_keeps_brackets(self):
        out = self.repl()
        for usage in ("/cd [folder]", "/review [target]", "/map [query]", "/model [name]"):
            self.assertIn(usage, out)

    def test_test_gen_usage_keeps_brackets(self):
        self.assertIn("<class|file> [method]", self.repl("/test-gen"))

    def test_ctrl_c_in_any_command_keeps_the_repl(self):
        cases = {
            "/review": "agent_cli.main._do_review",
            "/debt": "agent_cli.debt.find_shortcuts",
            "/skills": "agent_cli.main.list_skills_display",
            "/notes": "agent_cli.main.load_notes",
            "/model x": "agent_cli.main.switch_model",
            "/grammar some text": "agent_cli.main.handle_writing",
            "/micro change it": "agent_cli.coder.Coder.micro",
            "/analyze explain it": "agent_cli.coder.Coder.analyze",
            "/test-gen A": "agent_cli.coder.Coder.test_gen",
            "/doc A": "agent_cli.coder.Coder.doc_gen",
            "/map": "agent_cli.agent.Agent.load_map",
            "/chat": "agent_cli.agent.Agent.set_mode",
            "/clear": "agent_cli.agent.Agent.reset",
            "a question": "agent_cli.agent.Agent.ask",
        }
        for command, target in cases.items():
            patcher = mock.patch(target)
            lines = [lambda: setattr(patcher.start(), "side_effect", KeyboardInterrupt),
                     command, patcher.stop, "/exit"]
            self.prompts.clear()
            try:
                out = self.repl(*lines, notes_enabled=True)
            except KeyboardInterrupt:
                patcher.stop()
                self.fail(f"Ctrl-C in {command} ended the REPL")
            self.assertIn("Interrupted", out, command)
            self.assertEqual(len(self.prompts), 2, command)
            self.out.seek(0)
            self.out.truncate()

    def test_ctrl_c_keeps_every_folders_session(self):
        with mock.patch.object(main, "_do_review", side_effect=KeyboardInterrupt):
            try:
                self.repl("first question", f"/cd {self.b}", "/review", "/cd alpha", "again")
            except KeyboardInterrupt:
                self.fail("Ctrl-C in /review ended the REPL")
        self.assertIn("first question", json.dumps(self.sent[-1][1]))

    def test_interrupted_cd_keeps_the_repl(self):
        with mock.patch.object(folders.Folders, "cd", side_effect=KeyboardInterrupt):
            self.assertIn("Interrupted", self.repl("/cd beta", "/skills"))
        self.assertEqual(len(self.prompts), 3)

    def test_deleted_folder_can_be_left(self):
        out = self.repl(f"/cd {self.b}", self.b.rmdir, "question", "/cd alpha")
        self.assertIn("no longer exists", out)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.prompts[-1], "agent alpha › ")

    def test_one_shot_is_not_saved(self):
        self.repl(prompt="question")
        self.assertEqual(folders.load_recent(), [])


class HeaderPathTest(FolderCase):
    def header(self, path, width=80):
        out = io.StringIO()
        Console(file=out, width=width, color_system=None).print(
            prompt_header(2, None, "agent", "green", when=datetime(2026, 11, 22, 14, 39), path=path))
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, lines)
        self.assertLessEqual(cell_len(lines[0]), width)
        return lines[0]

    def test_path_is_between_mode_and_time(self):
        self.assertIn("#2 · agent · ~/work/billing · 14:39", self.header("~/work/billing"))

    def test_long_path_keeps_folder_and_time(self):
        line = self.header("~/Documents/projects/clients/acme/billing", width=45)
        self.assertIn("…/acme/billing · 14:39", line)

    def test_long_name_keeps_its_end(self):
        line = self.header("~/a-very-long-folder-name-for-billing", width=40)
        self.assertRegex(line, r"…\S*billing · 14:39")

    def test_every_width_keeps_the_time(self):
        for path in ("~/work/billing", "~/" + "\u9879\u76ee" * 20, "/", "~"):
            for width in range(26, 121):
                self.assertIn("14:39", self.header(path, width), (path, width))

    def test_odd_names_are_plain_text(self):
        self.assertIn("~/[bold]/:smile:", self.header("~/[bold]/:smile:"))
        line = self.header("~/a\nb\x1b[31m")
        self.assertIn("~/a b\\x1b[31m", line)

    def test_home_is_shortened(self):
        for home, start, shown in ((self.tmp, self.a, "~/alpha"), (self.a, self.a, "~"),
                                   (self.b, self.a, str(self.a))):
            with mock.patch.object(Path, "home", return_value=home):
                active = folders.Folders(Config(workdir=start), SimpleNamespace(), None)
                self.assertEqual(active.short_path, shown)

    def test_no_home(self):
        active = folders.Folders(Config(workdir=self.a), SimpleNamespace(), None)
        with mock.patch.object(Path, "home", side_effect=RuntimeError):
            self.assertEqual(active.short_path, str(self.a))


class ReviewFolderTest(FolderCase):
    def setUp(self):
        super().setUp()
        self.sub = self.folder("alpha/sub")
        self.folder("alpha/sub/private")
        self.files = [self.a / "outside.py", self.sub / "inside.py", self.sub / "private/key.txt"]
        for path in self.files:
            path.write_text("before\n")
        (self.sub / ".agentignore").write_text("/private/\n")
        self.git(self.a, "init", "-q")
        self.git(self.a, "checkout", "-q", "-b", "main")
        self.git(self.a, "add", "-A")
        self.git(self.a, "commit", "-qm", "base")
        self.git(self.a, "checkout", "-q", "-b", "feature")
        for path in self.files:
            path.write_text(f"changed {path.name}\n")

    def check(self, target):
        material, files, _ = review.collect_material(self.sub, target)
        self.assertEqual(files, ["inside.py"])
        self.assertNotIn("changed key.txt", material)
        self.assertNotIn("changed outside.py", material)

    def test_uncommitted(self):
        self.check("")

    def test_staged(self):
        self.git(self.a, "add", "-A")
        self.check("staged")

    def test_committed(self):
        self.git(self.a, "commit", "-qam", "change")
        for target in ("main", "main..feature", "commits"):
            self.check(target)

    def test_file_in_bracket_folder(self):
        odd = self.folder("app [old]")
        (odd / "x.py").write_text("x = 1\n")
        self.assertEqual(review.collect_material(odd, "x.py")[1], ["x.py"])


class SharedStateTest(FolderCase):
    def test_dispatcher_rule_does_not_stick(self):
        (self.a / "A.kt").write_text("class AViewModel\n")
        setup = {"libs": set(), "dsl": "kts"}

        def context(rule):
            return testgen.collect_context(self.a, "A.kt", "AViewModel", None, "viewmodel", setup,
                                           [], [], "", None, {}, 10000, dispatcher_setup=rule)[0]
        self.assertIn("FirstRule", context("FirstRule"))
        second = context("SecondRule")
        self.assertIn("SecondRule", second)
        self.assertNotIn("FirstRule", second)

    def test_ctrl_c_keeps_texts_waiting_for_the_strict_check(self):
        texts = ["first " * 100, "second " * 100]
        for step in ("agent_cli.llm.chat", "agent_cli.redact.Redactor.mask_text"):
            with mock.patch.dict(os.environ, {"AGENT_REDACT_STRICT": "1", "AGENT_REDACT": "1"}):
                red = redact.Redactor(self.a)
                redact.swap_pending(list(texts))
                with mock.patch(step, side_effect=KeyboardInterrupt):
                    try:
                        red.run_strict(Config(workdir=self.a))
                    except KeyboardInterrupt:
                        pass
            self.assertEqual(redact.swap_pending([]), texts, step)

    def test_odd_names_in_prompt(self):
        for name in ("R&D <x>", "[bold]", "app [1]", ":smile:", "esc\x1b[31m"):
            label = folders.safe_markup(name)
            shown = to_plain_text(input_reader._rich_to_pt(f"[dim] {label} [/dim]"))
            self.assertEqual(shown, f" {safe_text(name)} ")

    def test_prompt_keeps_colors(self):
        parts = input_reader._rich_to_pt("[bold green]code[/bold green][dim] x[/dim]")
        self.assertEqual(list(parts), [("fg:green bold", "code"), ("fg:gray", " x")])

    def test_cd_is_completed(self):
        doc = Document("/cd", 3)
        for completer in (AgentCompleter(), CoderCompleter(SimpleNamespace())):
            self.assertIn("/cd", [c.text for c in completer.get_completions(doc, None)])


if __name__ == "__main__":
    unittest.main()
