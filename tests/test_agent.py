"""Tests for tool-call parsing in agent mode."""
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from agent_cli import llm, playground
from agent_cli.agent import (Agent, parse_action_calls, parse_tool_calls,
                             strip_action_blocks, thinking_gist)
from agent_cli.config import Config
from agent_cli.llm import Reply
from agent_cli.skills import format_skill_context, load_skills


class ParseToolCallsTest(unittest.TestCase):
    def test_wrapped_json(self):
        calls, errors = parse_tool_calls(
            '<tool_call>\n{"name": "search", "arguments": {"pattern": "Login"}}\n</tool_call>'
        )
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "search", "arguments": {"pattern": "Login"}}])

    def test_xml_function_form(self):
        calls, errors = parse_tool_calls(
            "<tool_call><function=search><parameter=pattern>Login</parameter></function></tool_call>"
        )
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "search", "arguments": {"pattern": "Login"}}])

    def test_bare_json_no_wrapper(self):
        # The reported bug: model emits bare JSON with no <tool_call> wrapper.
        calls, errors = parse_tool_calls('{"name": "repo_map"}')
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "repo_map", "arguments": {}}])

    def test_bare_json_with_args(self):
        calls, errors = parse_tool_calls('{"name": "read_file", "arguments": {"path": "a.py"}}')
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "read_file", "arguments": {"path": "a.py"}}])

    def test_bare_json_fenced(self):
        calls, errors = parse_tool_calls('```json\n{"name": "repo_map"}\n```')
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "repo_map", "arguments": {}}])

    def test_prose_then_bare_json(self):
        # The exact reported transcript: a line of prose, then a bare call.
        reply = (
            "To understand how this codebase works, I'll start by identifying the "
            "most important files using the repo_map tool.\n\n"
            '{"name": "repo_map"}'
        )
        calls, errors = parse_tool_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "repo_map", "arguments": {}}])

    def test_prose_then_bare_json_with_args(self):
        reply = 'Let me read it.\n{"name": "read_file", "arguments": {"path": "a.py"}}'
        calls, errors = parse_tool_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "read_file", "arguments": {"path": "a.py"}}])

    def test_prose_then_json_then_trailing_noise(self):
        # Reported transcript: prose, the call on its own line, then "STOP".
        reply = (
            "Let's start by examining the dependencies:\n\n"
            '{"name": "dependency_graph", "arguments": {"path": "agent_cli/coder.py"}}\n\n'
            "STOP"
        )
        calls, errors = parse_tool_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(
            calls,
            [{"name": "dependency_graph", "arguments": {"path": "agent_cli/coder.py"}}],
        )

    def test_two_bare_calls_on_their_own_lines(self):
        reply = '{"name": "repo_map"}\n{"name": "outline", "arguments": {"path": "a.py"}}'
        calls, errors = parse_tool_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [
            {"name": "repo_map", "arguments": {}},
            {"name": "outline", "arguments": {"path": "a.py"}},
        ])

    def test_json_mentioned_mid_sentence_is_not_a_call(self):
        # JSON quoted inside a sentence (not at a line start) must not run.
        calls, errors = parse_tool_calls(
            'You can call it like {"name": "repo_map"} to map the repo, for example.'
        )
        self.assertEqual(calls, [])
        self.assertEqual(errors, [])

    def test_plain_answer_is_not_a_call(self):
        calls, errors = parse_tool_calls("The codebase has two REPLs: agent and coder.")
        self.assertEqual(calls, [])
        self.assertEqual(errors, [])


class ThinkingGistTest(unittest.TestCase):
    def test_prefers_reasoning_field(self):
        self.assertEqual(
            thinking_gist("The token is checked in middleware.py.", "ignored"),
            "The token is checked in middleware.py.",
        )

    def test_falls_back_to_inline_think_tag(self):
        self.assertEqual(
            thinking_gist("", "<think>Need the repo map first.</think>Here goes."),
            "Need the repo map first.",
        )

    def test_skips_trailing_leadin_for_substantive_line(self):
        reasoning = ("Let me look at the auth flow.\n"
                     "The token is validated in middleware.py.\n"
                     "So I will read that file now.")
        self.assertEqual(thinking_gist(reasoning, ""),
                         "The token is validated in middleware.py.")

    def test_keeps_leadin_when_nothing_else(self):
        self.assertEqual(thinking_gist("Let me check the cache.", ""),
                         "Let me check the cache.")

    def test_empty_when_no_thinking(self):
        self.assertEqual(thinking_gist("", "a normal answer"), "")
        self.assertEqual(thinking_gist("   ", ""), "")

    def test_truncates_long_gist(self):
        gist = thinking_gist("x" * 300, "")
        self.assertEqual(len(gist), 140)
        self.assertTrue(gist.endswith("..."))


class ActionWordingTest(unittest.TestCase):
    """In action format nothing the agent sends says "tool"."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = Config(workdir=self.root, tool_format="action", use_map=False,
                          notes_enabled=False, auto_approve=True,
                          jira_url="https://x.example", jira_email="a@b.c", jira_token="t")

    def sent(self, first, reasoning="", prompt="hello"):
        """All text sent to the model while it replies `first`, then finishes."""
        replies, sent = iter([Reply(first, "stop", reasoning), Reply("ok", "stop", "")]), []

        def chat(cfg, messages, stop=None, **kwargs):
            sent.extend(m["content"] for m in messages if m["role"] != "assistant")
            sent.extend(stop or [])
            return next(replies)

        with mock.patch("agent_cli.agent.chat", chat):
            Agent(self.cfg).ask(prompt)
        text = "\n".join(sent)
        self.assertIsNone(re.search("tool", text, re.IGNORECASE), text)
        return text

    def test_results_and_stop(self):
        text = self.sent("ACTION list_dir path=.\nACTION list_dir path=.")
        self.assertIn("# Actions", text)
        self.assertIn("<action_response>\n[list_dir]", text)

    def test_unknown_action(self):
        self.assertIn("Unknown action: nope", self.sent("ACTION nope"))

    def test_parse_error(self):
        self.assertIn("<action_response>\nError:", self.sent("ACTION todo arg={bad"))

    def test_argument_error(self):
        self.assertIn("missing 1 required", self.sent("ACTION read_file"))
        error = TypeError("Tools.t_search() missing 'pattern'")  # Python 3.10+ wording
        with mock.patch("agent_cli.tools.Tools.run", side_effect=error):
            self.assertIn("Error: search() missing", self.sent("ACTION search"))

    def test_empty_reply_nudges(self):
        self.assertIn("empty reply. Write an ACTION line", self.sent(""))
        self.assertIn("Now do it: Write an ACTION line", self.sent("", reasoning="hmm"))

    def test_file_nudge_example_works(self):
        text = self.sent("```\nhello\n```", prompt="create a file notes.md")
        example = re.search(r"ACTION write_file path=<path>\n.*?CONTENT>>>", text, re.S).group()
        self.sent(example.replace("<path>", "notes.md").replace("<full content>", "one\ntwo"))
        self.assertEqual((self.root / "notes.md").read_text(), "one\ntwo\n")

    def test_file_with_thinking_tags_and_tokens_is_saved_intact(self):
        for body in ("a <think>b</think> c\n", "only <think> opens\n", "only </think> closes\n",
                     "token <|im_end|> kept\n"):
            self.sent(f"<think>plan</think>ACTION write_file path=t.md\n<<<CONTENT\n{body}CONTENT>>>")
            self.assertEqual((self.root / "t.md").read_text(), body)

    def test_draft_inside_thinking_is_not_written(self):
        draft = "ACTION write_file path=d.md\n<<<CONTENT\ndraft\nCONTENT>>>\n"
        for reply in (f"<think>\n{draft}</think>Answer", f"plan\n{draft}</think>Answer"):
            self.sent(reply)
            self.assertFalse((self.root / "d.md").exists())

    def test_bundled_skills_fit_the_default_budget(self):
        skills = load_skills(Path(__file__).resolve().parent.parent)
        for skill in skills:
            if skill.name in ("click-dummy", "readme-gen", "big-task"):
                self.assertLessEqual(len(skill.body), Config().skill_chars, skill.name)

    def test_edit_with_thinking_tags_is_saved_intact(self):
        for new in ("a <think>b</think> c", "only </think> closes", "token <|im_end|> kept"):
            (self.root / "e.md").write_text("x\n")
            self.sent(f"<think>plan</think>ACTION edit_file path=e.md\n<<<OLD\nx\nOLD\n<<<NEW\n{new}\nNEW")
            self.assertEqual((self.root / "e.md").read_text(), new + "\n")

    def test_cut_off_edit_leaves_the_file_alone(self):
        (self.root / "e.py").write_text("x\n")
        for end in ("half", "half\nNEW"):  # the second: cut inside "NEW_LIMIT = 5"
            replies = iter([Reply(f"ACTION edit_file path=e.py\n<<<OLD\nx\nOLD\n<<<NEW\n{end}",
                                  "length", ""), Reply("ok", "stop", "")])
            with mock.patch("agent_cli.agent.chat", lambda *a, **k: next(replies)):
                Agent(self.cfg).ask("edit it")
            self.assertEqual((self.root / "e.py").read_text(), "x\n")

    def test_bundled_skills(self):
        skills = load_skills(Path(__file__).resolve().parent.parent)
        for skill in skills:
            if skill.name in ("click-dummy", "readme-gen", "big-task"):
                with self.subTest(skill.name):
                    self.sent("ok", prompt="hello\n" + format_skill_context(skill, 100_000))

    def test_markup_like_content_does_not_crash_the_preview(self):
        self.cfg.auto_approve = False  # so the diff preview is printed
        body = "re = /[/]/\n[/red] [bold]\nends in \\\n"
        with mock.patch("agent_cli.tools.Confirm.ask", return_value=True):
            self.sent(f"ACTION write_file path=m.js\n<<<CONTENT\n{body}CONTENT>>>")
        self.assertEqual((self.root / "m.js").read_text(), body)

    def test_playground_preset(self):
        preset = playground.defaults(self.cfg)["presets"]["agent"]
        self.assertNotIn("tool", preset["system"].lower())
        self.assertEqual(preset["stop"], "<action_response>")


class ParseActionCallsTest(unittest.TestCase):
    """The ACTION-line tool protocol (cfg.tool_format == 'action')."""

    def test_simple_call(self):
        calls, errors = parse_action_calls("ACTION list_dir path=.")
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "list_dir", "arguments": {"path": "."}}])

    def test_integer_args_are_ints(self):
        calls, errors = parse_action_calls(
            "ACTION read_file path=app/build.gradle.kts start_line=1 end_line=80"
        )
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "read_file", "arguments": {
            "path": "app/build.gradle.kts", "start_line": 1, "end_line": 80}}])

    def test_value_with_spaces_and_parens(self):
        # search patterns and commands carry spaces; no quoting required.
        calls, errors = parse_action_calls("ACTION search pattern=fun login(")
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "search", "arguments": {"pattern": "fun login("}}])

    def test_command_keeps_flags(self):
        calls, errors = parse_action_calls(
            "ACTION run_command command=./gradlew :app:assembleDebug --offline"
        )
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "run_command", "arguments": {
            "command": "./gradlew :app:assembleDebug --offline"}}])

    def test_boolean_arg(self):
        calls, errors = parse_action_calls("ACTION search pattern=Login regex=true")
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "search", "arguments": {
            "pattern": "Login", "regex": True}}])

    def test_several_calls_in_one_reply(self):
        reply = ("I'll look around first.\n"
                 "ACTION list_dir path=.\n"
                 "ACTION search pattern=LoginViewModel")
        calls, errors = parse_action_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [
            {"name": "list_dir", "arguments": {"path": "."}},
            {"name": "search", "arguments": {"pattern": "LoginViewModel"}},
        ])

    def test_edit_file_heredoc(self):
        reply = ("Bump the timeout.\n"
                 "ACTION edit_file path=app/src/Login.kt\n"
                 "<<<OLD\nval timeout = 30\nval retries = 3\nOLD\n"
                 "<<<NEW\nval timeout = 60\nval retries = 3\nNEW\n"
                 "Done.")
        calls, errors = parse_action_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "edit_file", "arguments": {
            "path": "app/src/Login.kt",
            "old_str": "val timeout = 30\nval retries = 3",
            "new_str": "val timeout = 60\nval retries = 3"}}])

    def test_edit_file_missing_block_reports_error(self):
        reply = "ACTION edit_file path=a.kt\n<<<OLD\nx\nOLD\n"  # no NEW block
        calls, errors = parse_action_calls(reply)
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("NEW", errors[0])

    def test_todo_inline_json(self):
        reply = ('ACTION todo arg={"action": "set", '
                 '"items": [{"text": "read flow", "status": "in_progress"}]}')
        calls, errors = parse_action_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "todo", "arguments": {
            "action": "set",
            "items": [{"text": "read flow", "status": "in_progress"}]}}])

    def test_apply_patch_inline_json_list(self):
        reply = 'ACTION apply_patch arg=[{"path": "a.kt", "old_str": "x", "new_str": "y"}]'
        calls, errors = parse_action_calls(reply)
        self.assertEqual(errors, [])
        self.assertEqual(calls, [{"name": "apply_patch", "arguments": {
            "edits": [{"path": "a.kt", "old_str": "x", "new_str": "y"}]}}])

    def test_bad_json_arg_reports_error(self):
        calls, errors = parse_action_calls("ACTION todo arg={not json}")
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("todo", errors[0])

    def test_plain_answer_is_not_a_call(self):
        calls, errors = parse_action_calls("The login flow lives in LoginViewModel.")
        self.assertEqual(calls, [])
        self.assertEqual(errors, [])


class WriteBlockTest(unittest.TestCase):
    HEAD = "ACTION write_file path=a.md\n<<<CONTENT\n"

    def write(self, body):
        return parse_action_calls(f"{self.HEAD}{body}CONTENT>>>")

    def test_content_is_verbatim(self):
        for body in ("", "\n", "one\n", "  indented\n\n\n", 'say "hi" \\n a=b\n',
                     "page\fbreak\u2028line\n", "  <<<CONTENT\n  CONTENT>>>\n",
                     "<<<OLD\nOLD\n<<<NEW\nNEW\n"):
            with self.subTest(body=body):
                self.assertEqual(self.write(body), (
                    [{"name": "write_file", "arguments": {"path": "a.md", "content": body}}], []))

    def test_actions_inside_content_do_not_run(self):
        calls, errors = self.write("ACTION run_command command=rm -rf build\n")
        self.assertEqual([c["name"] for c in calls], ["write_file"])
        self.assertEqual(errors, [])

    def test_path_with_spaces(self):
        calls, _ = parse_action_calls("ACTION write_file path=my notes.md\n<<<CONTENT\nCONTENT>>>")
        self.assertEqual(calls[0]["arguments"]["path"], "my notes.md")

    def test_crlf(self):
        calls, _ = parse_action_calls((self.HEAD + "a\nb\nCONTENT>>>").replace("\n", "\r\n"))
        self.assertEqual(calls[0]["arguments"]["content"], "a\nb\n")

    def test_several_calls_around_blocks(self):
        calls, errors = parse_action_calls(
            f"ACTION list_dir path=.\n{self.HEAD}x\nCONTENT>>>\n"
            f"{self.HEAD}y\nCONTENT>>>\nACTION list_dir path=.")
        self.assertEqual([c["name"] for c in calls],
                         ["list_dir", "write_file", "write_file", "list_dir"])
        self.assertEqual(errors, [])

    def test_unclear_block_runs_nothing_from_there_on(self):
        danger = "ACTION run_command command=rm -rf build\n"
        end = "CONTENT>>>\n"
        for reply in (
            self.HEAD + danger,                                  # cut off
            self.HEAD + danger + "  " + end,                     # indented close
            "ACTION write_file path=a.md\n" + danger,            # no opening
            "ACTION write_file path=a.md content=first\n" + danger,
            "ACTION write_file\n<<<CONTENT\n" + danger + end,    # no path
            "<<<CONTENT\n" + danger + end,                       # no ACTION line
            self.HEAD + "x\n" + end + danger + end,              # close marker in content
            self.HEAD + "<<<CONTENT\n" + end + danger + end,     # nested block
            self.HEAD + "x\n" + self.HEAD + danger + end,        # first block left open
        ):
            with self.subTest(reply=reply):
                calls, errors = parse_action_calls("ACTION list_dir path=.\n" + reply)
                self.assertEqual([c["name"] for c in calls], ["list_dir"])
                self.assertEqual(len(errors), 1)
                self.assertNotIn("rm -rf", strip_action_blocks(reply))


class EditBlockTest(unittest.TestCase):
    HEAD = "ACTION edit_file path=a.kt\n<<<OLD\nx\nOLD\n<<<NEW\n"

    def test_indented_marker_word_is_content(self):
        calls, errors = parse_action_calls(self.HEAD + "enum class S {\n    NEW\n}\nNEW")
        self.assertEqual(errors, [])
        self.assertEqual(calls[0]["arguments"]["new_str"], "enum class S {\n    NEW\n}")

    def test_empty_new_deletes(self):
        calls, errors = parse_action_calls(self.HEAD + "NEW")
        self.assertEqual((calls[0]["arguments"]["new_str"], errors), ("", []))

    def test_two_edits(self):
        calls, errors = parse_action_calls(f"{self.HEAD}one\nNEW\n{self.HEAD}two\nNEW")
        self.assertEqual([c["arguments"]["new_str"] for c in calls], ["one", "two"])
        self.assertEqual(errors, [])

    def test_unclear_edit_runs_nothing_from_there_on(self):
        danger = "ACTION run_command command=rm -rf build\n"
        start = "ACTION edit_file path=a.kt\n"
        for reply in (
            self.HEAD + danger,                              # NEW not closed
            self.HEAD + danger + "  NEW",                    # indented close
            self.HEAD + "one\n" + self.HEAD + danger + "NEW",  # first edit left open
            self.HEAD + "a\nNEW\n" + danger + "NEW",         # close marker in content
            start + danger,                                  # no blocks
            start + "<<<OLD\n" + danger,                     # OLD not closed
            start + "<<<OLD\nx\nOLD\n" + danger,             # no NEW
            "ACTION edit_file\n<<<OLD\n" + danger + "OLD\n<<<NEW\nNEW",  # no path
            "<<<OLD\n" + danger + "OLD",                     # no ACTION line
        ):
            with self.subTest(reply=reply):
                calls, errors = parse_action_calls("ACTION list_dir path=.\n" + reply)
                self.assertEqual([c["name"] for c in calls], ["list_dir"])
                self.assertEqual(len(errors), 1)
                self.assertNotIn("rm -rf", strip_action_blocks(reply))


class RunFailureTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = Config(workdir=self.root, tool_format="xml", use_map=False,
                          notes_enabled=False, auto_approve=True)

    def test_exception_in_second_call_keeps_both_results(self):
        (self.root / "a").write_text("hi")
        reply = ('<tool_call>{"name": "write_file", "arguments": {"path": "b", "content": "B"}}</tool_call>'
                 '<tool_call>{"name": "read_file", "arguments": {"path": "a", "start_line": "abc"}}</tool_call>')
        replies, seen = iter([Reply(reply, "stop", ""), Reply("ok", "stop", "")]), []

        def chat(cfg, messages, **kwargs):
            seen.append(messages[-1]["content"])
            return next(replies)

        with mock.patch("agent_cli.agent.chat", chat):
            Agent(self.cfg).ask("go")
        self.assertIn("[write_file]\nWrote b", seen[1])
        self.assertIn("[read_file]\nError: invalid literal", seen[1])

    def test_exception_without_message_is_named(self):
        with mock.patch("agent_cli.tools.Tools.run", side_effect=RuntimeError):
            result = Agent(self.cfg)._run_tool({"name": "search", "arguments": {}})
        self.assertIn("Error: RuntimeError", result)

    def test_ctrl_c_still_interrupts(self):
        with mock.patch("agent_cli.tools.Tools.run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                Agent(self.cfg)._run_tool({"name": "search", "arguments": {}})

    def test_markup_in_names_and_errors_does_not_crash(self):
        agent = Agent(self.cfg)
        self.assertIn("Unknown tool", agent._run_tool({"name": "[/x]", "arguments": {}}))
        self.assertIn("Error", agent._run_tool({"name": "read_file", "arguments": {"path": "[/x]"}}))

    def test_null_command_does_not_wait_on_stdin(self):
        with mock.patch("shlex.split", side_effect=AssertionError("would read stdin")):
            result = Agent(self.cfg)._run_tool({"name": "run_command", "arguments": {"command": None}})
        self.assertIn("command must be a string", result)

    def test_approval_prompt_shows_the_whole_command(self):
        self.cfg.auto_approve = False
        command = "echo ok [x; touch hidden; echo ]"
        with mock.patch("agent_cli.tools.Confirm.ask", return_value=False) as ask:
            Agent(self.cfg)._run_tool({"name": "run_command", "arguments": {"command": command}})
        self.assertIn(command, ask.call_args[0][0].plain)


class CutOffTest(unittest.TestCase):
    def test_last_action_of_a_cut_off_reply_is_not_run(self):
        for last in ("ACTION run_command command=rm -rf build",
                     'ACTION apply_patch arg=[{"path": "a", "old_str": "x", "new_str": "y"}]',
                     "ACTION edit_file path=a\n<<<OLD\nx\nOLD\n<<<NEW\ny\nNEW\nmore code"):
            with self.subTest(last=last):
                calls, errors = parse_action_calls("ACTION list_dir path=.\n" + last, cut_off=True)
                self.assertEqual([c["name"] for c in calls], ["list_dir"])
                self.assertIn("cut off", errors[0])

    def test_broken_last_action_does_not_drop_the_one_before(self):
        calls, errors = parse_action_calls("ACTION list_dir path=.\nACTION todo arg={bad", cut_off=True)
        self.assertEqual([c["name"] for c in calls], ["list_dir"])
        self.assertEqual(len(errors), 1)

    def test_xml_function_must_be_closed_when_cut_off(self):
        open_call = "<tool_call><function=run_command><parameter=command>rm -rf build"
        for reply in (open_call, open_call + "</tool_call>"):
            self.assertEqual(parse_tool_calls(reply, cut_off=True)[0], [])
        closed = open_call + "</parameter></function>"
        self.assertEqual(len(parse_tool_calls(closed, cut_off=True)[0]), 1)

    def test_complete_json_call_runs_when_cut_off(self):
        reply = '<tool_call>{"name": "repo_map"}</tool_call> and then'
        self.assertEqual(len(parse_tool_calls(reply, cut_off=True)[0]), 1)

    def test_stream_that_just_stops_is_incomplete(self):
        piece = b'data: {"choices":[{"delta":{"content":"hi"}}]}'
        finish = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        for lines, expected in (([piece], "incomplete"), ([piece, b"data: [DONE]"], ""),
                                ([piece, finish], "stop")):
            self.assertEqual(llm._read_stream(Config(), lines, None).finish_reason, expected)

    def test_cut_off_command_is_never_executed(self):
        for fmt, text in (("action", "ACTION run_command command=rm -rf build"),
                          ("xml", "<tool_call><function=run_command><parameter=command>rm -rf build")):
            for reason in llm.CUT_OFF_REASONS:
                with self.subTest(fmt=fmt, reason=reason), tempfile.TemporaryDirectory() as tmp:
                    cfg = Config(workdir=Path(tmp), tool_format=fmt, use_map=False,
                                 notes_enabled=False, auto_approve=True)
                    replies = iter([Reply(text, reason, ""), Reply("ok", "stop", "")])
                    with mock.patch("agent_cli.agent.chat", lambda *a, **k: next(replies)), \
                            mock.patch("agent_cli.tools.Tools.run") as run:
                        Agent(cfg).ask("clean up")
                    run.assert_not_called()


class ScreenSafetyTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = Config(workdir=Path(tmp.name), use_map=False, notes_enabled=False,
                          auto_approve=False)
        self.agent = Agent(self.cfg)
        self.out = io.StringIO()
        screen = Console(file=self.out, width=200, emoji=False)
        for module in ("agent", "tools"):
            patcher = mock.patch(f"agent_cli.{module}.console", screen)
            patcher.start()
            self.addCleanup(patcher.stop)

    def approve(self, action):
        with mock.patch("agent_cli.tools.Confirm.ask", return_value=False) as ask:
            self.agent.tools._confirm(action)
        return ask.call_args[0][0].plain, ask.call_args[1]["default"]

    def test_hidden_characters_are_shown_and_enter_means_no(self):
        for char, code in (("\x1b", "\\x1b"), ("\r", "\\r"), ("\x9b", "\\x9b"), ("\x7f", "\\x7f"),
                           ("\u202e", "\\u202e"), ("\u200b", "\\u200b"), ("\u3164", "\\u3164"),
                           ("\ud800", "\\ud800")):
            with self.subTest(code=code):
                shown, default = self.approve(f"run `a{char}b`")
                self.assertIn(f"a{code}b", shown)
                self.assertFalse(default)

    def test_enter_means_yes_only_for_short_visible_actions(self):
        for action, expected in (("run `./gradlew test`", True), ("run `cat <<E\nhi\nE`", True),
                                 ("run `echo :smile: caf\u00e9`", True),
                                 ("run `" + " " * 600 + "`", False), ("run `" + "\n" * 12 + "`", False)):
            self.assertEqual(self.approve(action), (f"Allow: {action}", expected))

    def test_diff_shows_every_changed_line(self):
        self.agent.tools._show_diff("f", "-- old\nok\n", "ok\n++ ; touch x\ncp :smile: \x1b[8m\rz\n")
        shown = self.out.getvalue()
        for part in ("--- old", "+++ ; touch x", "cp :smile: \\x1b[8m\\rz", "(+2 -1)"):
            self.assertIn(part, shown)

    def test_hidden_characters_in_written_content_make_enter_no(self):
        (self.agent.tools.root / "e").write_text("x\n")
        for body, expected in (("ok\n", True), ("a\x1b[8m\n", False), ("line\n" * 80 + "\u202e\n", False)):
            for name, args in (("write_file", {"path": "n", "content": body}),
                               ("edit_file", {"path": "e", "old_str": "x", "new_str": body}),
                               ("apply_patch", {"edits": [{"path": "p", "new_str": body}]})):
                with self.subTest(name=name, body=body[-10:]):
                    with mock.patch("agent_cli.tools.Confirm.ask", return_value=False) as ask:
                        self.agent.tools.run(name, args)
                    self.assertEqual(ask.call_args[1]["default"], expected)

    def test_model_text_cannot_drive_the_terminal(self):
        reply = Reply("answer \x1b[2J `\x1b[8m`", "stop", "check the [/x] tag \x1b]0;title\x07")
        self.cfg.show_thinking = True
        with mock.patch("agent_cli.agent.chat", return_value=reply):
            self.agent.ask("q")
        self.assertNotIn("\x1b", self.out.getvalue())
        self.assertIn("check the [/x] tag", self.out.getvalue())


class StripActionBlocksTest(unittest.TestCase):
    def test_strips_write_block(self):
        text = "Writing.\nACTION write_file path=a\n<<<CONTENT\n<h1>x</h1>\nCONTENT>>>\nDone."
        self.assertEqual(strip_action_blocks(text), "Writing.\nDone.")

    def test_strips_action_lines_keeps_prose(self):
        text = "I'll look first.\nACTION list_dir path=.\nACTION search pattern=Login"
        self.assertEqual(strip_action_blocks(text), "I'll look first.")

    def test_strips_heredoc_body(self):
        text = ("Bump it.\nACTION edit_file path=a.kt\n"
                "<<<OLD\nold\nOLD\n<<<NEW\nnew\nNEW\nDone.")
        self.assertEqual(strip_action_blocks(text), "Bump it.\nDone.")

    def test_plain_prose_unchanged(self):
        text = "Here is the final answer with no actions."
        self.assertEqual(strip_action_blocks(text), text)


if __name__ == "__main__":
    unittest.main()
