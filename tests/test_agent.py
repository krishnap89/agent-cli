"""Tests for tool-call parsing in agent mode."""
import unittest

from agent_cli.agent import parse_tool_calls, thinking_gist


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


if __name__ == "__main__":
    unittest.main()
