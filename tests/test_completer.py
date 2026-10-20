"""Tests for the agent REPL's slash-command completer."""
import unittest

from prompt_toolkit.document import Document

from agent_cli.completer import AgentCompleter, AGENT_COMMANDS


def _complete(text: str):
    c = AgentCompleter()
    doc = Document(text, len(text))
    return [(x.text, x.display_meta_text) for x in c.get_completions(doc, None)]


class TestAgentCompleter(unittest.TestCase):
    def test_slash_shows_all_commands(self):
        comps = _complete("/")
        self.assertEqual(len(comps), len(AGENT_COMMANDS))

    def test_prefix_filters(self):
        self.assertEqual(_complete("/gr"), [("/grammar", "fix grammar & spelling")])

    def test_every_entry_has_a_hint(self):
        for text, meta in _complete("/"):
            self.assertTrue(meta, f"{text} has no hint")

    def test_no_completion_after_a_space(self):
        # Once a command + space is typed, arguments are free-form.
        self.assertEqual(_complete("/tone "), [])
        self.assertEqual(_complete("/email hi"), [])

    def test_non_slash_text_ignored(self):
        self.assertEqual(_complete("hello"), [])
        self.assertEqual(_complete(""), [])

    def test_opts_into_complete_while_typing(self):
        self.assertTrue(AgentCompleter.complete_while_typing)


if __name__ == "__main__":
    unittest.main()
