"""Tests for the agent REPL's slash-command completer."""
import unittest

from prompt_toolkit.document import Document

from agent_cli.completer import AgentCompleter, AGENT_COMMANDS


def _complete(text: str, files_fn=None):
    c = AgentCompleter(files_fn=files_fn)
    doc = Document(text, len(text))
    return [(x.text, x.display_meta_text) for x in c.get_completions(doc, None)]


SAMPLE_FILES = [
    "agent_cli/main.py",
    "agent_cli/completer.py",
    "tests/test_completer.py",
    "README.md",
]


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


class TestAtFileCompletion(unittest.TestCase):
    def _texts(self, text):
        return [t for t, _ in _complete(text, files_fn=lambda: SAMPLE_FILES)]

    def test_bare_at_lists_all_files(self):
        self.assertEqual(len(self._texts("look @")), len(SAMPLE_FILES))

    def test_at_keeps_the_leading_at(self):
        self.assertEqual(self._texts("see @main"), ["@agent_cli/main.py"])

    def test_substring_match_anywhere_in_path(self):
        # 'compl' matches both completer.py and test_completer.py
        self.assertEqual(
            self._texts("explain @compl"),
            ["@agent_cli/completer.py", "@tests/test_completer.py"],
        )

    def test_at_works_even_after_a_slash_command(self):
        self.assertEqual(self._texts("/review @test"), ["@tests/test_completer.py"])

    def test_no_files_fn_yields_nothing_for_at(self):
        self.assertEqual(_complete("@main"), [])

    def test_files_fn_called_once_and_cached(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return SAMPLE_FILES

        c = AgentCompleter(files_fn=fn)
        for s in ("@m", "@ma", "@main"):
            list(c.get_completions(Document(s, len(s)), None))
        self.assertEqual(calls["n"], 1)

    def test_files_fn_error_is_swallowed(self):
        def boom():
            raise RuntimeError("git failed")

        self.assertEqual(_complete("@x", files_fn=boom), [])


if __name__ == "__main__":
    unittest.main()
