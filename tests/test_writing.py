"""Tests for the one-shot writing commands (grammar, rephrase, tone, email, slack)."""
import unittest
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli import writing
from tests.fake_server import FakeServer


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_unknown_command_returns_false(self):
        with patch.object(writing, "_run") as run:
            self.assertFalse(writing.handle(self.cfg, "/map", "foo"))
            run.assert_not_called()

    def test_grammar_routes_to_grammar_prompt(self):
        with patch.object(writing, "_run") as run:
            self.assertTrue(writing.handle(self.cfg, "/grammar", "he go to school"))
        run.assert_called_once()
        _, system, body = run.call_args[0]
        self.assertEqual(system, writing.GRAMMAR_PROMPT)
        self.assertEqual(body, "he go to school")

    def test_rephrase_routes(self):
        with patch.object(writing, "_run") as run:
            writing.handle(self.cfg, "/rephrase", "make this nicer")
        self.assertEqual(run.call_args[0][1], writing.REPHRASE_PROMPT)

    def test_email_and_slack_route(self):
        with patch.object(writing, "_run") as run:
            writing.handle(self.cfg, "/email", "tell bob the build is fixed")
            writing.handle(self.cfg, "/slack", "standup at 10")
        self.assertEqual(run.call_args_list[0][0][1], writing.EMAIL_PROMPT)
        self.assertEqual(run.call_args_list[1][0][1], writing.SLACK_PROMPT)


class TestTone(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_tone_splits_style_from_text(self):
        with patch.object(writing, "_run") as run:
            writing.handle(self.cfg, "/tone", "formal can you send that over")
        _, system, body = run.call_args[0]
        self.assertIn("formal", system)
        self.assertEqual(body, "can you send that over")

    def test_tone_without_text_warns_and_does_not_call(self):
        with patch.object(writing, "_run") as run:
            handled = writing.handle(self.cfg, "/tone", "formal")
        self.assertTrue(handled)  # still a writing command
        run.assert_not_called()


class TestEmptyText(unittest.TestCase):
    def test_empty_body_warns_and_does_not_call(self):
        cfg = Config()
        with patch.object(writing, "_run") as run:
            self.assertTrue(writing.handle(cfg, "/grammar", "   "))
            self.assertTrue(writing.handle(cfg, "/email", ""))
        run.assert_not_called()


class TestEndToEnd(unittest.TestCase):
    def test_grammar_sends_prompt_and_prints_result(self):
        with FakeServer(["He goes to school."]) as srv:
            cfg = Config(base_url=srv.base_url)
            with patch.object(writing.console, "print") as out:
                writing.handle(cfg, "/grammar", "he go to school")

            body = srv.requests[-1]  # recorded as the parsed JSON payload
            # system prompt is the grammar prompt, user content is the text
            self.assertEqual(body["messages"][0]["content"], writing.GRAMMAR_PROMPT)
            self.assertEqual(body["messages"][1]["content"], "he go to school")

            printed = " ".join(str(c.args[0]) for c in out.call_args_list if c.args)
            self.assertIn("He goes to school.", printed)


if __name__ == "__main__":
    unittest.main()
