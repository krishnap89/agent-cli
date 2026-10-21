"""Tests for natural-language intent routing to the test-gen/doc-gen pipelines."""
import unittest
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli.intent import (
    CHAT,
    DOC_GEN,
    TEST_GEN,
    Intent,
    classify,
    looks_like_candidate,
    route,
    _parse,
)
from rich.console import Console
from tests.fake_server import FakeServer


class ParseTest(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(
            _parse('{"intent":"test_gen","target":"NotificationInteractor","method":""}'),
            Intent(TEST_GEN, "NotificationInteractor", ""),
        )

    def test_method_preserved(self):
        self.assertEqual(
            _parse('{"intent":"test_gen","target":"Foo","method":"bar"}'),
            Intent(TEST_GEN, "Foo", "bar"),
        )

    def test_json_wrapped_in_prose(self):
        self.assertEqual(_parse('sure! {"intent":"doc_gen","target":"Foo.kt"} ok').kind, DOC_GEN)

    def test_garbage_is_chat(self):
        self.assertEqual(_parse("not json at all").kind, CHAT)

    def test_unknown_intent_is_chat(self):
        self.assertEqual(_parse('{"intent":"deploy","target":"X"}').kind, CHAT)

    def test_command_without_target_is_chat(self):
        # A test_gen with no target is unusable; degrade to chat.
        self.assertEqual(_parse('{"intent":"test_gen","target":""}').kind, CHAT)

    def test_strips_mention_sigil_from_target(self):
        # The model often copies the user's "@path" verbatim; resolve_target
        # needs the bare path.
        self.assertEqual(
            _parse('{"intent":"test_gen","target":"@app/src/Foo.kt"}').target,
            "app/src/Foo.kt",
        )

    def test_strips_quotes_and_backticks(self):
        self.assertEqual(_parse('{"intent":"doc_gen","target":"`Foo`"}').target, "Foo")
        self.assertEqual(_parse('{"intent":"doc_gen","target":"\\"Bar.kt\\""}').target, "Bar.kt")


class CandidateGateTest(unittest.TestCase):
    def test_test_requests_pass(self):
        self.assertTrue(looks_like_candidate("add unit tests for NotificationInteractor"))
        self.assertTrue(looks_like_candidate("write a spec for LoginViewModel"))

    def test_doc_requests_pass(self):
        self.assertTrue(looks_like_candidate("document the Repository class"))
        self.assertTrue(looks_like_candidate("add kdoc to Foo"))

    def test_unrelated_messages_skip(self):
        self.assertFalse(looks_like_candidate("explain how login works"))
        self.assertFalse(looks_like_candidate("fix the null pointer in Bar"))


class ClassifyTest(unittest.TestCase):
    def test_classify_reads_model_reply(self):
        reply = '{"intent":"test_gen","target":"NotificationInteractor","method":""}'
        with FakeServer([reply]) as srv:
            cfg = Config(base_url=srv.base_url)
            intent = classify(cfg, "add unit tests for NotificationInteractor")
        self.assertEqual(intent, Intent(TEST_GEN, "NotificationInteractor", ""))

    def test_classify_unreachable_returns_none(self):
        cfg = Config(base_url="http://127.0.0.1:1/v1", retries=0)
        self.assertIsNone(classify(cfg, "add tests for Foo"))


class _StubCoder:
    """Records which pipeline the router dispatched to."""
    def __init__(self):
        self.test_gen_calls = []
        self.doc_gen_calls = []

    def test_gen(self, target, method=""):
        self.test_gen_calls.append((target, method))

    def doc_gen(self, target):
        self.doc_gen_calls.append(target)


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.console = Console(quiet=True)

    def _route(self, reply, text, auto_approve=True, intent_routing=True):
        coder = _StubCoder()
        with FakeServer([reply]) as srv:
            cfg = Config(base_url=srv.base_url, auto_approve=auto_approve)
            cfg.intent_routing = intent_routing
            handled = route(cfg, coder, text, self.console)
        return handled, coder

    def test_routes_test_gen(self):
        handled, coder = self._route(
            '{"intent":"test_gen","target":"NotificationInteractor","method":""}',
            "add unit tests for NotificationInteractor",
        )
        self.assertTrue(handled)
        self.assertEqual(coder.test_gen_calls, [("NotificationInteractor", "")])

    def test_routes_doc_gen(self):
        handled, coder = self._route(
            '{"intent":"doc_gen","target":"Foo.kt"}',
            "document Foo.kt",
        )
        self.assertTrue(handled)
        self.assertEqual(coder.doc_gen_calls, ["Foo.kt"])

    def test_chat_intent_falls_through(self):
        handled, coder = self._route(
            '{"intent":"chat"}',
            "explain how the tests are structured",
        )
        self.assertFalse(handled)
        self.assertEqual(coder.test_gen_calls, [])

    def test_gate_skips_classifier_entirely(self):
        # No test/doc keyword -> no LLM call, no dispatch. FakeServer with an
        # empty script would error if a request were made.
        coder = _StubCoder()
        with FakeServer([]) as srv:
            cfg = Config(base_url=srv.base_url)
            handled = route(cfg, coder, "fix the crash on startup", self.console)
        self.assertFalse(handled)
        self.assertEqual(srv.requests, [])

    def test_routing_disabled(self):
        coder = _StubCoder()
        with FakeServer([]) as srv:
            cfg = Config(base_url=srv.base_url)
            cfg.intent_routing = False
            handled = route(cfg, coder, "add tests for Foo", self.console)
        self.assertFalse(handled)
        self.assertEqual(srv.requests, [])

    def test_declined_falls_through(self):
        # User says no at the confirm prompt -> fall back to normal chat.
        with patch("agent_cli.intent.Confirm.ask", return_value=False):
            handled, coder = self._route(
                '{"intent":"test_gen","target":"Foo"}',
                "write tests for Foo",
                auto_approve=False,
            )
        self.assertFalse(handled)
        self.assertEqual(coder.test_gen_calls, [])


if __name__ == "__main__":
    unittest.main()
