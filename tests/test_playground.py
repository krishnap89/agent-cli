"""Tests for the response playground (agent_cli.playground)."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from agent_cli import redact
from agent_cli.config import Config
from agent_cli.playground import inspect_reply, make_server
from tests.fake_server import FakeServer


class TestPlayground(unittest.TestCase):
    def setUp(self):
        redact.reset_sessions()
        self.tmp = tempfile.TemporaryDirectory()
        self.llm = FakeServer()
        self.llm.start()
        self.cfg = Config(base_url=self.llm.base_url, model="fake-model", api_key="private-key",
                          workdir=Path(self.tmp.name), extra_body="", tool_format="xml", http_log="")
        self.server = make_server(self.cfg, "tok")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.llm.stop()
        self.llm.httpd.server_close()
        self.tmp.cleanup()
        redact.reset_sessions()

    def get(self, path, token="tok"):
        with urllib.request.urlopen(f"{self.base}{path}?token={token}", timeout=5) as resp:
            return resp.read().decode()

    def run_prompt(self, **fields):
        body = {"user": "hi", "model": "fake-model", **fields}
        req = urllib.request.Request(f"{self.base}/api/run?token=tok", data=json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=10) as resp:
            return [json.loads(line) for line in resp.read().decode().splitlines()]

    def test_requires_token(self):
        for path in ("/", "/api/config", "/api/models"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path, token="wrong")
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()

    def test_page_config_and_models(self):
        self.assertIn("agent-playground", self.get("/"))
        config = self.get("/api/config")
        self.assertNotIn("private-key", config)
        self.assertIn("read_file", json.loads(config)["presets"]["agent"]["system"])
        self.assertEqual(json.loads(self.get("/api/models"))["models"], ["fake-model"])

    def test_stream_and_json_replies_match(self):
        self.llm.reset(["<think>hm</think>Hello"] * 2)
        for stream in (True, False):
            done = self.run_prompt(stream=stream)[-1]
            self.assertEqual(done["content"], "<think>hm</think>Hello")
            self.assertEqual(done["cleaned"], "Hello")
            self.assertEqual(done["finish"], "stop")
            self.assertEqual(done["warnings"], [])
            self.assertTrue(done["raw"])
            self.assertIs(self.llm.requests[-1]["stream"], stream)

    def test_form_settings_reach_the_server(self):
        self.llm.reset(["ok"])
        events = self.run_prompt(system="be brief", temperature="0.7", max_tokens="55",
                                 stop="END\nHALT", extra_body='{"top_p": 0.5}')
        sent = self.llm.requests[0]
        self.assertEqual(events[0]["body"], sent)
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual((sent["temperature"], sent["max_tokens"], sent["top_p"]), (0.7, 55, 0.5))
        self.assertEqual(sent["stop"], ["END", "HALT"])
        self.assertEqual(self.cfg.max_tokens, Config().max_tokens)  # session config untouched

    def test_secrets_are_masked_before_sending(self):
        if not redact.for_config(self.cfg).enabled:
            self.skipTest("masking is off in this environment")
        secret = "ghp_" + "a8Kd93LmQp27XzR5tVb1NcY6wEh4JfG0sUi2"
        self.llm.reset([lambda messages: messages[-1]["content"]])
        events = self.run_prompt(user=f"token {secret}")
        self.assertNotIn(secret, json.dumps(self.llm.requests))
        self.assertNotIn(secret, json.dumps(events[0]))
        self.assertNotIn(secret, events[-1]["raw"])
        self.assertIn(secret, events[-1]["content"])

    def test_tool_calls_are_parsed(self):
        self.llm.reset(['<tool_call>{"name": "search", "arguments": {"pattern": "X"}}</tool_call>'])
        self.assertEqual(self.run_prompt()[-1]["tool_calls"],
                         [{"name": "search", "arguments": {"pattern": "X"}}])

    def test_bad_input_is_rejected_without_calling_the_model(self):
        for fields in ({"user": " "}, {"model": ""}, {"max_tokens": "0"}, {"temperature": "x"},
                       {"extra_body": "{"}, {"extra_body": "[]"}, {"extra_body": '{"messages": []}'}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.run_prompt(**fields)
            self.assertEqual(caught.exception.code, 400, fields)
            caught.exception.close()
        self.assertEqual(self.llm.requests, [])

    def test_server_error_is_reported(self):
        self.cfg.base_url += "/missing"
        error = self.run_prompt()[-1]
        self.assertEqual(error["type"], "error")
        self.assertIn("404", error["text"])

    def test_warnings(self):
        def warnings(content, reasoning="", finish="stop", native=()):
            return " ".join(inspect_reply(self.cfg, content, reasoning, finish, list(native))["warnings"])

        self.assertIn("Empty reply", warnings(""))
        self.assertIn("Only thinking", warnings("", reasoning="hm"))
        self.assertIn("Cut off", warnings("partial", finish="length"))
        self.assertIn("No finish_reason", warnings("partial", finish=""))
        self.assertIn("Could not parse", warnings("<tool_call>{bad</tool_call>"))
        self.assertIn("invalid JSON", warnings("", native=[{"name": "search", "arguments": "{bad"}]))


if __name__ == "__main__":
    unittest.main()
