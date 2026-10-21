"""Tests for the web UI server (agent_cli.web)."""
import json
import os
import secrets
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from agent_cli.config import Config
from agent_cli.web import Handler, WebCoder
from tests.fake_server import FakeServer


def _make_server(cfg, token=None):
    """Create a test web server bound to a random port."""
    token = token or secrets.token_urlsafe(16)
    wc = WebCoder(cfg)
    handler_class = type("TestHandler", (Handler,), {
        "web_coder": wc,
        "session_token": token,
    })
    httpd = HTTPServer(("127.0.0.1", 0), handler_class)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    return httpd, wc, token, f"http://127.0.0.1:{port}"


def _get(base, path, token):
    url = f"{base}{path}?token={token}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _post(base, path, token, data=None):
    url = f"{base}{path}?token={token}"
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read().decode()
        if "application/json" in ctype:
            return json.loads(raw)
        return raw


def _post_sse(base, path, token, data=None):
    """POST and collect SSE events."""
    url = f"{base}{path}?token={token}"
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=15) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return events


class TestWebAuth(unittest.TestCase):
    """Test that API calls require a valid session token."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = Config(workdir=Path(self.tmpdir), use_map=False)
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()

    def test_index_no_token_ok(self):
        """Index page is served without a token (it embeds the token)."""
        with urllib.request.urlopen(f"{self.base}/", timeout=5) as resp:
            html = resp.read().decode()
            self.assertIn("agent-ui", html)
            self.assertIn(self.token, html)

    def test_api_no_token_403(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.base}/api/files", timeout=5)
        self.assertEqual(ctx.exception.code, 403)

    def test_api_wrong_token_403(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.base}/api/files?token=wrong", timeout=5)
        self.assertEqual(ctx.exception.code, 403)

    def test_api_correct_token(self):
        data = _get(self.base, "/api/files", self.token)
        self.assertIsInstance(data, list)


class TestWebFiles(unittest.TestCase):
    """Test file add/drop API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create some test files
        Path(self.tmpdir, "hello.py").write_text("print('hello')\n")
        Path(self.tmpdir, "world.py").write_text("print('world')\n")
        os.makedirs(Path(self.tmpdir, "sub"), exist_ok=True)
        Path(self.tmpdir, "sub", "nested.py").write_text("x = 1\n")

        cfg = Config(workdir=Path(self.tmpdir), use_map=False)
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()

    def test_add_and_list(self):
        result = _post(self.base, "/api/add", self.token, {"files": ["hello.py"]})
        self.assertEqual(result["added"], ["hello.py"])

        chat = _get(self.base, "/api/chat-files", self.token)
        self.assertIn("hello.py", chat)
        self.assertEqual(chat["hello.py"], "editable")

    def test_add_read_only(self):
        result = _post(self.base, "/api/add", self.token, {"files": ["world.py"], "read_only": True})
        self.assertEqual(result["added"], ["world.py"])

        chat = _get(self.base, "/api/chat-files", self.token)
        self.assertEqual(chat["world.py"], "read-only")

    def test_drop(self):
        _post(self.base, "/api/add", self.token, {"files": ["hello.py"]})
        _post(self.base, "/api/add", self.token, {"files": ["world.py"]})

        result = _post(self.base, "/api/drop", self.token, {"files": ["hello.py"]})
        self.assertIn("hello.py", result["dropped"])

        chat = _get(self.base, "/api/chat-files", self.token)
        self.assertNotIn("hello.py", chat)
        self.assertIn("world.py", chat)

    def test_drop_all(self):
        _post(self.base, "/api/add", self.token, {"files": ["hello.py", "world.py"]})
        result = _post(self.base, "/api/drop", self.token, {"files": []})
        self.assertEqual(len(result["dropped"]), 2)

        chat = _get(self.base, "/api/chat-files", self.token)
        self.assertEqual(len(chat), 0)


class TestWebChat(unittest.TestCase):
    """Test chat with a fake model server."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        Path(self.tmpdir, "main.py").write_text("def greet():\n    return 'hello'\n")

        self.fake = FakeServer(["Here is the answer: use `greet()`."])
        self.fake.start()

        cfg = Config(
            workdir=Path(self.tmpdir),
            base_url=self.fake.base_url,
            model="fake-model",
            use_map=False,
            stream=True,
        )
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()
        self.fake.stop()

    def test_chat_returns_streamed_content(self):
        events = _post_sse(self.base, "/api/chat", self.token, {
            "message": "What does greet do?",
            "mode": "ask",
        })
        content_events = [e for e in events if e.get("type") == "content"]
        self.assertTrue(len(content_events) > 0)
        full_text = "".join(e["text"] for e in content_events)
        self.assertIn("greet", full_text)

        done_events = [e for e in events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)

    def test_chat_applies_edits(self):
        edit_reply = """I'll fix the function.

main.py
<<<<<<< SEARCH
def greet():
    return 'hello'
=======
def greet(name):
    return f'hello {name}'
>>>>>>> REPLACE
"""
        self.fake.reset([edit_reply])

        _post(self.base, "/api/add", self.token, {"files": ["main.py"]})

        events = _post_sse(self.base, "/api/chat", self.token, {
            "message": "Add a name parameter",
            "mode": "code",
        })

        done_events = [e for e in events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        diffs = done_events[0].get("diffs", [])
        self.assertTrue(len(diffs) > 0)
        self.assertEqual(diffs[0]["file"], "main.py")

        # Verify the file was actually changed
        content = Path(self.tmpdir, "main.py").read_text()
        self.assertIn("name", content)


class TestWebUndo(unittest.TestCase):
    """Test undo via the web API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        Path(self.tmpdir, "app.py").write_text("x = 1\n")

        edit_reply = """app.py
<<<<<<< SEARCH
x = 1
=======
x = 42
>>>>>>> REPLACE
"""
        self.fake = FakeServer([edit_reply])
        self.fake.start()

        cfg = Config(
            workdir=Path(self.tmpdir),
            base_url=self.fake.base_url,
            model="fake-model",
            use_map=False,
            stream=True,
        )
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()
        self.fake.stop()

    def test_undo_reverts_edit(self):
        _post(self.base, "/api/add", self.token, {"files": ["app.py"]})
        _post_sse(self.base, "/api/chat", self.token, {
            "message": "change x",
            "mode": "code",
        })

        # Verify edit applied
        self.assertIn("42", Path(self.tmpdir, "app.py").read_text())

        # Undo
        result = _post(self.base, "/api/undo", self.token)
        self.assertIn("app.py", result["reverted"])

        # Verify reverted
        self.assertEqual(Path(self.tmpdir, "app.py").read_text(), "x = 1\n")

    def test_undo_nothing(self):
        try:
            _post(self.base, "/api/undo", self.token)
            self.fail("Should have raised HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


class TestWebSettings(unittest.TestCase):
    """Test settings API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = Config(workdir=Path(self.tmpdir), use_map=False)
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()

    def test_get_settings(self):
        s = _get(self.base, "/api/settings", self.token)
        self.assertIn("base_url", s)
        self.assertIn("model", s)
        self.assertIn("temperature", s)

    def test_update_settings(self):
        _post(self.base, "/api/settings", self.token, {
            "model": "new-model",
            "temperature": 0.8,
        })
        s = _get(self.base, "/api/settings", self.token)
        self.assertEqual(s["model"], "new-model")
        self.assertEqual(s["temperature"], 0.8)

    def test_api_key_not_in_get(self):
        """API key should not be returned in GET /settings."""
        s = _get(self.base, "/api/settings", self.token)
        self.assertNotIn("api_key", s)


class TestWebTokens(unittest.TestCase):
    """Test token meter API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = Config(workdir=Path(self.tmpdir), use_map=False)
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()

    def test_token_info(self):
        info = _get(self.base, "/api/tokens", self.token)
        self.assertIn("used_chars", info)
        self.assertIn("budget_chars", info)
        self.assertIn("pct", info)
        self.assertGreaterEqual(info["pct"], 0)
        self.assertLessEqual(info["pct"], 100)


class TestWebClear(unittest.TestCase):
    """Test clear chat API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fake = FakeServer(["hello!"])
        self.fake.start()
        cfg = Config(
            workdir=Path(self.tmpdir),
            base_url=self.fake.base_url,
            model="fake-model",
            use_map=False,
            stream=True,
        )
        self.httpd, self.wc, self.token, self.base = _make_server(cfg)

    def tearDown(self):
        self.httpd.shutdown()
        self.fake.stop()

    def test_clear(self):
        _post_sse(self.base, "/api/chat", self.token, {
            "message": "hi",
            "mode": "ask",
        })
        history = _get(self.base, "/api/history", self.token)
        self.assertTrue(len(history) > 0)

        _post(self.base, "/api/clear", self.token)
        history = _get(self.base, "/api/history", self.token)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
