"""A small OpenAI-compatible fake server for testing.

Starts an HTTP server on a free port in a background thread.  It answers
POST /v1/chat/completions from a *script* (a list of replies, or a callable
that receives the request messages and returns a reply string).  Every
request is recorded so tests can assert on the messages sent.

Supports both JSON and SSE streaming responses.
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, List, Optional, Union

# A script entry is either a plain string (the assistant reply) or a callable
# that receives (messages: list[dict]) -> str.
ScriptEntry = Union[str, Callable[[List[dict]], str]]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence logs

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        req = json.loads(raw)
        self.server.requests.append(req)  # type: ignore[attr-defined]

        script = self.server.script  # type: ignore[attr-defined]
        idx = self.server.call_index  # type: ignore[attr-defined]
        self.server.call_index = idx + 1  # type: ignore[attr-defined]

        if idx < len(script):
            entry = script[idx]
            if callable(entry):
                reply_text = entry(req["messages"])
            else:
                reply_text = entry
        else:
            reply_text = "(no more scripted replies)"

        stream = req.get("stream", False)
        if stream:
            self._send_stream(reply_text)
        else:
            self._send_json(reply_text)

    def _send_json(self, text: str):
        body = json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, text: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # Send the whole text as one chunk for simplicity
        chunk = json.dumps({
            "choices": [{
                "delta": {"content": text},
                "finish_reason": None,
            }]
        })
        self.wfile.write(f"data: {chunk}\n\n".encode())
        # finish chunk
        done_chunk = json.dumps({
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
            }]
        })
        self.wfile.write(f"data: {done_chunk}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class FakeServer:
    """Usage::

        server = FakeServer(["reply 1", "reply 2"])
        server.start()
        # ... run tests using server.base_url ...
        server.stop()

    Or as a context manager::

        with FakeServer(["reply"]) as srv:
            cfg = Config(base_url=srv.base_url, ...)
    """

    def __init__(self, script: Optional[List[ScriptEntry]] = None):
        self.script = script or []
        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.script = self.script  # type: ignore[attr-defined]
        self.httpd.requests = []  # type: ignore[attr-defined]
        self.httpd.call_index = 0  # type: ignore[attr-defined]
        self._thread = None  # type: Optional[threading.Thread]

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list:
        return self.httpd.requests  # type: ignore[attr-defined]

    def reset(self, script: Optional[List[ScriptEntry]] = None):
        """Reset for reuse: clear requests, reset index, optionally set a new script."""
        self.httpd.requests = []  # type: ignore[attr-defined]
        self.httpd.call_index = 0  # type: ignore[attr-defined]
        if script is not None:
            self.script = script
            self.httpd.script = script  # type: ignore[attr-defined]

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
