"""Local playground for checking LLM replies: what was sent, what came back, what the agent parses.

Binds to 127.0.0.1 only and needs the random session token. Never runs tools.
"""
from __future__ import annotations

import dataclasses
import json
import math
import secrets
import time
import urllib.error
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

import typer

from . import redact
from .agent import THINK_RE, parse_action_calls, parse_tool_calls, render_tool_docs
from .config import ACTION_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, SYSTEM_PROMPT, Config
from .llm import _parse_full, clean_content, iter_stream_events, list_models, open_stream

app = typer.Typer(add_completion=False)

PAGE = Path(__file__).parent / "static" / "playground.html"
# The form owns these; in extra JSON they would also bypass masking ("messages").
RESERVED_KEYS = {"messages", "model", "stream"}


class ClientGone(Exception):
    """The browser closed the connection (Stop button or closed tab)."""


def defaults(cfg: Config) -> dict:
    template = ACTION_SYSTEM_PROMPT if cfg.tool_format == "action" else SYSTEM_PROMPT
    agent = template.format(workdir=cfg.workdir, tool_docs=render_tool_docs(cfg), repo_map_section="")
    return {
        "base_url": redact.mask_for_log(cfg.workdir, cfg.base_url),
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": cfg.stream,
        "thinking": cfg.thinking,
        "keep_special_tokens": cfg.keep_special_tokens,
        "extra_body": cfg.extra_body,
        "tool_format": cfg.tool_format,
        "masking": redact.for_config(cfg).enabled,
        "presets": {
            "none": {"system": "", "stop": ""},
            "chat": {"system": CHAT_SYSTEM_PROMPT, "stop": ""},
            "agent": {"system": agent, "stop": "<tool_response>"},
        },
    }


def build_request(base: Config, data: dict) -> tuple:
    """Validate the form. Returns (cfg, messages, stop); raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    user = str(data.get("user") or "")
    system = str(data.get("system") or "")
    model = str(data.get("model") or "").strip()
    if not user.strip():
        raise ValueError("User prompt is empty")
    if not model:
        raise ValueError("Model is empty")
    try:
        temperature = float(data.get("temperature", base.temperature))
        max_tokens = int(data.get("max_tokens", base.max_tokens))
    except (TypeError, ValueError) as e:
        raise ValueError("Temperature and max tokens must be numbers") from e
    if not math.isfinite(temperature) or max_tokens < 1:
        raise ValueError("Temperature must be finite and max tokens at least 1")

    extra = str(data.get("extra_body", base.extra_body) or "").strip()
    if extra:
        try:
            parsed = json.loads(extra)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extra JSON is not valid: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError("Extra JSON must be an object")
        clash = RESERVED_KEYS & set(parsed)
        if clash:
            raise ValueError(f"Extra JSON can't set {', '.join(sorted(clash))}")

    cfg = dataclasses.replace(
        base, model=model, temperature=temperature, max_tokens=max_tokens, extra_body=extra,
        stream=bool(data.get("stream", base.stream)),
        thinking=str(data.get("thinking", base.thinking)),
        keep_special_tokens=bool(data.get("keep_special_tokens", base.keep_special_tokens)),
    )
    messages = [{"role": "system", "content": system}] if system.strip() else []
    messages.append({"role": "user", "content": user})
    stop = [s for s in str(data.get("stop") or "").splitlines() if s]
    return cfg, messages, stop


def inspect_reply(cfg: Config, content: str, reasoning: str, finish: str, native: list) -> dict:
    """What the agent would make of this reply, plus anything that looks wrong."""
    text = THINK_RE.sub("", clean_content(content)).strip()
    parse = parse_action_calls if cfg.tool_format == "action" else parse_tool_calls
    calls, errors = parse(text)
    warnings = list(errors)
    for call in native:
        args = call["arguments"]
        if isinstance(args, str) and args:
            try:
                json.loads(args)
            except json.JSONDecodeError:
                warnings.append(f"Native call {call['name'] or '?'} has invalid JSON arguments")
    if not text and not native:
        thought = reasoning.strip() or THINK_RE.search(content)
        warnings.append("Only thinking, no answer" if thought else "Empty reply")
    if finish == "length":
        warnings.append(f"Cut off at max_tokens={cfg.max_tokens}")
    elif not finish:
        warnings.append("No finish_reason: the reply may be incomplete")
    return {"cleaned": text, "tool_calls": calls, "native_tool_calls": native, "warnings": warnings}


def run(cfg: Config, messages: list, stop: List[str], send: Callable[[dict], None]) -> None:
    """Make one request and report it through send() as request/delta/done/error events."""
    started = time.monotonic()
    first: Optional[float] = None
    raw: List[str] = []
    usage: dict = {}

    def elapsed() -> int:
        return round((time.monotonic() - started) * 1000)

    def recorded(resp):
        for line in resp:
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                raw.append(text)
            if text.startswith("data:") and '"usage"' in text:
                try:
                    usage.update(json.loads(text[5:]).get("usage") or {})
                except ValueError:
                    pass
            yield line

    try:
        for message in messages:
            redact.note_untrusted(message["content"], "paste")
        with open_stream(cfg, messages, stop=stop, stream=cfg.stream,
                         on_request=lambda body: send({"type": "request", "body": body})) as resp:
            content, reasoning, finish = "", "", ""
            native: list = []
            streamed = "text/event-stream" in resp.headers.get("Content-Type", "")
            if streamed:
                slots: dict = {}
                for kind, value in iter_stream_events(cfg, recorded(resp)):
                    if first is None and kind != "finish":
                        first = elapsed()
                    if kind == "finish":
                        finish = value
                        continue
                    if kind == "content":
                        content += value
                    elif kind == "thinking":
                        reasoning += value
                    else:
                        index, name, args = value
                        slot = slots.setdefault(index if index is not None else len(slots),
                                                {"name": "", "arguments": ""})
                        slot["name"] += name or ""
                        if isinstance(args, dict):
                            slot["arguments"] = args
                        elif args:
                            slot["arguments"] += args
                    # Tool deltas are sent too, so a closed browser is noticed during them.
                    send({"type": "delta", "kind": kind, "text": value if kind != "tool" else ""})
                native = [slots[i] for i in sorted(slots)]
            else:
                body = resp.read().decode("utf-8", errors="replace")
                raw.append(body)
                data = json.loads(body)
                reply = _parse_full(cfg, data)
                usage.update(data.get("usage") or {})
                # Reply.content is already cleaned; show the text as the server sent it.
                content = redact.for_config(cfg).unmask_text(
                    data["choices"][0]["message"].get("content") or "")
                reasoning, finish, native = reply.reasoning, reply.finish_reason, reply.tool_calls

        result = inspect_reply(cfg, content, reasoning, finish, native)
        if cfg.stream and not streamed:
            result["warnings"].append("Server ignored stream=true")
        send(dict(result, type="done", content=content, reasoning=reasoning, finish=finish,
                  first_ms=first, total_ms=elapsed(), usage=usage, raw="\n".join(raw)))
    except ClientGone:
        return
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        send({"type": "error", "text": f"HTTP {e.code} {e.reason}", "raw": body, "total_ms": elapsed()})
    except Exception as e:  # shown in the page: failures are what the playground is for
        send({"type": "error", "text": f"{type(e).__name__}: {e}", "raw": "\n".join(raw),
              "total_ms": elapsed()})


class Handler(BaseHTTPRequestHandler):
    cfg: Config
    session_token: str

    def log_message(self, fmt, *args):
        pass  # the URL holds the token

    def _respond(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "img-src data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._respond(json.dumps(data).encode(), "application/json", status)

    def _authorized(self) -> bool:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        given = query.get("token", [""])[0]
        if secrets.compare_digest(given.encode(), self.session_token.encode()):
            return True
        self._json({"error": "Invalid or missing token"}, 403)
        return False

    def do_GET(self):
        if not self._authorized():
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._respond(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._json(defaults(self.cfg))
        elif path == "/api/models":
            try:
                self._json({"models": list_models(self.cfg)})
            except Exception as e:
                self._json({"models": [], "error": str(e)})
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if not self._authorized():
            return
        if urllib.parse.urlparse(self.path).path != "/api/run":
            self._json({"error": "Not found"}, 404)
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            cfg, messages, stop = build_request(self.cfg, data)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(event: dict) -> None:
            try:
                self.wfile.write(json.dumps(event).encode() + b"\n")
                self.wfile.flush()
            except OSError as e:
                raise ClientGone() from e

        try:
            run(cfg, messages, stop, send)
        except ClientGone:
            pass


def make_server(cfg: Config, token: str, port: int = 0) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"cfg": cfg, "session_token": token})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


@app.command()
def main(
    port: int = typer.Option(0, "--port", "-p", help="Port (0 = random available)"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-u", help="OpenAI-compatible endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to start with"),
):
    """Open a playground for checking LLM responses."""
    cfg = Config()
    if base_url:
        cfg.base_url = base_url
    if model:
        cfg.model = model
    redact.startup_notice(cfg)
    token = secrets.token_urlsafe(32)
    server = make_server(cfg, token, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/?token={token}"

    print(f"agent-playground running at {url}")
    print(f"Model server: {redact.mask_for_log(cfg.workdir, cfg.base_url)}")
    print("Press Ctrl+C to stop.", flush=True)
    if not no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()
