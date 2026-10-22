"""Local web UI for the coder – serves a single-page app and an SSE API.

Binds to 127.0.0.1 only.  A random session token prevents other sites from
calling the API (checked via query parameter, not cookies).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import urllib.parse
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from .config import Config, UNLIMITED_CHARS, parse_context_chars
from .coder import Coder
from .checks import builtin_lint, custom_lint, ext_lint_cmd, run_tests, truncate_output
from .editblock import parse_edit_blocks, unified_diff
from .llm import (
    LLMError, chat, list_models, open_stream, iter_stream_events, clean_content,
)
from .repomap import RepoMap
from .review import (
    collect_material, needs_split, split_material_by_file,
    build_review_messages, build_merge_messages,
)

app = typer.Typer(add_completion=False)

STATIC_DIR = Path(__file__).parent / "static"

# ---- helpers -----------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200,
                   content_type: str = "text/plain") -> None:
    body = text.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)
    return json.loads(raw) if raw else {}


# ---- the Coder wrapper (thread-safe bits) ------------------------------------

class WebCoder:
    """Wraps a Coder instance for use from the web API.

    The Coder class was built for a single-threaded REPL.  We run model calls
    in a background thread so the HTTP server stays responsive, and serialize
    all mutations through a lock.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.coder = Coder(cfg)
        self.lock = threading.Lock()  # guards coder mutations (files, history, undo)
        self._busy_lock = threading.Lock()
        self._cancel = threading.Event()
        self._busy = False
        if self.coder.repo_map is not None:
            self.coder.repo_map.build()

    def _claim_busy(self) -> bool:
        """Atomically take the single-worker slot. Returns False if already busy."""
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
            return True

    # ---- file ops (fast, under lock) ----------------------------------------

    def all_files(self) -> List[str]:
        return self.coder.all_files()

    def chat_files(self) -> Dict[str, str]:
        return {
            **{f: "editable" for f in self.coder.editable},
            **{f: "read-only" for f in self.coder.read_only},
        }

    def add_files(self, patterns: List[str], read_only: bool = False) -> List[str]:
        import glob as glob_mod
        added = []
        with self.lock:
            target = self.coder.read_only if read_only else self.coder.editable
            for pat in patterns:
                matches = sorted(glob_mod.glob(str(self.coder.root / pat), recursive=True))
                matches = [m for m in matches if Path(m).is_file()]
                for m in matches:
                    try:
                        rel = self.coder._rel(m)
                    except ValueError:
                        continue
                    self.coder.editable.pop(rel, None)
                    self.coder.read_only.pop(rel, None)
                    target[rel] = None
                    added.append(rel)
        return added

    def drop_files(self, patterns: List[str]) -> List[str]:
        import fnmatch as fn_mod
        with self.lock:
            if not patterns:
                dropped = list(self.coder.editable) + list(self.coder.read_only)
                self.coder.editable.clear()
                self.coder.read_only.clear()
                return dropped
            dropped = []
            for pat in patterns:
                gone = [f for f in list(self.coder.editable) + list(self.coder.read_only)
                        if f == pat or fn_mod.fnmatch(f, pat) or f.endswith("/" + pat)]
                for f in gone:
                    self.coder.editable.pop(f, None)
                    self.coder.read_only.pop(f, None)
                    dropped.append(f)
        return dropped

    # ---- undo ---------------------------------------------------------------

    def undo(self) -> Optional[List[str]]:
        with self.lock:
            if not self.coder.undo_stack:
                return None
            snapshot = self.coder.undo_stack.pop()
            reverted = []
            for rel, old_content in snapshot.items():
                path = self.coder.root / rel
                if old_content is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_text(old_content, encoding="utf-8")
                reverted.append(rel)
            self.coder._all_files = None
        return reverted

    # ---- chat (streamed via SSE, run in a background thread) ----------------

    def send_message(self, text: str, mode: str, on_chunk, on_done, on_error):
        """Send a message to the model and stream the reply.

        on_chunk(text) is called for each piece of the reply.
        on_done(full_text, diffs) is called when the reply is complete.
        on_error(msg) is called on failure.
        All callbacks are called from a background thread.
        """
        if not self._claim_busy():
            on_error("Model is already working on a request.")
            return
        self._cancel.clear()

        def _run():
            try:
                self._do_send(text, mode, on_chunk, on_done)
            except Exception as e:
                on_error(str(e))
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()

    def cancel(self):
        self._cancel.set()

    def _do_send(self, user_text: str, mode: str, on_chunk, on_done):
        old_mode = self.coder.mode
        self.coder.mode = mode

        self.coder.history.append({"role": "user", "content": user_text})
        messages = self.coder.build_messages(user_text)

        content_parts: List[str] = []
        finish = ""

        try:
            with open_stream(self.cfg, messages) as resp:
                for kind, value in iter_stream_events(self.cfg, resp):
                    if self._cancel.is_set():
                        on_chunk(json.dumps({"type": "cancelled"}) + "\n")
                        break
                    if kind == "content":
                        content_parts.append(value)
                        on_chunk(json.dumps({"type": "content", "text": value}) + "\n")
                    elif kind == "thinking":
                        on_chunk(json.dumps({"type": "thinking", "text": value}) + "\n")
                    elif kind == "finish":
                        finish = value
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(str(e))
        finally:
            self.coder.mode = old_mode

        from .agent import THINK_RE
        full_text = THINK_RE.sub("", clean_content("".join(content_parts))).strip()

        self.coder.history.append({"role": "assistant", "content": full_text})

        # Apply edit blocks if in code mode. Guarded by self.lock so a concurrent
        # /api/add, /api/drop or /api/undo can't mutate editable/undo_stack mid-apply.
        diffs = []
        if mode == "code":
            blocks, problems = parse_edit_blocks(full_text, default_path=self.coder._single_file())
            if blocks:
                from .editblock import apply_edit, EditError
                changed: Dict[str, Optional[str]] = {}
                errors = []
                with self.lock:
                    for b in blocks:
                        try:
                            rel = self.coder._rel(b.path.strip())
                        except ValueError as e:
                            errors.append(str(e))
                            continue
                        path = self.coder.root / rel
                        if rel not in self.coder.editable:
                            if path.exists():
                                self.coder.editable[rel] = None
                            elif not b.search.strip():
                                pass  # new file creation
                            else:
                                errors.append(f"Edit for {rel} failed: file not in chat.")
                                continue
                        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
                        try:
                            new = apply_edit(old or "", b.search, b.replace)
                        except EditError as e:
                            errors.append(str(e))
                            continue
                        changed.setdefault(rel, old)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(new, encoding="utf-8")
                        if rel not in self.coder.editable:
                            self.coder.editable[rel] = None

                    for rel, old_content in changed.items():
                        new_content = (self.coder.root / rel).read_text(encoding="utf-8", errors="replace") \
                            if (self.coder.root / rel).exists() else ""
                        d = unified_diff(rel, old_content or "", new_content)
                        if d:
                            diffs.append({"file": rel, "diff": d})

                    if changed:
                        self.coder.undo_stack.append(changed)
                        self.coder._all_files = None

                if errors:
                    on_chunk(json.dumps({"type": "errors", "errors": errors}) + "\n")

        if finish == "length":
            on_chunk(json.dumps({"type": "warning", "text": "Reply was cut off (raise AGENT_MAX_TOKENS)"}) + "\n")

        on_done(full_text, diffs)

    # ---- token info ----------------------------------------------------------

    def token_info(self) -> dict:
        msgs = self.coder.build_messages("")
        total = sum(len(m["content"]) for m in msgs)
        budget = self.cfg.context_chars
        unlimited = budget >= UNLIMITED_CHARS
        return {
            "used_chars": total,
            "budget_chars": None if unlimited else budget,
            "used_tokens": total // 4,
            "budget_tokens": None if unlimited else budget // 4,
            "pct": 0 if unlimited or not budget else min(100, round(total * 100 / budget)),
        }

    # ---- settings ------------------------------------------------------------

    def get_settings(self) -> dict:
        return {
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "context_chars": 0 if self.cfg.context_chars >= UNLIMITED_CHARS else self.cfg.context_chars,
            "stream": self.cfg.stream,
        }

    def update_settings(self, data: dict) -> dict:
        if "base_url" in data:
            self.cfg.base_url = data["base_url"]
        if "model" in data:
            self.cfg.model = data["model"]
        if "temperature" in data:
            self.cfg.temperature = float(data["temperature"])
        if "max_tokens" in data:
            self.cfg.max_tokens = int(data["max_tokens"])
        if "context_chars" in data:
            self.cfg.context_chars = parse_context_chars(str(data["context_chars"]))
        if "api_key" in data:
            self.cfg.api_key = data["api_key"]
        return self.get_settings()

    def test_connection(self) -> dict:
        try:
            models = list_models(self.cfg)
            return {"ok": True, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- slash commands ------------------------------------------------------

    def run_lint(self, files=None) -> dict:
        errors = self.coder.run_lint(files)
        return {"ok": not errors, "errors": errors}

    def run_test(self) -> dict:
        if not self.cfg.test_cmd:
            return {"ok": False, "error": "No test command set. Set AGENT_TEST_CMD."}
        err = self.coder.run_test()
        if err:
            truncated = truncate_output(err, self.cfg.check_output_chars)
            return {"ok": False, "output": truncated}
        return {"ok": True}

    def run_command(self, command: str) -> dict:
        import subprocess
        try:
            r = subprocess.run(command, shell=True, cwd=self.coder.root,
                               capture_output=True, text=True, timeout=600)
            output = (r.stdout + r.stderr).strip()
            return {"exit_code": r.returncode, "output": output[-8000:]}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "output": "(timed out after 600s)"}

    def repo_map(self, query: str = "") -> dict:
        if self.coder.repo_map is None:
            return {"ok": False, "error": "Repo map is off (AGENT_MAP=0)."}
        self.coder.repo_map.build()
        text = self.coder.repo_map.render(query=query, budget=self.cfg.map_chars)
        return {"ok": True, "text": text}

    def token_report(self) -> dict:
        msgs = self.coder.build_messages("")
        parts = []
        parts.append(("system prompt", len(msgs[0]["content"])))
        if self.cfg.notes_enabled:
            from .notes import load_notes
            _, notes_text = load_notes(self.coder.root, self.cfg.notes_file,
                                       self.cfg.notes_chars, warn_once=False)
            if notes_text:
                parts.append(("project notes", len(notes_text)))
        if self.coder.repo_map is not None and len(msgs) > 1:
            parts.append(("repo map", len(msgs[1]["content"])))
        for f in list(self.coder.editable) + list(self.coder.read_only):
            parts.append((f, len(self.coder._file_block(f))))
        parts.append(("chat history", sum(len(m["content"]) for m in self.coder.history)))
        total = sum(n for _, n in parts)
        budget = self.cfg.context_chars
        return {
            "parts": [{"name": name, "tokens": n // 4} for name, n in parts],
            "total_tokens": total // 4,
            "budget_tokens": None if budget >= UNLIMITED_CHARS else budget // 4,
        }

    def show_diff(self) -> dict:
        if not self.coder.last_diffs:
            return {"ok": False, "error": "No edits in the last request."}
        diffs = []
        for d in self.coder.last_diffs:
            fname = ""
            for line in d.splitlines():
                if line.startswith("+++ b/"):
                    fname = line[6:]
                    break
                if line.startswith("--- a/"):
                    fname = line[6:]
            diffs.append({"file": fname, "diff": d})
        return {"ok": True, "diffs": diffs}

    def review(self, target: str, on_chunk, on_done, on_error):
        if not self._claim_busy():
            on_error("Model is already working on a request.")
            return
        self._cancel.clear()

        def _run():
            try:
                self._do_review(target, on_chunk, on_done)
            except Exception as e:
                on_error(str(e))
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()

    def _do_review(self, target: str, on_chunk, on_done):
        try:
            material, files, desc = collect_material(self.coder.root, target)
        except ValueError as e:
            on_chunk(json.dumps({"type": "error", "text": str(e)}) + "\n")
            on_done("", [])
            return

        budget = self.cfg.context_chars
        split = needs_split(material, budget)
        info = f"Reviewing {desc}: {len(material):,} chars"
        if split:
            info += ", split per file"
        on_chunk(json.dumps({"type": "info", "text": info}) + "\n")

        if split:
            chunks = split_material_by_file(material)
            if not chunks:
                chunks = [("all", material)]
            findings = []
            for i, (fname, chunk) in enumerate(chunks):
                if self._cancel.is_set():
                    break
                on_chunk(json.dumps({"type": "info", "text": f"[{i + 1}/{len(chunks)}] {fname}"}) + "\n")
                messages = build_review_messages(chunk, fname)
                text = self._call_model_blocking(messages)
                if text:
                    findings.append(f"## {fname}\n{text}")

            if len(findings) > 1:
                on_chunk(json.dumps({"type": "info", "text": "Merging findings..."}) + "\n")
                messages = build_merge_messages(findings)
                full = self._stream_model(messages, on_chunk)
            else:
                full = findings[0] if findings else "No findings."
                on_chunk(json.dumps({"type": "content", "text": full}) + "\n")
        else:
            messages = build_review_messages(material, desc)
            full = self._stream_model(messages, on_chunk)

        self.coder.history.append({"role": "user", "content": f"Review of {desc}"})
        self.coder.history.append({"role": "assistant", "content": full})
        on_done(full, [])

    def _call_model_blocking(self, messages: list) -> Optional[str]:
        """Non-streaming call used for per-file review chunks (reviews run at temp 0.2)."""
        reply = chat(self.cfg, messages, temperature=0.2)  # retries; reviews are deterministic
        from .agent import THINK_RE
        return THINK_RE.sub("", reply.content).strip()

    def _stream_model(self, messages: list, on_chunk) -> str:
        from .agent import THINK_RE
        content_parts: List[str] = []
        with open_stream(self.cfg, messages, temperature=0.2) as resp:
            for kind, value in iter_stream_events(self.cfg, resp):
                if self._cancel.is_set():
                    break
                if kind == "content":
                    content_parts.append(value)
                    on_chunk(json.dumps({"type": "content", "text": value}) + "\n")
        return THINK_RE.sub("", clean_content("".join(content_parts))).strip()


# ---- HTTP handler ------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    web_coder: WebCoder
    session_token: str

    def log_message(self, fmt, *args):
        pass  # suppress noisy request logging

    def _check_token(self) -> bool:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = params.get("token", [None])[0]
        if token != self.session_token:
            _json_response(self, {"error": "invalid token"}, 403)
            return False
        return True

    def _route_path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    # ---- GET ----------------------------------------------------------------

    def do_GET(self):
        path = self._route_path()

        if path == "/":
            self._serve_index()
            return

        if not self._check_token():
            return

        if path == "/api/files":
            _json_response(self, self.web_coder.all_files())
        elif path == "/api/chat-files":
            _json_response(self, self.web_coder.chat_files())
        elif path == "/api/tokens":
            _json_response(self, self.web_coder.token_info())
        elif path == "/api/settings":
            _json_response(self, self.web_coder.get_settings())
        elif path == "/api/history":
            _json_response(self, self.web_coder.coder.history)
        else:
            self.send_error(404)

    def _serve_index(self):
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            self.send_error(500, "index.html not found")
            return
        html = html_path.read_text(encoding="utf-8")
        html = html.replace("__SESSION_TOKEN__", self.session_token)
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- POST ---------------------------------------------------------------

    def do_POST(self):
        if not self._check_token():
            return
        path = self._route_path()

        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/add":
            data = _read_json_body(self)
            added = self.web_coder.add_files(
                data.get("files", []),
                read_only=data.get("read_only", False),
            )
            _json_response(self, {"added": added})
        elif path == "/api/drop":
            data = _read_json_body(self)
            dropped = self.web_coder.drop_files(data.get("files", []))
            _json_response(self, {"dropped": dropped})
        elif path == "/api/undo":
            reverted = self.web_coder.undo()
            if reverted is None:
                _json_response(self, {"error": "nothing to undo"}, 400)
            else:
                _json_response(self, {"reverted": reverted})
        elif path == "/api/cancel":
            self.web_coder.cancel()
            _json_response(self, {"ok": True})
        elif path == "/api/settings":
            data = _read_json_body(self)
            result = self.web_coder.update_settings(data)
            _json_response(self, result)
        elif path == "/api/test-connection":
            result = self.web_coder.test_connection()
            _json_response(self, result)
        elif path == "/api/clear":
            self.web_coder.coder.history.clear()
            _json_response(self, {"ok": True})
        elif path == "/api/command":
            self._handle_command()
        elif path == "/api/review":
            self._handle_review()
        else:
            self.send_error(404)

    def _handle_chat(self):
        data = _read_json_body(self)
        message = data.get("message", "").strip()
        mode = data.get("mode", "code")
        if not message:
            _json_response(self, {"error": "empty message"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        done_event = threading.Event()
        error_msg = [None]

        def on_chunk(chunk_json: str):
            try:
                self.wfile.write(f"data: {chunk_json}\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.web_coder.cancel()

        def on_done(full_text: str, diffs: list):
            try:
                msg = json.dumps({"type": "done", "diffs": diffs})
                self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            done_event.set()

        def on_error(msg: str):
            try:
                err = json.dumps({"type": "error", "text": msg})
                self.wfile.write(f"data: {err}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            error_msg[0] = msg
            done_event.set()

        self.web_coder.send_message(message, mode, on_chunk, on_done, on_error)
        done_event.wait(timeout=self.web_coder.cfg.timeout + 30)

    def _handle_command(self):
        data = _read_json_body(self)
        cmd = data.get("command", "").strip()
        arg = data.get("arg", "").strip()

        if cmd == "lint":
            result = self.web_coder.run_lint()
            _json_response(self, result)
        elif cmd == "test":
            result = self.web_coder.run_test()
            _json_response(self, result)
        elif cmd == "run":
            if not arg:
                _json_response(self, {"error": "Usage: /run <command>"}, 400)
            else:
                result = self.web_coder.run_command(arg)
                _json_response(self, result)
        elif cmd == "map":
            result = self.web_coder.repo_map(arg)
            _json_response(self, result)
        elif cmd == "tokens":
            result = self.web_coder.token_report()
            _json_response(self, result)
        elif cmd == "diff":
            result = self.web_coder.show_diff()
            _json_response(self, result)
        else:
            _json_response(self, {"error": f"Unknown command: /{cmd}"}, 400)

    def _handle_review(self):
        data = _read_json_body(self)
        target = data.get("target", "").strip()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        done_event = threading.Event()

        def on_chunk(chunk_json: str):
            try:
                self.wfile.write(f"data: {chunk_json}\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.web_coder.cancel()

        def on_done(full_text: str, diffs: list):
            try:
                msg = json.dumps({"type": "done"})
                self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            done_event.set()

        def on_error(msg: str):
            try:
                err = json.dumps({"type": "error", "text": msg})
                self.wfile.write(f"data: {err}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            done_event.set()

        self.web_coder.review(target, on_chunk, on_done, on_error)
        done_event.wait(timeout=self.web_coder.cfg.timeout + 30)


# ---- CLI entry point ---------------------------------------------------------

@app.command()
def main(
    port: int = typer.Option(0, "--port", "-p", help="Port (0 = random available)"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser"),
    workdir: Path = typer.Option(Path.cwd(), "--workdir", "-w", help="Project directory"),
):
    """Start the web UI for the coder."""
    cfg = Config(workdir=workdir)
    web_coder = WebCoder(cfg)

    token = secrets.token_urlsafe(32)

    handler_class = type("BoundHandler", (Handler,), {
        "web_coder": web_coder,
        "session_token": token,
    })

    # Threaded so /api/cancel (and other requests) are served while a chat or
    # review is streaming; the single-threaded HTTPServer blocked them.
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/?token={token}"

    print(f"agent-ui running at {url}")
    print(f"Working directory: {cfg.workdir}")
    print("Press Ctrl+C to stop.")

    if not no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
