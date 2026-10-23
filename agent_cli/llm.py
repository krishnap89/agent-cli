"""Plain-HTTP client for an OpenAI-compatible /chat/completions endpoint.

Uses only the Python standard library (no openai SDK, no requests).

Two features protect against 504 Gateway Timeout errors:
- Streaming: the server sends the reply piece by piece, so proxies and load
  balancers in between see constant traffic and don't time out.
- Retries: 502/503/504 and dropped connections are retried with a short wait.
"""
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Callable, List, Optional

from .config import Config

RETRY_STATUS = {429, 502, 503, 504}


def log_http(cfg: Config, direction: str, text: str) -> None:
    """Append one HTTP request or reply to cfg.http_log, if set.

    direction is a short tag like 'REQUEST' or 'REPLY'. The Authorization
    header's bearer token is redacted so the log is safe to share. Logging
    never raises: a bad path is silently skipped rather than killing a reply.
    """
    if not cfg.http_log:
        return
    text = re.sub(r"(Bearer )\S+", r"\1<redacted>", text)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(cfg.http_log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {stamp}  {direction} =====\n{text}\n")
    except OSError:
        pass


def _log_request(cfg: Config, req: "urllib.request.Request") -> None:
    headers = "\n".join(f"{k}: {v}" for k, v in req.header_items())
    body = req.data.decode("utf-8", errors="replace") if req.data else ""
    log_http(cfg, "REQUEST", f"{req.get_method()} {req.full_url}\n{headers}\n\n{body}")

# on_progress(phase, chars) is called while streaming; phase is "thinking" or "writing"
ProgressFn = Optional[Callable[[str, int], None]]


class LLMError(Exception):
    pass


class _Retryable(Exception):
    pass


# Control tokens that can appear once special tokens are kept.
_CONTROL_TOKENS_RE = re.compile(r"<\|(?:im_end|im_start|endoftext|end_of_text|eot_id)\|>")


class Reply:
    def __init__(self, content: str, finish_reason: str, reasoning: str,
                 tool_calls: Optional[list] = None):
        self.content = clean_content(content)
        self.finish_reason = finish_reason  # "stop", "length", "tool_calls", ...
        self.reasoning = reasoning          # thinking text, if the server splits it out
        # Native tool calls, if the server's tool parser extracted them:
        # [{"name": str, "arguments": dict or str}]
        self.tool_calls = tool_calls or []


def _headers(cfg: Config) -> dict:
    h = {"Content-Type": "application/json"}
    if cfg.api_key and cfg.api_key != "not-needed":
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def build_payload(cfg: Config, messages: list, stop: Optional[List[str]] = None,
                  stream: Optional[bool] = None, temperature: Optional[float] = None) -> dict:
    """Build the /chat/completions request body, honoring every Config option.

    web.py and llm.chat share this so request options can't drift between them.
    """
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature if temperature is None else temperature,
        "stream": cfg.stream if stream is None else stream,
    }
    if stop:
        payload["stop"] = stop
    if cfg.keep_special_tokens:
        payload["skip_special_tokens"] = False
    if cfg.thinking in ("0", "1"):
        payload["chat_template_kwargs"] = {"enable_thinking": cfg.thinking == "1"}
    if cfg.extra_body:
        try:
            payload.update(json.loads(cfg.extra_body))
        except json.JSONDecodeError as e:
            raise LLMError(f"AGENT_EXTRA_BODY is not valid JSON: {e}") from e
    return payload


def clean_content(content: str) -> str:
    """Strip control tokens and leaked <think> text from a completed reply."""
    content = _CONTROL_TOKENS_RE.sub("", content or "")
    if "</think>" in content and "<think>" not in content:
        content = content.split("</think>", 1)[1]  # thinking leaked into content
    return content


def chat(
    cfg: Config,
    messages: list,
    stop: Optional[List[str]] = None,
    on_progress: ProgressFn = None,
    on_retry: Optional[Callable[[str, int, float], None]] = None,
    temperature: Optional[float] = None,
) -> Reply:
    """POST to {base_url}/chat/completions and return the reply, retrying on gateway errors."""
    attempts = cfg.retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return _chat_once(cfg, messages, stop, on_progress, temperature)
        except _Retryable as e:
            if attempt == attempts:
                raise LLMError(f"{e} (gave up after {attempts} attempts)") from e
            wait = min(2 ** attempt, 20)
            if on_retry:
                on_retry(str(e), attempt, wait)
            time.sleep(wait)
    raise LLMError("unreachable")


def _chat_once(cfg: Config, messages: list, stop, on_progress: ProgressFn,
               temperature: Optional[float] = None) -> Reply:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(cfg, messages, stop=stop, temperature=temperature)

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=_headers(cfg), method="POST"
    )
    _log_request(cfg, req)
    try:
        # With streaming, the timeout applies to each gap between chunks,
        # not to the whole reply.
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if cfg.stream and "text/event-stream" in ctype:
                return _read_stream(cfg, resp, on_progress)
            # Server ignored stream=true (or streaming is off): plain JSON.
            return _parse_full(cfg, json.loads(resp.read().decode()))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300].strip()
        msg = f"HTTP {e.code} from {url}: {body}"
        if e.code in RETRY_STATUS:
            raise _Retryable(msg) from e
        raise LLMError(msg) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (ConnectionResetError, TimeoutError)):
            raise _Retryable(f"Connection problem with {url}: {e.reason}") from e
        raise LLMError(f"Cannot reach {url}: {e.reason}") from e
    except (TimeoutError, ConnectionResetError) as e:
        raise _Retryable(f"No data from {url} for {cfg.timeout}s ({e.__class__.__name__})") from e


def iter_stream_events(cfg: Config, resp):
    """Yield parsed Server-Sent Events from a streaming /chat/completions response.

    Each event is one of:
      ("content", text)   a piece of the reply
      ("thinking", text)  a piece of the model's reasoning
      ("tool", (index, name_delta, args_delta))  a tool-call delta
      ("finish", reason)  the finish_reason for a choice

    The response is lines like 'data: {json}' ending with 'data: [DONE]'.
    Shared by llm.chat and web.py so the SSE wire format lives in one place.
    """
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if cfg.debug:
            print("[stream]", data[:300])
        log_http(cfg, "REPLY chunk", data)
        if "error" in chunk:
            raise LLMError(f"Server error during streaming: {json.dumps(chunk['error'])[:300]}")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield ("content", delta["content"])
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                yield ("thinking", think)
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                yield ("tool", (tc.get("index"), fn.get("name"), fn.get("arguments")))
            if choice.get("finish_reason"):
                yield ("finish", choice["finish_reason"])


def _read_stream(cfg: Config, resp, on_progress: ProgressFn) -> Reply:
    """Accumulate the shared stream events into a finished Reply."""
    content, reasoning, finish = [], [], ""
    n_content = n_reasoning = 0
    tools = {}  # index -> {"name": str, "arguments": str}, built up from deltas
    for kind, value in iter_stream_events(cfg, resp):
        if kind == "content":
            content.append(value)
            n_content += len(value)
            if on_progress:
                on_progress("writing", n_content)
        elif kind == "thinking":
            reasoning.append(value)
            n_reasoning += len(value)
            if on_progress:
                on_progress("thinking", n_reasoning)
        elif kind == "tool":
            index, name, args = value
            slot = tools.setdefault(index if index is not None else len(tools),
                                    {"name": "", "arguments": ""})
            slot["name"] += name or ""
            if isinstance(args, dict):
                slot["arguments"] = args
            elif args:
                slot["arguments"] += args
            if on_progress:
                on_progress("calling a tool", len(tools))
        elif kind == "finish":
            finish = value
    calls = [tools[i] for i in sorted(tools)]
    return Reply("".join(content), finish, "".join(reasoning), calls)


def _parse_full(cfg: Config, data: dict) -> Reply:
    log_http(cfg, "REPLY", json.dumps(data, indent=2))
    if cfg.debug:
        print("\n----- RAW RESPONSE -----\n" + json.dumps(data, indent=2)[:4000] + "\n------------------------\n")
    try:
        choice = data["choices"][0]
        msg = choice["message"]
        return Reply(
            content=msg.get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=[{"name": (tc.get("function") or {}).get("name", ""),
                         "arguments": (tc.get("function") or {}).get("arguments", "")}
                        for tc in msg.get("tool_calls") or []],
        )
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {json.dumps(data)[:500]}") from e


def open_stream(cfg: Config, messages: list, temperature: Optional[float] = None):
    """Open a streaming /chat/completions request. Returns the urlopen context manager.

    Use with `iter_stream_events` to consume deltas. Callers that need
    cancellation between chunks iterate the events themselves (see web.py).
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(cfg, messages, stream=True, temperature=temperature)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=_headers(cfg), method="POST"
    )
    _log_request(cfg, req)
    return urllib.request.urlopen(req, timeout=cfg.timeout)


def list_models(cfg: Config) -> list:
    """GET {base_url}/models: handy for checking the server and model name."""
    url = cfg.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers=_headers(cfg))
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return [m["id"] for m in data.get("data", [])]


def switch_model(cfg: Config, name: str, console) -> None:
    """Handle the /model command for both the agent and coder REPLs.

    With no name, show the current model and the server's available ones.
    With a name, point cfg.model at it; every later request reads cfg.model,
    so the switch takes effect on the next message with no history reset.
    """
    name = name.strip()
    if not name:
        console.print(f"[dim]Current model: {cfg.model}[/dim]")
        try:
            models = list_models(cfg)
        except Exception as e:
            console.print(f"[yellow]Could not list models from {cfg.base_url}: {e}[/yellow]")
            console.print("[dim]Usage: /model <name>[/dim]")
            return
        if models:
            console.print("[dim]Available:[/dim]")
            for m in models:
                marker = " [green]*[/green]" if m == cfg.model else ""
                console.print(f"  {m}{marker}")
        console.print("[dim]Switch with: /model <name>[/dim]")
        return

    # Warn if the name isn't one the server advertises, but still allow it:
    # some endpoints don't list every model they accept.
    try:
        models = list_models(cfg)
    except Exception:
        models = []
    if models and name not in models:
        console.print(f"[yellow]'{name}' is not in the server's model list; "
                      "switching anyway.[/yellow]")
    cfg.model = name
    console.print(f"[green]Model: {cfg.model}[/green]")
