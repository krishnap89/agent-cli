"""Plain-HTTP client for an OpenAI-compatible /chat/completions endpoint.

Uses only the Python standard library (no openai SDK, no requests).

Two features protect against 504 Gateway Timeout errors:
- Streaming: the server sends the reply piece by piece, so proxies and load
  balancers in between see constant traffic and don't time out.
- Retries: 502/503/504 and dropped connections are retried with a short wait.
"""
import json
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from .config import Config

RETRY_STATUS = {429, 502, 503, 504}

# on_progress(phase, chars) is called while streaming; phase is "thinking" or "writing"
ProgressFn = Optional[Callable[[str, int], None]]


class LLMError(Exception):
    pass


class _Retryable(Exception):
    pass


class Reply:
    def __init__(self, content: str, finish_reason: str, reasoning: str):
        self.content = content
        self.finish_reason = finish_reason  # "stop", "length", ...
        self.reasoning = reasoning          # thinking text, if the server splits it out


def _headers(cfg: Config) -> dict:
    h = {"Content-Type": "application/json"}
    if cfg.api_key and cfg.api_key != "not-needed":
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def chat(
    cfg: Config,
    messages: list,
    stop: Optional[List[str]] = None,
    on_progress: ProgressFn = None,
    on_retry: Optional[Callable[[str, int, float], None]] = None,
) -> Reply:
    """POST to {base_url}/chat/completions and return the reply, retrying on gateway errors."""
    attempts = cfg.retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return _chat_once(cfg, messages, stop, on_progress)
        except _Retryable as e:
            if attempt == attempts:
                raise LLMError(f"{e} (gave up after {attempts} attempts)") from e
            wait = min(2 ** attempt, 20)
            if on_retry:
                on_retry(str(e), attempt, wait)
            time.sleep(wait)
    raise LLMError("unreachable")


def _chat_once(cfg: Config, messages: list, stop, on_progress: ProgressFn) -> Reply:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": cfg.stream,
    }
    if stop:
        payload["stop"] = stop

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=_headers(cfg), method="POST"
    )
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


def _read_stream(cfg: Config, resp, on_progress: ProgressFn) -> Reply:
    """Read Server-Sent Events: lines like 'data: {json}' ending with 'data: [DONE]'."""
    content, reasoning, finish = [], [], ""
    n_content = n_reasoning = 0
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
        if "error" in chunk:
            raise LLMError(f"Server error during streaming: {json.dumps(chunk['error'])[:300]}")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
                n_content += len(delta["content"])
                if on_progress:
                    on_progress("writing", n_content)
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                reasoning.append(think)
                n_reasoning += len(think)
                if on_progress:
                    on_progress("thinking", n_reasoning)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return Reply("".join(content), finish, "".join(reasoning))


def _parse_full(cfg: Config, data: dict) -> Reply:
    if cfg.debug:
        print("\n----- RAW RESPONSE -----\n" + json.dumps(data, indent=2)[:4000] + "\n------------------------\n")
    try:
        choice = data["choices"][0]
        msg = choice["message"]
        return Reply(
            content=msg.get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
        )
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {json.dumps(data)[:500]}") from e


def list_models(cfg: Config) -> list:
    """GET {base_url}/models: handy for checking the server and model name."""
    url = cfg.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers=_headers(cfg))
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return [m["id"] for m in data.get("data", [])]
