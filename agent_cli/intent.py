"""Classify a natural-language message into a special-command intent.

The coder and agent loops have purpose-built pipelines behind slash commands
(/test-gen, /doc). Without this, a message like "add unit tests for
NotificationInteractor" falls through to the generic chat loop, which just
answers with a code block instead of running the pipeline. This module asks the
model to map such a message to an intent and target so the caller can route it.

The classifier is deliberately conservative: on any doubt, parse failure, or
error it returns a `chat` intent (or None), so routing never does worse than the
existing fallthrough. The caller still confirms before diverting.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from rich.console import Console
from rich.prompt import Confirm

from .config import Config
from .llm import LLMError, chat

# Intents the classifier can emit. "chat" means "not a special command: send it
# to the normal loop" — the safe default.
TEST_GEN = "test_gen"
DOC_GEN = "doc_gen"
CHAT = "chat"
_VALID = {TEST_GEN, DOC_GEN, CHAT}


@dataclass
class Intent:
    kind: str                 # TEST_GEN | DOC_GEN | CHAT
    target: str = ""          # class name or file path, when kind != CHAT
    method: str = ""          # optional single method, test_gen only


# Only pay for a classifier call when the message plausibly asks for one of the
# pipelines. This pre-filter keeps "hi" / "explain this" / "fix the bug" on the
# fast path (no LLM round-trip) while letting anything test/doc-shaped through.
_CANDIDATE_RE = re.compile(
    r"\b(test|tests|testing|unit\s*test|spec|coverage|"
    r"doc|docs|document|documentation|kdoc|javadoc|comment)\b",
    re.IGNORECASE,
)


def looks_like_candidate(text: str) -> bool:
    """Cheap gate: could this message be a test-gen / doc-gen request?"""
    # A target is almost always named; a bare "write tests" with nothing in the
    # chat is still worth classifying, so we only require the topic keyword.
    return bool(_CANDIDATE_RE.search(text))


_SYSTEM = """You route a developer's message to one of three actions. Reply with ONE JSON object and nothing else.

Actions:
- "test_gen": the user wants unit tests generated for a specific Kotlin class or file.
- "doc_gen": the user wants documentation (KDoc/comments) generated for a class, file, or folder.
- "chat": anything else — questions, explanations, bug fixes, edits, or a request too vague to name a target.

Rules:
- Only choose test_gen or doc_gen when the message clearly asks to GENERATE tests/docs AND names a target (a class name like NotificationInteractor, or a path like app/src/.../Foo.kt).
- "why does this test fail", "explain the tests", "fix the failing test" are chat, not test_gen.
- If unsure, choose chat.
- "target": the class name or file path, copied verbatim. Empty if none.
- "method": a single method name if the user restricts tests to one method, else empty.

Format: {"intent": "test_gen"|"doc_gen"|"chat", "target": "", "method": ""}"""


def classify(cfg: Config, text: str) -> Optional[Intent]:
    """Classify `text`. Returns an Intent, or None if the model was unreachable.

    A None result means "could not decide" and the caller should fall back to
    normal chat. A CHAT intent means "decided it's not a command".
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": text.strip()[:2000]},
    ]
    try:
        # Low temperature for a deterministic label; no streaming needed for a
        # one-shot JSON reply.
        reply = chat(cfg, messages, temperature=0.0)
    except LLMError:
        return None
    return _parse(reply.content)


def _parse(raw: str) -> Intent:
    """Pull the JSON object out of the reply and validate it. Falls back to CHAT."""
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return Intent(CHAT)
    kind = str(obj.get("intent", "")).strip()
    if kind not in _VALID:
        return Intent(CHAT)
    target = _clean_target(str(obj.get("target", "") or ""))
    method = str(obj.get("method", "") or "").strip()
    # A command intent with no target is unusable; treat it as chat so the
    # caller doesn't prompt for a target it was never given.
    if kind in (TEST_GEN, DOC_GEN) and not target:
        return Intent(CHAT)
    return Intent(kind, target, method)


def _clean_target(target: str) -> str:
    """Normalize a target the model copied from the message.

    Strips surrounding quotes/backticks and a leading '@' — the file-mention
    sigil the user types (e.g. "@app/.../Foo.kt"). resolve_target wants the bare
    path or class name, not the sigil.
    """
    t = target.strip().strip("`\"'").strip()
    return t[1:].lstrip() if t.startswith("@") else t


_LABELS = {TEST_GEN: "generate tests for", DOC_GEN: "generate docs for"}


def route(cfg: Config, coder: Any, text: str, console: Console) -> bool:
    """Try to route a natural-language message to a pipeline on `coder`.

    Returns True if it dispatched to test_gen/doc_gen (so the caller should not
    also send the message to the normal loop). Returns False when the message is
    ordinary chat, routing is off, the model was unreachable, or the user
    declined the detected action — in every one of those cases the caller falls
    back to its usual handling.

    `coder` must provide test_gen(target, method) and doc_gen(target).
    """
    if not cfg.intent_routing or not looks_like_candidate(text):
        return False
    with console.status("[dim]routing...[/dim]"):
        intent = classify(cfg, text)
    if intent is None or intent.kind == CHAT:
        return False

    label = _LABELS[intent.kind]
    detail = f"{label} {intent.target}"
    if intent.method:
        detail += f" (method {intent.method})"
    console.print(f"[dim]\N{RIGHTWARDS ARROW} Detected: {detail}[/dim]")
    if not (cfg.auto_approve or Confirm.ask("Run this?", default=True)):
        # Not what they meant: fall back so the message is handled as chat.
        return False

    if intent.kind == TEST_GEN:
        coder.test_gen(intent.target, intent.method)
    else:
        coder.doc_gen(intent.target)
    return True


def _extract_json(raw: str) -> Optional[dict]:
    """Find the first balanced {...} object in the text and parse it."""
    text = raw.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
