"""One-shot writing helpers: grammar, rephrase, tone, email, slack.

These are general-purpose, non-coding commands for everyday writing — fixing
grammar, rephrasing a sentence, adjusting tone, or drafting an email or Slack
message. They don't touch files or tools: each takes the text you pass and asks
the model to transform it, then prints the result.

Each command runs a single model call with its own system prompt. No history is
kept, so one command never bleeds into the next.
"""
from __future__ import annotations

from typing import Optional

from rich.console import Console

from .config import Config

console = Console()

# Shared rule: return only the rewritten text, nothing else. Local models love
# to add "Here is your corrected text:" preambles, so we forbid them explicitly.
_BARE = ("Return only the resulting text, with no preamble, no explanation, no "
         "quotes around it, and no commentary. Preserve the original meaning.")

GRAMMAR_PROMPT = (
    "You are a careful copy editor. Fix spelling, grammar, punctuation and "
    "awkward phrasing in the user's text. Keep their voice and wording where "
    "it is already correct; change as little as possible. " + _BARE
)

REPHRASE_PROMPT = (
    "You are an editor. Rewrite the user's text so it reads more clearly and "
    "naturally, keeping the same meaning and roughly the same length. " + _BARE
)

TONE_PROMPT = (
    "You are an editor. Rewrite the user's text in a {tone} tone, keeping the "
    "same meaning. " + _BARE
)

EMAIL_PROMPT = (
    "You turn rough notes into a clear, professional email. Keep it concise and "
    "well structured; add a short greeting and sign-off only if they fit. Do not "
    "invent facts the notes don't contain. If a detail is clearly missing, leave "
    "a short [bracketed placeholder]. " + _BARE
)

SLACK_PROMPT = (
    "You turn rough notes into a clear, friendly Slack message. Keep it short "
    "and conversational — no formal greeting or sign-off. Do not invent facts "
    "the notes don't contain. " + _BARE
)


def _run(cfg: Config, system: str, text: str) -> None:
    """Make one model call with the given system prompt and print the result."""
    from .agent import THINK_RE
    from .llm import chat, LLMError

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    try:
        with console.status("[dim]waiting for model...[/dim]"):
            reply = chat(cfg, messages)
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        return
    result = THINK_RE.sub("", reply.content).strip()
    if result:
        console.print(result, markup=False, highlight=False)
    else:
        console.print("[yellow]Empty reply from the model.[/yellow]")


def handle(cfg: Config, cmd: str, arg: str) -> bool:
    """Dispatch a writing command. Returns True if `cmd` was a writing command.

    `arg` is the text after the command word (may be empty, meaning the user
    passed no inline text).
    """
    if cmd == "/grammar":
        system, body = GRAMMAR_PROMPT, arg
    elif cmd == "/rephrase":
        system, body = REPHRASE_PROMPT, arg
    elif cmd == "/email":
        system, body = EMAIL_PROMPT, arg
    elif cmd == "/slack":
        system, body = SLACK_PROMPT, arg
    elif cmd == "/tone":
        # First word is the tone (e.g. "formal", "friendly"); rest is the text.
        parts = arg.split(None, 1)
        if len(parts) < 2:
            console.print('[yellow]Usage: /tone <style> <text>  '
                          '(e.g. /tone formal Can you send that over?)[/yellow]')
            return True
        tone, body = parts[0], parts[1]
        system = TONE_PROMPT.format(tone=tone)
    else:
        return False

    if not body.strip():
        console.print(f"[yellow]Give some text: {cmd} <text>  "
                      'or start a multi-line block with """[/yellow]')
        return True
    _run(cfg, system, body)
    return True
