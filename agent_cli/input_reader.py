"""Read user input in the REPL, keeping multi-line pastes together.

The terminal delivers pasted text line by line, so a plain input() call would
send each line as a separate message. Two ways to send multi-line text:

1. Paste detection (macOS/Linux): the terminal's "bracketed paste" mode marks
   where a paste starts and ends, so the whole paste is kept together. (For
   terminals without it, lines arriving within a few ms count as a paste.)
   After a paste you can type more (e.g. "explain this"), then press Enter
   on an empty line to send.
2. Explicit block (any OS): type \"\"\" on its own line, paste or type anything,
   then \"\"\" on its own line again to send.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from prompt_toolkit.completion import Completer

console = Console()

PASTE_GAP = 0.05  # seconds: lines arriving faster than this are part of a paste
BLOCK_MARK = '"""'
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"

try:
    import select

    def _more_input_waiting(timeout: float) -> bool:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(ready)

    _CAN_DETECT_PASTE = sys.stdin.isatty()
except (ImportError, OSError, ValueError):  # e.g. Windows consoles
    _CAN_DETECT_PASTE = False


# ---- prompt_toolkit integration ------------------------------------------------

_pt_available = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
    _pt_available = True
except ImportError:
    pass

_session: Optional[PromptSession] = None
_completer: Optional[Completer] = None
_last_erased = False  # did the last read_message() erase the typed text?


def set_completer(completer: Completer) -> None:
    global _completer
    _completer = completer


def _get_session() -> PromptSession:
    global _session
    if _session is None:
        _session = PromptSession(history=InMemoryHistory())
    return _session


def _strip_rich_markup(prompt: str) -> str:
    """Convert a Rich markup prompt to prompt_toolkit HTML."""
    import re
    tag_re = re.compile(r"\[/?[^\]]*\]")
    return tag_re.sub("", prompt)


def _rich_to_pt(prompt: str) -> FormattedText:
    """Convert a Rich markup prompt for prompt_toolkit. Not through its HTML, which
    is XML and fails on a folder name with & or <."""
    parts = []
    for segment in Text.from_markup(prompt, emoji=False).render(console):
        style = segment.style
        names = []
        if style and style.dim:
            names.append("fg:gray")
        elif style and style.color:
            names.append(f"fg:{style.color.name}")
        if style and style.bold:
            names.append("bold")
        parts.append((" ".join(names), segment.text))
    return FormattedText(parts)


# ---- fallback (non-prompt_toolkit) path ----------------------------------------

def _drain_pasted_lines() -> list:
    """Collect lines that are already waiting (the rest of a paste)."""
    lines = []
    while _more_input_waiting(PASTE_GAP):
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        lines.append(line.rstrip("\n"))
    return lines


def _read_block() -> str:
    console.print(f'[dim]Multi-line mode: paste or type, then {BLOCK_MARK} on its own line to send.[/dim]')
    lines = []
    while True:
        line = console.input("[dim]... [/dim]")
        if line.strip() == BLOCK_MARK:
            return "\n".join(lines)
        lines.append(line)


class _BracketedPaste:
    """Turn on bracketed paste and hide the marker echo while reading input."""

    def __enter__(self):
        self.saved = None
        try:
            import termios
            fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(fd)
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~getattr(termios, "ECHOCTL", 0)  # don't echo ESC as ^[
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            sys.stdout.write("\x1b[?2004h")
            sys.stdout.flush()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        try:
            sys.stdout.write("\x1b[?2004l")
            sys.stdout.flush()
            if self.saved is not None:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self.saved)
        except Exception:
            pass


def _clean(line: str) -> str:
    return line.replace(PASTE_START, "").replace(PASTE_END, "")


# ---- public API ----------------------------------------------------------------

def was_erased() -> bool:
    """True if the last read_message(erase=True) removed the typed text from
    the screen, so the caller should print it again (e.g. as a turn header)."""
    return _last_erased


def read_message(prompt: str, erase: bool = False) -> str:
    """Read one message. Raises EOFError / KeyboardInterrupt like input().

    With erase=True the typed text is cleared from the screen once submitted,
    where the input method allows it (prompt_toolkit, single message). Check
    was_erased() afterwards to know whether it was.
    """
    text = _read_message(prompt, erase)
    # A long message is almost always pasted (a log, a ticket, a data dump):
    # strict masking mode has the local PII model look it over before sending.
    from . import redact
    redact.note_untrusted(text, "paste")
    return text


def _read_message(prompt: str, erase: bool = False) -> str:
    global _last_erased
    _last_erased = False
    if _pt_available and _completer is not None and sys.stdin.isatty():
        return _read_with_prompt_toolkit(prompt, erase)

    if not _CAN_DETECT_PASTE:
        first = console.input(prompt, emoji=False)
        return _read_block() if first.strip() == BLOCK_MARK else first

    with _BracketedPaste():
        return _read_with_paste_detection(prompt)


def _read_with_prompt_toolkit(prompt: str, erase: bool = False) -> str:
    global _last_erased
    session = _get_session()
    pt_prompt = _rich_to_pt(prompt)
    # A completer can opt into showing the menu as you type (the agent REPL's
    # command completer does); otherwise the menu waits for Tab (coder REPL).
    while_typing = getattr(_completer, "complete_while_typing", False)
    # prompt() takes no erase_when_done; the Application reads it on exit.
    session.app.erase_when_done = erase
    text = session.prompt(pt_prompt, completer=_completer, complete_while_typing=while_typing)
    if text.strip() == BLOCK_MARK:
        # The block's lines are read with console.input and stay on screen.
        return _read_block()
    _last_erased = erase
    return text


def _read_with_paste_detection(prompt: str) -> str:
    first = console.input(prompt, emoji=False)

    if first.strip() == BLOCK_MARK:
        return _read_block()

    if PASTE_START in first and PASTE_END not in first:
        # Multi-line paste: read until the end marker. The last line arrives
        # once you press Enter, even if the copied text had no final newline.
        lines = [_clean(first)]
        while True:
            line = console.input("")
            lines.append(_clean(line))
            if PASTE_END in line:
                break
        lines += _drain_pasted_lines()
    else:
        lines = [_clean(first)] + _drain_pasted_lines()  # fallback detection

    if len(lines) == 1:
        return lines[0]  # typed line, or a single-line paste

    # It was a paste. The last pasted line may still be waiting for Enter,
    # so keep reading until an empty line.
    console.print(
        f"[dim]({len(lines)} lines pasted. Type more if you like, "
        "then press Enter on an empty line to send.)[/dim]"
    )
    while True:
        line = console.input("[dim]... [/dim]")
        if line == "":
            break
        lines.append(_clean(line))
        lines.extend(_clean(l) for l in _drain_pasted_lines())
    return "\n".join(lines).strip("\n")
