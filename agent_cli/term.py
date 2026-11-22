"""Make untrusted text (model, files, commands, servers) safe to print."""
import re

from rich.markup import escape

# What can drive the terminal or reorder a line: C0/C1 controls except newline
# and tab, bidi overrides, and lone surrogates (which can't be printed at all).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
# Invisible, though str.isprintable() says otherwise.
_BLANK_RE = re.compile("[\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180f\u3164\ufe00-\ufe0f"
                       "\uffa0\U000e0100-\U000e01ef]")


def _code(char: str) -> str:
    return char.encode("unicode_escape").decode("ascii")


def safe_text(text: str) -> str:
    """For display: terminal controls become visible codes such as \\x1b."""
    return _CONTROL_RE.sub(lambda m: _code(m.group()), text.replace("\r\n", "\n"))


def safe_markup(text: str) -> str:
    """safe_text, for use inside rich markup."""
    return escape(safe_text(text))


def reveal(text: str) -> str:
    """For approvals, stricter: every character that can't be seen becomes a code."""
    return "".join(c if c in "\n\t" or (c.isprintable() and not _BLANK_RE.match(c)) else _code(c)
                   for c in text)
