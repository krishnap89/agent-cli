"""Turn headers: make each prompt easy to spot when scrolling a long chat."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from rich.console import Group
from rich.rule import Rule
from rich.text import Text

BAR = "▌ "


def prompt_header(n: int, text: Optional[str], mode: str, color: str,
                  when: Optional[datetime] = None) -> Group:
    """A full-width rule titled "#n · mode · HH:MM", then the message with a
    colored bar on each line. With text=None only the rule is shown (for when
    the typed text is still on screen above it)."""
    when = when or datetime.now()
    title = Text(f" #{n} · {mode} · {when:%H:%M} ", style=f"bold {color}")
    parts = [Text(""), Rule(title, style=color, characters="━", align="left")]
    if text is not None:
        body = Text()
        for i, line in enumerate(text.splitlines() or [""]):
            if i:
                body.append("\n")
            body.append(BAR, style=color)
            body.append(line, style="bold")  # Text, so [brackets] aren't markup
        parts.append(body)
    parts.append(Text(""))
    return Group(*parts)
