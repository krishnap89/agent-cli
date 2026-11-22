"""Turn headers: make each prompt easy to spot when scrolling a long chat."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.rule import Rule
from rich.text import Text

from .term import safe_text

BAR = "▌ "


def _fit_path(path: str, room: int) -> str:
    """Shorten from the left, at a "/" if possible: the folder's name is at the end."""
    if cell_len(path) <= room:
        return path
    if room < 2:
        return ""
    for i, char in enumerate(path):
        if i and char == "/" and cell_len(path[i:]) < room:
            return "…" + path[i:]
    while cell_len(path) >= room:
        path = path[1:]
    return "…" + path


class _TitleRule:
    """The rule, with the path fitted to the width. Rule alone cuts the title's
    end, which would lose the folder's name and the time."""

    def __init__(self, start: str, path: str, end: str, color: str):
        self.start, self.end, self.color = start, end, color
        self.path = safe_text(path).replace("\n", " ").replace("\t", " ")

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        separator = " · "
        used = cell_len(self.start + separator + self.end) + 2  # 2: what Rule keeps for itself
        path = _fit_path(self.path, options.max_width - used) if self.path else ""
        middle = separator + path if path else ""
        title = Text(self.start + middle + self.end, style=f"bold {self.color}")
        yield Rule(title, style=self.color, characters="━", align="left")


def prompt_header(n: int, text: Optional[str], mode: str, color: str,
                  when: Optional[datetime] = None, path: str = "") -> Group:
    """A full-width rule titled "#n · mode · path · HH:MM", then the message with a
    colored bar on each line. With text=None only the rule is shown (for when
    the typed text is still on screen above it)."""
    when = when or datetime.now()
    parts = [Text(""), _TitleRule(f" #{n} · {mode}", path, f" · {when:%H:%M} ", color)]
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
