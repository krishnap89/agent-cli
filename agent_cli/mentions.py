"""Expand @path mentions in a user message by inlining the file contents.

When a message mentions a repo file as ``@agent_cli/main.py``, we read that
file and append its contents to the message so the model sees them directly —
in chat mode (no tools) as well as agent mode. The original ``@path`` is left
in the text so each mention maps to its appended block.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# An @path token: '@' followed by a path (letters, digits, / . _ -). We match
# anywhere in the text, but not inside an email-like "foo@bar" — the '@' must
# start the token (preceded by start-of-string or whitespace).
_MENTION_RE = re.compile(r"(?:(?<=\s)|^)@([\w./\\-]+)")

# Cap per file so one @mention can't blow the context window.
MAX_FILE_CHARS = 20_000


def find_mentions(text: str) -> List[str]:
    """Return the unique @paths in `text`, in first-seen order."""
    seen = []  # type: List[str]
    for m in _MENTION_RE.finditer(text):
        path = m.group(1)
        if path not in seen:
            seen.append(path)
    return seen


def expand(text: str, root: Path) -> Tuple[str, List[str]]:
    """Inline the contents of each @path file mentioned in `text`.

    Returns (expanded_text, notes). `notes` is a list of short status lines to
    show the user (one per mention): what was attached, or why it wasn't.
    Files are resolved under `root`; a path escaping it is refused.
    """
    mentions = find_mentions(text)
    if not mentions:
        return text, []

    blocks = []  # type: List[str]
    notes = []  # type: List[str]
    for rel in mentions:
        try:
            resolved = (root / rel).resolve()
            if not resolved.is_relative_to(root.resolve()):
                notes.append(f"{rel}: outside the working directory, not attached")
                continue
            if not resolved.exists():
                notes.append(f"{rel}: not found, not attached")
                continue
            if resolved.is_dir():
                notes.append(f"{rel}: is a directory, not attached")
                continue
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            notes.append(f"{rel}: could not read ({e}), not attached")
            continue

        if not content.strip():
            notes.append(f"{rel}: empty, not attached")
            continue

        truncated = ""
        if len(content) > MAX_FILE_CHARS:
            dropped = len(content) - MAX_FILE_CHARS
            content = content[:MAX_FILE_CHARS]
            truncated = f" (truncated {dropped:,} chars)"
        blocks.append(f"--- {rel} ---\n{content}\n--- end {rel} ---")
        notes.append(f"attached {rel} ({len(content):,} chars){truncated}")

    if not blocks:
        return text, notes
    return text + "\n\n" + "\n\n".join(blocks), notes
