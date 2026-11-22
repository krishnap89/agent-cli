"""Edit text files without damaging them.

Code works on text with \\n line endings. encode() turns that back into bytes with
the file's own endings, and what isn't UTF-8 text is refused, never guessed at.
"""
import difflib
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

_BREAK = re.compile(r"(\r\n|\r|\n)")  # the breaks read_text() knows, unlike splitlines()
_BOM = "﻿"


class NotEditable(Exception):
    pass


def lines(text: str) -> List[str]:
    """splitlines() that breaks only at line endings, not at form feeds or U+2028."""
    out = _BREAK.split(text)[0::2]
    if out[-1] == "":
        out.pop()
    return out


def _decode(path: Path) -> str:
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError as e:
        raise NotEditable(f"it can't be read ({e.strerror})")
    except UnicodeDecodeError:
        text = None
    if text is None or "\0" in text:  # NUL: binary, or UTF-16 without a BOM
        raise NotEditable("it is not UTF-8 text, so editing would corrupt it")
    return text


def read(path: Path) -> str:
    """The file's text with \\n endings. Raises NotEditable."""
    return _BREAK.sub("\n", _decode(path))


def problem(path: Path) -> Optional[str]:
    """Why an existing file can't be edited, or None."""
    if path.is_file():
        try:
            _decode(path)
        except NotEditable as e:
            return str(e)
    return None


def encode(path: Path, text: str) -> bytes:
    """The bytes to write for `text`. Unchanged lines keep their ending, new lines
    get the file's most common one, a BOM and a missing final newline stay.
    Encodes before anything is written, so a failure can't truncate the file."""
    old = _decode(path) if path.exists() else ""
    parts = _BREAK.split(old)
    bodies, ends = parts[0::2], parts[1::2]
    new = _BREAK.sub("\n", text).split("\n")

    if old.startswith(_BOM) and not new[0].startswith(_BOM):
        new[0] = _BOM + new[0]
    # apply_edit always ends in a newline: drop it if the last line is untouched
    if bodies[-1] and len(new) > 1 and not new[-1] and new[-2] == bodies[-1]:
        new.pop()

    usual = Counter(ends).most_common(1)[0][0] if ends else os.linesep
    if len(set(ends)) < 2:
        joined = usual.join(new)
    else:
        joined = "".join(b + e for b, e in zip(new, _endings(bodies, ends, new, usual) + [""]))
    try:
        return joined.encode("utf-8")
    except UnicodeEncodeError:
        raise NotEditable("the new text has characters that UTF-8 can't store")


def _endings(bodies: List[str], ends: List[str], new: List[str], usual: str) -> List[str]:
    """For a file with mixed endings: give each line of `new` the ending it had."""
    head, shortest = 0, min(len(bodies), len(new))
    while head < shortest and bodies[head] == new[head]:
        head += 1
    tail = 0
    while tail < shortest - head and bodies[-1 - tail] == new[-1 - tail]:
        tail += 1
    pairs = [(i, i) for i in range(head)]
    pairs += [(len(bodies) - k, len(new) - k) for k in range(tail, 0, -1)]
    # autojunk would lose track of repeated lines such as "}" in a long file
    matcher = difflib.SequenceMatcher(None, bodies[head:len(bodies) - tail],
                                      new[head:len(new) - tail], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            pairs += [(head + i, head + j) for i, j in zip(range(i1, i2), range(j1, j2))]
    out = [usual] * (len(new) - 1)
    for i, j in pairs:
        if i < len(ends) and j < len(out):
            out[j] = ends[i]
    return out
