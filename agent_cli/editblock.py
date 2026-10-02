"""Parse and apply SEARCH/REPLACE edit blocks (the edit format popularised by Aider).

The model writes edits as plain text:

    path/to/file.py
    <<<<<<< SEARCH
    def old():
        return 1
    =======
    def old():
        return 2
    >>>>>>> REPLACE

An empty SEARCH section creates a new file (or appends to an empty one).

Matching is tolerant, because small models often get whitespace slightly wrong:
1. exact text match
2. match ignoring trailing whitespace on each line
3. match ignoring indentation (the replacement is re-indented to fit)
If nothing matches, the error includes the most similar lines from the file so
the model can correct itself.
"""
import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

HEAD_RE = re.compile(r"^\s*<{5,9} ?SEARCH\b.*$")
DIVIDER_RE = re.compile(r"^\s*={5,9}\s*$")
UPDATED_RE = re.compile(r"^\s*>{5,9} ?REPLACE\b.*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class EditBlock:
    path: str
    search: str
    replace: str


class EditError(Exception):
    pass


def _clean_filename(line: str) -> Optional[str]:
    """Turn a line like '`src/app.py`:' or '# File: src/app.py' into 'src/app.py'."""
    s = line.strip()
    if not s or FENCE_RE.match(s) and len(s) <= 3:
        return None
    s = re.sub(r"^(#+|//|\*+)\s*", "", s)
    s = re.sub(r"^(file|filename|path)\s*:\s*", "", s, flags=re.I)
    s = s.strip("`*\"': ")
    if FENCE_RE.match(line.strip()):  # e.g. ```python src/app.py
        parts = line.strip().strip("`~").split()
        s = parts[-1] if len(parts) > 1 else ""
    if not s or " " in s or len(s) > 200:
        return None
    if "." not in s and "/" not in s:
        return None
    return s


def parse_edit_blocks(text: str, default_path: Optional[str] = None) -> Tuple[List[EditBlock], List[str]]:
    """Find all SEARCH/REPLACE blocks. Returns (blocks, problems)."""
    lines = text.splitlines()
    blocks, problems = [], []
    last_path = default_path
    i = 0
    while i < len(lines):
        if not HEAD_RE.match(lines[i]):
            i += 1
            continue
        # Filename: look back up to 3 lines (skipping fences / blank lines).
        path = None
        for j in range(i - 1, max(i - 4, -1), -1):
            path = _clean_filename(lines[j])
            if path:
                break
        path = path or last_path
        i += 1
        search, replace = [], []
        while i < len(lines) and not DIVIDER_RE.match(lines[i]):
            if HEAD_RE.match(lines[i]) or UPDATED_RE.match(lines[i]):
                break
            search.append(lines[i])
            i += 1
        if i >= len(lines) or not DIVIDER_RE.match(lines[i]):
            problems.append("A SEARCH section was not followed by a ======= divider line.")
            continue
        i += 1
        while i < len(lines) and not UPDATED_RE.match(lines[i]):
            if HEAD_RE.match(lines[i]):
                break
            replace.append(lines[i])
            i += 1
        if i >= len(lines) or not UPDATED_RE.match(lines[i]):
            problems.append("A block was missing its closing >>>>>>> REPLACE line.")
            continue
        i += 1
        if not path:
            problems.append("A SEARCH/REPLACE block had no file path on the line before it.")
            continue
        last_path = path
        blocks.append(EditBlock(path, _join(search), _join(replace)))
    return blocks, problems


def _join(lines: List[str]) -> str:
    return "\n".join(lines) + "\n" if lines else ""


# ---- Applying -------------------------------------------------------------------

def apply_edit(content: str, search: str, replace: str) -> str:
    """Return new content, or raise EditError with a helpful message."""
    if not search.strip():
        # Empty SEARCH: create file / append.
        if content and not content.endswith("\n"):
            content += "\n"
        return content + replace

    if not content.endswith("\n"):
        content += "\n"

    # 1. Exact
    if search in content:
        return content.replace(search, replace, 1)

    c_lines = content.splitlines()
    s_lines = search.splitlines()
    r_lines = replace.splitlines()
    while s_lines and not s_lines[0].strip():  # drop stray blank lines at the edges
        s_lines.pop(0)
    while s_lines and not s_lines[-1].strip():
        s_lines.pop()
    if not s_lines:
        raise EditError("The SEARCH section only contained blank lines.")
    n = len(s_lines)

    # 2. Ignore trailing whitespace
    target = [l.rstrip() for l in s_lines]
    for start in range(len(c_lines) - n + 1):
        if [l.rstrip() for l in c_lines[start:start + n]] == target:
            return _splice(c_lines, start, n, r_lines)

    # 3. Ignore indentation; shift the replacement by the indentation difference
    target = [l.strip() for l in s_lines]
    for start in range(len(c_lines) - n + 1):
        window = c_lines[start:start + n]
        if [l.strip() for l in window] == target:
            first = next(k for k, l in enumerate(s_lines) if l.strip())
            have = _indent(window[first])
            wrote = _indent(s_lines[first])
            return _splice(c_lines, start, n, _reindent(r_lines, wrote, have))

    raise EditError(_no_match_message(c_lines, s_lines))


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(lines: List[str], old: str, new: str) -> List[str]:
    out = []
    for l in lines:
        if not l.strip():
            out.append(l)
        elif l.startswith(old):
            out.append(new + l[len(old):])
        else:
            out.append(l)
    return out


def _splice(c_lines: List[str], start: int, n: int, r_lines: List[str]) -> str:
    new = c_lines[:start] + r_lines + c_lines[start + n:]
    return "\n".join(new) + "\n"


def _no_match_message(c_lines: List[str], s_lines: List[str]) -> str:
    n = len(s_lines)
    best, best_ratio = 0, 0.0
    target = "\n".join(l.strip() for l in s_lines)
    for start in range(max(len(c_lines) - n + 1, 1)):
        window = "\n".join(l.strip() for l in c_lines[start:start + n])
        r = difflib.SequenceMatcher(None, target, window).quick_ratio()
        if r > best_ratio * 0.98:
            r = difflib.SequenceMatcher(None, target, window).ratio()
            if r > best_ratio:
                best, best_ratio = start, r
    msg = "The SEARCH section must exactly match existing lines in the file, but it did not."
    if best_ratio > 0.5:
        lo, hi = max(best - 2, 0), min(best + n + 2, len(c_lines))
        similar = "\n".join(c_lines[lo:hi])
        msg += f"\nDid you mean to match these actual lines (lines {lo + 1}-{hi})?\n```\n{similar}\n```"
    return msg


def unified_diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=2,
    ))
