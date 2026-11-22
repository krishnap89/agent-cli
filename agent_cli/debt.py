"""/debt: list deliberate shortcuts marked in comments, by file.

    # shortcut: in-memory only, move to Redis when we run more than one worker
    // shortcut: no retry; add backoff once the API rate-limits us

The text after `shortcut:` is `<limit>, <when to upgrade>` (a `;` also
separates them). The coder style rules ask the model to leave these markers.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_MARK_RE = re.compile(r"(?:#|//|/\*|\*|--|<!--|;)\s*shortcut\s*:\s*(.+)$", re.I)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`')
_TRIPLE_RE = re.compile(r'"""|\'\'\'')
_MAX_BYTES = 1_000_000


class Shortcut:
    __slots__ = ("file", "line", "limit", "upgrade")

    def __init__(self, file: str, line: int, limit: str, upgrade: str):
        self.file, self.line, self.limit, self.upgrade = file, line, limit, upgrade


def parse_marker(text: str) -> Optional[Tuple[str, str]]:
    """(limit, when to upgrade) from one source line, or None. The marker must
    be a comment: one that opens inside a string literal on the line is skipped."""
    m = _MARK_RE.search(text)
    if not m:
        return None
    prefix = _STRING_RE.sub("", _TRIPLE_RE.sub("", text[:m.start()]))
    if any(q in prefix for q in "\"'`"):
        return None  # the comment token sits inside a string
    body = re.split(r"\*/|-->", m.group(1), maxsplit=1)[0].strip()
    sep = ";" if ";" in body else ","
    limit, _, upgrade = body.partition(sep)
    return limit.strip(), upgrade.strip()


def scan_text(text: str) -> List[Tuple[int, str, str]]:
    """(line, limit, upgrade) for each marker, skipping triple-quoted strings."""
    out, in_triple = [], False
    for i, line in enumerate(text.splitlines(), 1):
        triples = len(_TRIPLE_RE.findall(line))
        if not in_triple:
            hit = parse_marker(line)
            if hit:
                out.append((i, hit[0], hit[1]))
        if triples % 2:
            in_triple = not in_triple
    return out


def find_shortcuts(root: Path, files: Optional[List[str]] = None) -> List[Shortcut]:
    """Markers in the repo's source files (the repo map's file list, so the
    same skipped dirs and blocked files). Nothing is sent to a model."""
    if files is None:
        from .repomap import RepoMap
        files = RepoMap(root).list_files()
    found = []
    for rel in files:
        p = root / rel
        try:
            if not p.is_file() or p.stat().st_size > _MAX_BYTES:
                continue
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192] or b"shortcut" not in raw.lower():
            continue
        for line, limit, upgrade in scan_text(raw.decode("utf-8", errors="replace")):
            found.append(Shortcut(rel, line, limit, upgrade))
    return found


def format_debt(items: List[Shortcut]) -> str:
    if not items:
        return ("No shortcuts marked. Mark one with a comment like:\n"
                "  # shortcut: <limit>, <when to upgrade>")
    by_file: Dict[str, List[Shortcut]] = {}
    for s in items:
        by_file.setdefault(s.file, []).append(s)
    n = len(items)
    lines = [f"{n} shortcut{'s' if n != 1 else ''} in {len(by_file)} file{'s' if len(by_file) != 1 else ''}:"]
    for f in sorted(by_file):
        lines.append(f)
        for s in by_file[f]:
            when = s.upgrade or "(no upgrade trigger given)"
            lines.append(f"  {s.line:>5}  {s.limit}  ->  upgrade: {when}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    """`agent debt [dir]`."""
    root = Path(argv[0] if argv else ".").resolve()
    print(format_debt(find_shortcuts(root)))
    return 0
