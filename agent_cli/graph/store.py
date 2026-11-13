"""Storage for per-file parse results ("facts"), behind a small interface.

Only per-file facts are stored; the resolved graph is rebuilt in memory on
every load (resolution is fast and depends on all files at once, so caching
it would mean invalidating it on any change anyway).

To move to SQLite later, implement FactStore with the same four members
(e.g. a table ``files(path TEXT PRIMARY KEY, mtime REAL, size INT, facts TEXT)``)
and pass it to graph.build.index().
"""
import json
from pathlib import Path
from typing import Dict

# Bump when the facts format or a parser changes, to force a full re-parse.
FACTS_VERSION = 2


class FactStore:
    """Persistent map: repo-relative path -> {"key": [mtime, size], "facts": {...}}."""

    location = ""  # human-readable, for messages

    def load(self) -> Dict[str, dict]:
        """All stored entries ({} if nothing is stored or it is stale)."""
        raise NotImplementedError

    def save(self, entries: Dict[str, dict]) -> None:
        """Replace the stored entries with `entries`."""
        raise NotImplementedError

    def size_bytes(self) -> int:
        """Size of the backing storage, for reporting."""
        raise NotImplementedError


class JsonFactStore(FactStore):
    """One JSON file, e.g. .agent-cache/codegraph.json (git-ignored)."""

    def __init__(self, path: Path):
        self.path = path
        self.location = str(path)

    def load(self) -> Dict[str, dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != FACTS_VERSION:
            return {}
        files = data.get("files")
        return files if isinstance(files, dict) else {}

    def save(self, entries: Dict[str, dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ignore = self.path.parent / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n", encoding="utf-8")
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": FACTS_VERSION, "files": entries},
                                      separators=(",", ":")), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


class MemoryFactStore(FactStore):
    """In-memory store (tests, one-off use)."""

    location = "<memory>"

    def __init__(self):
        self.entries = {}  # type: Dict[str, dict]

    def load(self) -> Dict[str, dict]:
        return dict(self.entries)

    def save(self, entries: Dict[str, dict]) -> None:
        self.entries = dict(entries)

    def size_bytes(self) -> int:
        return len(json.dumps(self.entries))


def default_store(root: Path, cache_dir: Path = None) -> FactStore:
    return JsonFactStore((cache_dir or root / ".agent-cache") / "codegraph.json")
