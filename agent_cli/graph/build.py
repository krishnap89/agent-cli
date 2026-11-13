"""Index a repo into per-file facts (incremental) and load the resolved graph."""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .model import CodeGraph
from .resolver import resolve
from .sources import extract, list_sources
from .store import FactStore, default_store


class BuildInfo:
    """What happened during a load, for `agent graph build`."""

    def __init__(self):
        self.files = 0
        self.parsed = 0
        self.removed = 0
        self.parse_s = 0.0     # re-parsing changed files
        self.load_s = 0.0      # reading the cache + resolving the graph in memory
        self.cache_bytes = 0
        self.cache_location = ""


def index(root: Path, store: FactStore, force: bool = False) -> Tuple[Dict[str, dict], BuildInfo]:
    """Return {rel: facts} for every graph input, re-parsing only files whose
    (mtime, size) changed since the stored entry. Saves the store if anything
    changed."""
    info = BuildInfo()
    t0 = time.perf_counter()
    stored = {} if force else store.load()
    t_cache = time.perf_counter() - t0
    entries = {}  # type: Dict[str, dict]
    t1 = time.perf_counter()
    for rel in list_sources(root):
        try:
            st = (root / rel).stat()
        except OSError:
            continue
        key = [st.st_mtime, st.st_size]
        old = stored.get(rel)
        if old is not None and old.get("key") == key:
            entries[rel] = old
            continue
        facts = extract(root, rel)
        info.parsed += 1
        if facts is not None:
            entries[rel] = {"key": key, "facts": facts}
    info.parse_s = time.perf_counter() - t1
    info.removed = len(set(stored) - set(entries))
    if info.parsed or info.removed or force:
        store.save(entries)
    info.files = len(entries)
    info.load_s = t_cache
    info.cache_bytes = store.size_bytes()
    info.cache_location = store.location
    return {rel: e["facts"] for rel, e in entries.items()}, info


def load_graph(root: Path, store: Optional[FactStore] = None, force: bool = False,
               cache_dir: Optional[Path] = None) -> Tuple[CodeGraph, BuildInfo]:
    root = root.resolve()
    store = store or default_store(root, cache_dir)
    facts, info = index(root, store, force)
    t0 = time.perf_counter()
    graph = resolve(facts)
    info.load_s += time.perf_counter() - t0
    graph.stats["build"] = info
    return graph, info
