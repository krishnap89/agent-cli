"""Queries over a CodeGraph, rendered as compact text within a character
budget (they are meant to be pasted into a small model's context).

Every line points at source as path:line. Edge tags read
  [extracted 1.00]            resolved from scope/imports
  [inferred 0.81]             resolved via a receiver type or a guess
  [ambiguous 0.33, 3 candidates]  first candidate shown, others listed
"""
import re
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..repomap import _split_words
from .model import (CALLS, CLASS_KINDS, CONTAINS, DEPENDS_ON, IMPORTS, INHERITS, INJECTS,
                    OVERRIDES, REFERENCES, USES_TYPE, CodeGraph, Edge)

DEFAULT_BUDGET = 4000
PATH_KINDS = (CALLS, INJECTS, INHERITS, OVERRIDES, USES_TYPE, REFERENCES)
# Edge kinds along which a change propagates to the edge's source.
IMPACT_KINDS = (CALLS, INHERITS, INJECTS, USES_TYPE, OVERRIDES, REFERENCES, IMPORTS, DEPENDS_ON)
_TEST_RE = re.compile(r"(^|/)(test|androidTest|testDebug|testRelease)/|Test\.kt$|Tests\.kt$")


class _Out:
    """Line accumulator that stops at a character budget."""

    def __init__(self, budget: int):
        self.budget = budget
        self.lines = []  # type: List[str]
        self.used = 0
        self.dropped = 0

    def add(self, line: str) -> bool:
        if self.dropped or self.used + len(line) + 1 > self.budget:
            self.dropped += 1
            return False
        self.lines.append(line)
        self.used += len(line) + 1
        return True

    def text(self) -> str:
        out = list(self.lines)
        if self.dropped:
            out.append(f"... {self.dropped} more lines (raise --budget)")
        return "\n".join(out)


def _resolve_symbol(g: CodeGraph, symbol: str) -> Tuple[List[str], Optional[str]]:
    """Node ids for a symbol, or an error message."""
    ids = g.find(symbol)
    if not ids:
        return [], f"No symbol matching '{symbol}'. Try `agent graph query \"{symbol}\"`."
    return ids, None


def _label(g: CodeGraph, nid: str) -> str:
    n = g.nodes.get(nid)
    if n is None:
        return nid
    if n.kind == "file":
        return n.file
    return f"{g.short(nid)} ({n.kind}) {n.loc}"


def _many(g: CodeGraph, ids: List[str], out: _Out) -> None:
    out.add(f"{len(ids)} symbols match; showing all. Use a longer name (Class.member) to narrow:")


def _edge_line(g: CodeGraph, e: Edge, other: str, indent: str = "  ") -> str:
    tag = e.tag()
    s = f"{indent}{g.short(other)}  {e.loc}  [{tag}]"
    if e.cands and len(e.cands) > 1:
        s += "  or: " + ", ".join(g.short(c) for c in e.cands[1:4])
        if len(e.cands) > 4:
            s += ", ..."
    return s


def _group_targets(g: CodeGraph, nid: str) -> List[str]:
    """For a class, calls to it are constructor calls; also fold in its
    <init>. For other nodes, just the node."""
    n = g.nodes[nid]
    if n.kind in CLASS_KINDS:
        return [nid] + [c for c in g.children(nid) if g.nodes[c].kind == "constructor"]
    return [nid]


def callers(g: CodeGraph, symbol: str, budget: int = DEFAULT_BUDGET) -> str:
    ids, err = _resolve_symbol(g, symbol)
    if err:
        return err
    out = _Out(budget)
    if len(ids) > 1:
        _many(g, ids, out)
    for nid in ids:
        edges = [e for t in _group_targets(g, nid) for e in g.in_edges(t, (CALLS,))]
        # Also show callers that reach this member through an override.
        via = [e for o in g.in_edges(nid, (OVERRIDES,)) for e in g.in_edges(o.src, (CALLS,))]
        out.add(f"callers of {_label(g, nid)}: {len(edges)}" + (f" (+{len(via)} via overrides)" if via else ""))
        for e in sorted(edges, key=lambda e: (-e.conf, e.file, e.line)):
            out.add(_edge_line(g, e, e.src))
        for e in sorted(via, key=lambda e: (-e.conf, e.file, e.line)):
            out.add(_edge_line(g, e, e.src) + f"  (calls {g.short(e.dst)})")
        if not edges and not via:
            out.add("  (none found)")
    return out.text()


def callees(g: CodeGraph, symbol: str, budget: int = DEFAULT_BUDGET) -> str:
    ids, err = _resolve_symbol(g, symbol)
    if err:
        return err
    out = _Out(budget)
    if len(ids) > 1:
        _many(g, ids, out)
    for nid in ids:
        n = g.nodes[nid]
        sources = [nid]
        if n.kind in CLASS_KINDS:  # a class's callees are those of its members
            sources += g.descendants(nid)
        edges = [e for s in sources for e in g.out_edges(s, (CALLS,))]
        out.add(f"callees of {_label(g, nid)}: {len(edges)}")
        for e in sorted(edges, key=lambda e: (e.file, e.line)):
            prefix = f"  {g.short(e.src)} -> " if e.src != nid else "  "
            out.add(_edge_line(g, e, e.dst, indent=prefix))
        if not edges:
            out.add("  (none found)")
    return out.text()


def find_path(g: CodeGraph, src: str, dst: str, max_depth: int = 6,
              kinds: Sequence[str] = PATH_KINDS) -> Optional[List[Edge]]:
    """Shortest directed path from src (or any of its members) to dst (or
    any of its members), BFS over `kinds`. Contains edges are free hops
    down into members."""
    goal = {dst} | set(g.descendants(dst))
    start = [src] + g.descendants(src)
    prev = {}  # type: Dict[str, Optional[Edge]]
    depth = {}  # type: Dict[str, int]
    q = deque()  # type: deque
    for s in start:
        prev[s] = None
        depth[s] = 0
        q.append(s)
    ks = set(kinds)
    while q:
        cur = q.popleft()
        if cur in goal and cur not in start:
            path = []
            while prev[cur] is not None:
                e = prev[cur]
                path.append(e)
                cur = e.src
            return list(reversed(path))
        if depth[cur] >= max_depth:
            continue
        for e in g.out.get(cur, ()):
            if e.kind not in ks or e.dst in prev:
                continue
            prev[e.dst] = e
            depth[e.dst] = depth[cur] + 1
            q.append(e.dst)
    if src in goal or dst in start:
        return []
    return None


def path(g: CodeGraph, a: str, b: str, max_depth: int = 6,
         kinds: Sequence[str] = PATH_KINDS, budget: int = DEFAULT_BUDGET) -> str:
    aids, err = _resolve_symbol(g, a)
    if err:
        return err
    bids, err = _resolve_symbol(g, b)
    if err:
        return err
    out = _Out(budget)
    best = None  # type: Optional[Tuple[str, str, List[Edge]]]
    for x in aids[:5]:
        for y in bids[:5]:
            p = find_path(g, x, y, max_depth, kinds)
            if p is not None and (best is None or len(p) < len(best[2])):
                best = (x, y, p)
    if best is None:
        # try the other direction before giving up
        for y in bids[:5]:
            for x in aids[:5]:
                p = find_path(g, y, x, max_depth, kinds)
                if p:
                    out.add(f"No path {a} -> {b} within {max_depth} hops, but there is one "
                            f"the other way ({len(p)} hops):")
                    _render_path(g, y, p, out)
                    return out.text()
        return f"No path from {a} to {b} within {max_depth} hops over {', '.join(kinds)}."
    x, y, p = best
    if not p:
        out.add(f"{g.short(x)} and {g.short(y)} are the same symbol or one contains the other.")
        return out.text()
    conf = 1.0
    for e in p:
        conf *= e.conf
    out.add(f"path {g.short(x)} -> {g.short(y)}: {len(p)} hops, confidence {conf:.2f}")
    _render_path(g, x, p, out)
    return out.text()


def _render_path(g: CodeGraph, start: str, p: List[Edge], out: _Out) -> None:
    out.add(f"  {_label(g, p[0].src if p else start)}")
    for e in p:
        out.add(f"    -{e.kind}-> {_label(g, e.dst)}   at {e.loc} [{e.tag()}]")


def explain(g: CodeGraph, symbol: str, budget: int = DEFAULT_BUDGET) -> str:
    ids, err = _resolve_symbol(g, symbol)
    if err:
        return err
    out = _Out(budget)
    if len(ids) > 1:
        _many(g, ids, out)
    for nid in ids[:6]:
        _explain_one(g, nid, out)
        out.add("")
    return out.text().rstrip()


def _explain_one(g: CodeGraph, nid: str, out: _Out) -> None:
    n = g.nodes[nid]
    span = f"{n.file}:{n.line}-{n.end}" if n.end and n.end != n.line else n.loc
    out.add(f"{g.short(nid)}  [{n.kind}]  {span}" + (f"  module {n.module}" if n.module else ""))
    if n.sig:
        out.add(f"  {n.sig}")
    if n.parent:
        out.add(f"  in {g.short(n.parent)}")

    def section(title: str, edges: List[Edge], other_is_src: bool, limit: int = 8) -> None:
        if not edges:
            return
        seen = []  # type: List[Edge]
        keys = set()  # type: Set[str]
        for e in sorted(edges, key=lambda e: (-e.conf, e.file, e.line)):
            k = e.src if other_is_src else e.dst
            if k not in keys:
                keys.add(k)
                seen.append(e)
        out.add(f"  {title} ({len(keys)}):")
        for e in seen[:limit]:
            out.add(_edge_line(g, e, e.src if other_is_src else e.dst, indent="    "))
        if len(seen) > limit:
            out.add(f"    ... {len(seen) - limit} more")

    targets = _group_targets(g, nid)
    if n.kind in CLASS_KINDS or n.kind == "entry":
        section("supertypes", g.out_edges(nid, (INHERITS,)), False)
        section("subtypes", g.in_edges(nid, (INHERITS,)), True)
        section("injects", g.out_edges(nid, (INJECTS,)), False)
        section("injected into", g.in_edges(nid, (INJECTS,)), True)
        members = [c for c in g.children(nid) if g.nodes[c].kind != "constructor"]
        if members:
            names = ", ".join(g.nodes[c].name for c in members[:20])
            out.add(f"  members ({len(members)}): {names}{', ...' if len(members) > 20 else ''}")
    if n.kind in ("function", "property"):
        section("overrides", g.out_edges(nid, (OVERRIDES,)), False)
        section("overridden by", g.in_edges(nid, (OVERRIDES,)), True)
    if n.kind == "module":
        section("depends on", g.out_edges(nid, (DEPENDS_ON,)), False)
        section("used by modules", g.in_edges(nid, (DEPENDS_ON,)), True)
    section("called by", [e for t in targets for e in g.in_edges(t, (CALLS,))], True)
    srcs = [nid] + (g.descendants(nid) if n.kind in CLASS_KINDS else [])
    section("calls", [e for s in srcs for e in g.out_edges(s, (CALLS,))], False)
    section("references", g.out_edges(nid, (REFERENCES,)), False)
    section("referenced by", g.in_edges(nid, (REFERENCES,)), True)
    section("uses types", g.out_edges(nid, (USES_TYPE,)), False, limit=6)
    section("used as a type by", g.in_edges(nid, (USES_TYPE,)), True, limit=6)


def _is_test(path: str) -> bool:
    return bool(_TEST_RE.search(path))


def impact(g: CodeGraph, files: Iterable[str], depth: int = 3,
           budget: int = DEFAULT_BUDGET) -> str:
    """What may break if these files change: reverse traversal from every
    declaration in them along call/inherit/inject/type/override/reference/
    import/module edges, grouped by file with the nearest reason."""
    seeds = []  # type: List[str]
    missing = []  # type: List[str]
    norm = []  # type: List[str]
    for f in files:
        f = f.strip().lstrip("./")
        fid = "file:" + f
        if fid not in g.nodes:
            hits = [n for n in g.nodes if n.startswith("file:") and n.endswith("/" + f)]
            if len(hits) == 1:
                fid, f = hits[0], hits[0][5:]
            else:
                missing.append(f)
                continue
        norm.append(f)
        seeds.append(fid)
        seeds.extend(g.by_file.get(f, ()))
        for mod_edge in g.in_edges(fid, (CONTAINS,)):
            if g.nodes[mod_edge.src].kind == "module" and g.nodes[mod_edge.src].file == f:
                seeds.append(mod_edge.src)
        for nid, node in g.nodes.items():  # build file changed -> its module
            if node.kind == "module" and node.file == f:
                seeds.append(nid)
    out = _Out(budget)
    if missing:
        out.add("not in graph: " + ", ".join(missing))
    if not seeds:
        return out.text() or "Nothing to analyse."
    dist = {s: 0 for s in seeds}  # type: Dict[str, int]
    why = {}  # type: Dict[str, Edge]
    q = deque(seeds)
    ks = set(IMPACT_KINDS)
    while q:
        cur = q.popleft()
        if dist[cur] >= depth:
            continue
        for e in g.inc.get(cur, ()):
            if e.kind not in ks or e.src in dist:
                continue
            dist[e.src] = dist[cur] + 1
            why[e.src] = e
            q.append(e.src)
    changed = set(norm)
    by_file = {}  # type: Dict[str, Tuple[int, str]]
    modules = []  # type: List[Tuple[int, str]]
    for nid, d in dist.items():
        if d == 0:
            continue
        node = g.nodes[nid]
        if node.kind == "module":
            modules.append((d, nid))
            continue
        f = node.file
        if not f or f in changed:
            continue
        if f not in by_file or d < by_file[f][0]:
            by_file[f] = (d, nid)
    out.add(f"impact of {', '.join(norm)}: {len(by_file)} files"
            + (f", {len(modules)} modules" if modules else "") + f" (depth {depth})")
    tests = [f for f in by_file if _is_test(f)]
    for f in sorted(by_file, key=lambda f: (by_file[f][0], _is_test(f), f)):
        d, nid = by_file[f]
        e = why[nid]
        at = f":{e.line}" if e.file == f else e.loc
        src = "" if g.nodes[e.src].kind == "file" else g.short(e.src) + " "
        reason = f"{src}{e.kind} {g.short(e.dst)} at {at} [{e.prov}]"
        out.add(f"  d{d} {f}{'  (test)' if _is_test(f) else ''}  {reason}")
    for d, nid in sorted(modules):
        e = why[nid]
        out.add(f"  d{d} module {g.nodes[nid].name}  -{e.kind}-> {g.short(e.dst)}")
    if tests:
        out.add(f"tests to run ({len(tests)}): " + ", ".join(sorted(tests)))
    return out.text()


def query(g: CodeGraph, text: str, budget: int = DEFAULT_BUDGET, limit: int = 15,
          ranks: Optional[Dict[str, float]] = None) -> str:
    """Free-text search over symbol names, ids and paths, ranked by match
    strength and PageRank; each hit shows its signature and connections."""
    words = _split_words(text)
    if not words:
        return "Empty query."
    if ranks is None:
        from .rank import node_ranks
        ranks = node_ranks(g)
    n = max(len(ranks), 1)
    scored = []  # type: List[Tuple[float, str]]
    q_low = text.strip().lower()
    for nid, node in g.nodes.items():
        if node.kind in ("file", "constructor"):
            continue
        name = node.name.lower()
        hay = (nid + " " + node.file).lower()
        score = 0.0
        if name == q_low:
            score += 30
        for w in words:
            if w == name:
                score += 12
            elif w in name:
                score += 6
            elif w in hay:
                score += 2
        if score <= 0:
            continue
        score *= 1.0 + min(ranks.get(nid, 0.0) * n, 20.0) / 4.0
        if node.kind in CLASS_KINDS:
            score *= 1.2
        if _is_test(node.file):
            score *= 0.3
        scored.append((score, nid))
    if not scored:
        return f"No symbols match '{text}'."
    scored.sort(key=lambda s: (-s[0], s[1]))
    out = _Out(budget)
    out.add(f"{len(scored)} matches for '{text}' (top {min(limit, len(scored))}):")
    for _score, nid in scored[:limit]:
        node = g.nodes[nid]
        ncallers = len({e.src for t in _group_targets(g, nid) for e in g.in_edges(t, (CALLS,))})
        ncallees = len(g.out_edges(nid, (CALLS,)))
        extra = []
        if ncallers:
            extra.append(f"{ncallers} callers")
        if ncallees:
            extra.append(f"{ncallees} calls")
        subs = len(g.in_edges(nid, (INHERITS,)))
        if subs:
            extra.append(f"{subs} subtypes")
        refs = len(g.in_edges(nid, (REFERENCES,)))
        if refs:
            extra.append(f"{refs} refs")
        out.add(f"  {g.short(nid)} [{node.kind}] {node.loc}" + (f"  ({', '.join(extra)})" if extra else ""))
        if node.sig and node.sig != node.name:
            out.add(f"      {node.sig}")
    return out.text()


def build_summary(g: CodeGraph) -> str:
    """Stats printed by `agent graph build`."""
    info = g.stats.get("build")
    lines = []
    if info is not None:
        size = info.cache_bytes
        size_s = f"{size / 1e6:.2f} MB" if size >= 1e5 else f"{size / 1e3:.1f} KB"
        lines.append(f"indexed {info.files} files ({info.parsed} parsed, "
                     f"{info.files - min(info.parsed, info.files)} from cache"
                     + (f", {info.removed} removed" if info.removed else "") + ")")
        lines.append(f"parse {info.parse_s:.2f}s · load {info.load_s:.2f}s · "
                     f"cache {size_s} ({info.cache_location})")
    kinds = {}  # type: Dict[str, int]
    for node in g.nodes.values():
        kinds[node.kind] = kinds.get(node.kind, 0) + 1
    lines.append(f"nodes {len(g.nodes)}: " + ", ".join(
        f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    ek = {}  # type: Dict[str, int]
    prov = {}  # type: Dict[str, int]
    for e in g.edges:
        ek[e.kind] = ek.get(e.kind, 0) + 1
        if e.kind != CONTAINS:
            prov[e.prov] = prov.get(e.prov, 0) + 1
    lines.append(f"edges {len(g.edges)}: " + ", ".join(
        f"{k} {v}" for k, v in sorted(ek.items(), key=lambda kv: -kv[1])))
    if prov:
        lines.append("provenance (non-contains): " + ", ".join(
            f"{k} {v}" for k, v in sorted(prov.items(), key=lambda kv: -kv[1])))
    c = g.stats.get("calls", {})
    total = c.get("total", 0) - c.get("invoke", 0)
    if total > 0:
        resolved = c.get("extracted", 0) + c.get("inferred", 0)
        amb = c.get("ambiguous", 0)
        lines.append(
            f"call resolution: {resolved}/{total} resolved ({100.0 * resolved / total:.1f}%) — "
            f"extracted {c.get('extracted', 0)}, inferred {c.get('inferred', 0)}, "
            f"ambiguous {amb}, unresolved {c.get('unresolved', 0)}"
            + (f" (of which {c.get('external', 0)} on known external types)" if c.get("external") else ""))
        internal = c.get("internal_resolved", 0) + c.get("internal_unresolved", 0)
        if internal:
            ir = c.get("internal_resolved", 0)
            lines.append(f"  calls to names defined in this project (excluding known external "
                         f"receivers): {ir}/{internal} ({100.0 * ir / internal:.1f}%) resolved "
                         f"or ambiguous")
        if c.get("invoke"):
            lines.append(f"  ({c['invoke']} calls of function-typed values not counted)")
        if c.get("common_skipped"):
            lines.append(f"  ({c['common_skipped']} calls like .map/.filter on an unknown receiver "
                         f"left unresolved rather than guessed)")
        top = g.stats.get("top_unresolved") or []
        if top:
            lines.append("  most-missed project names: " + ", ".join(f"{n}×{k}" for n, k in top[:8]))
    return "\n".join(lines)
