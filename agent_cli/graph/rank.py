"""PageRank over the code graph, at node level (for `graph query`) and at
file level (for ranking the repo map)."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .model import CALLS, INHERITS, INJECTS, CodeGraph

RANK_KINDS = (CALLS, INHERITS, INJECTS)


def pagerank(nodes: Iterable[str], edges: Dict[str, Dict[str, float]],
             damping: float = 0.85, iterations: int = 30) -> Dict[str, float]:
    """Weighted PageRank. edges[src][dst] = weight. Dangling mass is spread
    evenly so ranks sum to 1."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    out_w = {s: sum(t.values()) for s, t in edges.items()}
    rev = defaultdict(list)  # type: Dict[str, List[Tuple[str, float]]]
    for s, targets in edges.items():
        for t, w in targets.items():
            rev[t].append((s, w))
    ranks = {x: 1.0 / n for x in nodes}
    for _ in range(iterations):
        dangling = sum(ranks[x] for x in nodes if not out_w.get(x))
        base = (1.0 - damping) / n + damping * dangling / n
        ranks = {x: base + damping * sum(ranks[s] * w / out_w[s] for s, w in rev.get(x, ())
                                         if s in ranks)
                 for x in nodes}
    return ranks


def node_edges(graph: CodeGraph, kinds=RANK_KINDS) -> Dict[str, Dict[str, float]]:
    """src -> dst -> summed confidence, over the given edge kinds."""
    ks = set(kinds)
    edges = defaultdict(lambda: defaultdict(float))  # type: Dict[str, Dict[str, float]]
    for e in graph.edges:
        if e.kind in ks and e.src != e.dst:
            edges[e.src][e.dst] += e.conf
    return edges


def node_ranks(graph: CodeGraph) -> Dict[str, float]:
    ids = [i for i, n in graph.nodes.items() if n.kind not in ("file", "module")]
    return pagerank(ids, node_edges(graph))


def file_edges(graph: CodeGraph, kinds=RANK_KINDS) -> Dict[str, Dict[str, float]]:
    """Code-graph edges aggregated to file -> file."""
    ks = set(kinds)
    edges = defaultdict(lambda: defaultdict(float))  # type: Dict[str, Dict[str, float]]
    nodes = graph.nodes
    for e in graph.edges:
        if e.kind not in ks:
            continue
        a, b = nodes.get(e.src), nodes.get(e.dst)
        if a is None or b is None or not a.file or not b.file or a.file == b.file:
            continue
        edges[a.file][b.file] += e.conf
    return edges


def graph_file_ranks(root: Path, files: List[str],
                     fallback_edges: Dict[str, Dict[str, float]],
                     cache_dir: Optional[Path] = None) -> Optional[Dict[str, float]]:
    """File ranks for the repo map. Files with outgoing call/inherit/inject
    edges in the code graph use those; every other file keeps its
    identifier-based edges (the old ranking). None when there is no Kotlin
    graph to use."""
    from .build import load_graph
    graph, _info = load_graph(root, cache_dir=cache_dir)
    gedges = file_edges(graph)
    if not gedges:
        return None
    fileset = set(files)
    merged = {}  # type: Dict[str, Dict[str, float]]
    for f in files:
        src = gedges.get(f)
        if src:
            merged[f] = {t: w for t, w in src.items() if t in fileset}
        elif f in fallback_edges:
            merged[f] = {t: w for t, w in fallback_edges[f].items() if t in fileset}
    return pagerank(files, merged, iterations=20)
