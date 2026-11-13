"""In-memory code graph: nodes, typed edges, adjacency and symbol lookup."""
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# Edge kinds
CONTAINS, IMPORTS, CALLS, INHERITS, INJECTS = "contains", "imports", "calls", "inherits", "injects"
USES_TYPE, OVERRIDES, DEPENDS_ON, REFERENCES = "uses_type", "overrides", "depends_on", "references"
EDGE_KINDS = (CONTAINS, IMPORTS, CALLS, INHERITS, INJECTS, USES_TYPE, OVERRIDES, DEPENDS_ON,
              REFERENCES)

# Provenance
EXTRACTED, INFERRED, AMBIGUOUS = "extracted", "inferred", "ambiguous"

CLASS_KINDS = frozenset({"class", "interface", "object", "enum", "annotation", "typealias"})
CODE_KINDS = frozenset({"function", "constructor", "property"})


class Node:
    __slots__ = ("id", "kind", "name", "file", "line", "end", "parent", "sig", "module", "extra")

    def __init__(self, id: str, kind: str, name: str, file: str = "", line: int = 0,
                 end: int = 0, parent: Optional[str] = None, sig: str = "", module: str = ""):
        self.id = id
        self.kind = kind          # file module class interface object enum annotation
        self.name = name          # typealias function constructor property entry resource
        self.file = file
        self.line = line
        self.end = end or line
        self.parent = parent
        self.sig = sig
        self.module = module
        self.extra = {}  # type: dict

    @property
    def loc(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file

    def __repr__(self) -> str:
        return f"Node({self.id!r}, {self.kind})"


class Edge:
    __slots__ = ("src", "dst", "kind", "prov", "conf", "file", "line", "cands")

    def __init__(self, src: str, dst: str, kind: str, prov: str = EXTRACTED, conf: float = 1.0,
                 file: str = "", line: int = 0, cands: Optional[List[str]] = None):
        self.src = src
        self.dst = dst
        self.kind = kind
        self.prov = prov
        self.conf = conf
        self.file = file   # where the edge comes from in source (call site)
        self.line = line
        self.cands = cands  # all candidates when prov == ambiguous (dst is the first)

    @property
    def loc(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file

    def tag(self) -> str:
        t = f"{self.prov} {self.conf:.2f}"
        if self.cands and len(self.cands) > 1:
            t += f", {len(self.cands)} candidates"
        return t

    def __repr__(self) -> str:
        return f"Edge({self.src} -{self.kind}-> {self.dst}, {self.prov})"


class CodeGraph:
    def __init__(self):
        self.nodes = {}  # type: Dict[str, Node]
        self.edges = []  # type: List[Edge]
        self.out = defaultdict(list)  # type: Dict[str, List[Edge]]
        self.inc = defaultdict(list)  # type: Dict[str, List[Edge]]
        self.by_name = defaultdict(list)  # type: Dict[str, List[str]]
        self.by_file = defaultdict(list)  # type: Dict[str, List[str]]
        self.stats = {}  # type: dict

    # -- construction --

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.by_name[node.name].append(node.id)
        if node.file and node.kind not in ("file", "module"):
            self.by_file[node.file].append(node.id)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        self.edges.append(edge)
        self.out[edge.src].append(edge)
        self.inc[edge.dst].append(edge)
        return edge

    # -- traversal helpers --

    def children(self, nid: str) -> List[str]:
        return [e.dst for e in self.out.get(nid, ()) if e.kind == CONTAINS]

    def descendants(self, nid: str) -> List[str]:
        out, stack = [], [nid]
        while stack:
            for c in self.children(stack.pop()):
                out.append(c)
                stack.append(c)
        return out

    def out_edges(self, nid: str, kinds: Optional[Iterable[str]] = None) -> List[Edge]:
        es = self.out.get(nid, ())
        if kinds is None:
            return list(es)
        ks = set(kinds)
        return [e for e in es if e.kind in ks]

    def in_edges(self, nid: str, kinds: Optional[Iterable[str]] = None) -> List[Edge]:
        es = self.inc.get(nid, ())
        if kinds is None:
            return list(es)
        ks = set(kinds)
        return [e for e in es if e.kind in ks]

    def short(self, nid: str) -> str:
        """Display name: the id without its package (and file:/res: prefixes kept)."""
        n = self.nodes.get(nid)
        if n is None:
            return nid
        if n.kind == "file":
            return n.file
        pkg = n.extra.get("pkg", "")
        if pkg and nid.startswith(pkg + "."):
            return nid[len(pkg) + 1:]
        return nid

    # -- lookup --

    def find(self, query: str) -> List[str]:
        """Nodes matching a user-supplied symbol: a full id, a suffix like
        'UserRepository.load', a simple name, or a file path."""
        q = query.strip()
        if not q:
            return []
        if q in self.nodes:
            return [q]
        if "file:" + q in self.nodes:
            return ["file:" + q]
        # Class.member or pkg-suffix match
        if "." in q or "/" in q:
            hits = [nid for nid in self.nodes
                    if nid.endswith("." + q) or nid.split("#", 1)[0].endswith("." + q)]
            if hits:
                return sorted(hits)
            fhits = [nid for nid in self.nodes if nid.startswith("file:") and nid.endswith("/" + q)]
            if fhits:
                return sorted(fhits)
        hits = [nid for nid in self.by_name.get(q, ()) if self.nodes[nid].kind != "file"]
        if hits:
            return sorted(hits, key=lambda i: (self.nodes[i].kind not in CLASS_KINDS, i))
        low = q.lower()
        return sorted(nid for name, ids in self.by_name.items() if name.lower() == low
                      for nid in ids)
