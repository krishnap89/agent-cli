"""`agent graph audit`: what could be deleted, merged or needs a test.

Three kinds of finding, ranked together on one score ("lines at stake"):

- dead     functions nothing calls, references or overrides; score = its lines
- dup      near-duplicate function bodies (token similarity >= 0.85);
           score = smaller body's lines x similarity x 1.5
- untested non-test files no test reaches through the graph, weighted by
           fan-in (other files depending on them) and branching;
           score = 3 x fan-in + branches / 2 + lines / 100

Only the graph is used to find them; with --explain (the default) the model is
shown the top 5 with their code and asked to judge each, nothing more.
"""
import difflib
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model import (CALLS, CLASS_KINDS, IMPORTS, INHERITS, INJECTS, OVERRIDES, REFERENCES,
                    USES_TYPE, CodeGraph)
from .queries import _Out, is_test

DEAD, DUP, UNTESTED = "dead", "dup", "untested"
_LIVE_IN = (CALLS, REFERENCES, OVERRIDES, INJECTS)
_DEP_KINDS = (CALLS, INJECTS, INHERITS, USES_TYPE, REFERENCES)
_REACH_KINDS = (CALLS, INJECTS, INHERITS, USES_TYPE, REFERENCES, IMPORTS)
# Called by frameworks or the runtime, never by project code.
_ENTRY_NAMES = frozenset("""main init deinit setUp tearDown setUpWithError tearDownWithError
viewDidLoad viewWillAppear viewDidAppear viewWillDisappear viewDidDisappear awakeFromNib
application scene onCreate onStart onResume onPause onStop onDestroy onCreateView
onViewCreated onAttach onDetach onCleared onBind onReceive constructor render
ngOnInit ngOnDestroy componentDidMount componentWillUnmount toString hashCode equals""".split())
_ENTRY_MODS = frozenset({"override", "abstract", "open", "external", "actual", "expect", "required",
                         "dynamic", "declare"})
_ENTRY_ANNOS = re.compile(r"Test|Preview|IBAction|objc|JvmStatic|Provides|Binds|Subscribe|"
                          r"BeforeEach|AfterEach|Before|After|HostListener|Get|Post|Put|Delete")
_TOKEN_RE = re.compile(r"\w+|[^\s\w]")
_COMMENT_RE = re.compile(r"^\s*(//|#|\*|/\*)")
_BRANCH_WORDS = re.compile(r"\b(if|elif|else if|when|switch|case|match|guard|catch|except)\b")
_CODE_EXTS = (".kt", ".kts", ".java", ".swift", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
DUP_MIN_LINES = 4
DUP_MIN_RATIO = 0.85


def _n(k: int, one: str, many: str = "") -> str:
    return f"{k} {one if k == 1 else (many or one + 's')}"


class Finding:
    __slots__ = ("kind", "score", "title", "loc", "detail", "nodes")

    def __init__(self, kind: str, score: float, title: str, loc: str, detail: str,
                 nodes: List[str]):
        self.kind, self.score, self.title = kind, score, title
        self.loc, self.detail, self.nodes = loc, detail, nodes

    def line(self) -> str:
        loc = f"  {self.loc}" if self.loc else ""
        return f"[{self.kind}] {self.title}{loc}  ({self.detail}; score {self.score:.1f})"


class _Sources:
    """File text cache (repo-relative paths)."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: Dict[str, List[str]] = {}

    def lines(self, rel: str) -> List[str]:
        if rel not in self._cache:
            try:
                text = (self.root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            self._cache[rel] = text.splitlines()
        return self._cache[rel]

    def body(self, g: CodeGraph, nid: str) -> List[str]:
        """Lines after the declaration line, without blanks and comments."""
        n = g.nodes[nid]
        src = self.lines(n.file)[n.line:n.end]
        return [ln.strip() for ln in src if ln.strip() and not _COMMENT_RE.match(ln)]


def _functions(g: CodeGraph) -> List[str]:
    return [nid for nid, n in g.nodes.items()
            if n.kind == "function" and n.file and not is_test(n.file)
            and not n.name.startswith("<")]


def _is_entry(g: CodeGraph, nid: str) -> bool:
    n = g.nodes[nid]
    decl = n.extra.get("decl", {})
    if n.name in _ENTRY_NAMES or n.name.startswith("test"):
        return True
    if _ENTRY_MODS & set(decl.get("mods", ())):
        return True
    if any(_ENTRY_ANNOS.search(str(a)) for a in decl.get("annos", ())):
        return True
    parent = g.nodes.get(n.parent or "")
    return parent is not None and parent.kind in ("interface", "protocol", "annotation")


def dead_code(g: CodeGraph) -> List[Finding]:
    out = []
    for nid in _functions(g):
        if g.in_edges(nid, _LIVE_IN) or g.out_edges(nid, (OVERRIDES,)) or _is_entry(g, nid):
            continue
        n = g.nodes[nid]
        size = max(1, n.end - n.line + 1)
        out.append(Finding(DEAD, float(size), g.short(nid), n.loc,
                           f"no callers, {_n(size, 'line')}", [nid]))
    return out


def near_duplicates(g: CodeGraph, src: _Sources, min_ratio: float = DUP_MIN_RATIO,
                    window: int = 40) -> List[Finding]:
    cands: List[Tuple[int, str, List[str]]] = []
    for nid in _functions(g):
        body = src.body(g, nid)
        if len(body) >= DUP_MIN_LINES:
            cands.append((len(body), nid, _TOKEN_RE.findall(" ".join(body))))
    cands.sort(key=lambda c: (len(c[2]), c[1]))
    out = []
    for i, (lines_a, a, ta) in enumerate(cands):
        for lines_b, b, tb in cands[i + 1:i + 1 + window]:
            if len(tb) > len(ta) * 1.3:
                break
            sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
            if sm.real_quick_ratio() < min_ratio or sm.quick_ratio() < min_ratio:
                continue
            ratio = sm.ratio()
            if ratio < min_ratio:
                continue
            na, nb = g.nodes[a], g.nodes[b]
            size = min(lines_a, lines_b)
            out.append(Finding(DUP, size * ratio * 1.5, f"{g.short(a)} ~ {g.short(b)}",
                               f"{na.loc} / {nb.loc}", f"{ratio:.0%} similar, {size} lines", [a, b]))
    return out


def _tested_files(g: CodeGraph, depth: int = 3) -> Tuple[set, int]:
    """Files reached from test code along dependency edges, and the test file count."""
    seeds = [nid for nid, n in g.nodes.items() if n.file and is_test(n.file)]
    test_files = {g.nodes[s].file for s in seeds}
    dist = {s: 0 for s in seeds}
    q = deque(seeds)
    while q:
        cur = q.popleft()
        if dist[cur] >= depth:
            continue
        for e in g.out_edges(cur, _REACH_KINDS):
            if e.dst not in dist:
                dist[e.dst] = dist[cur] + 1
                q.append(e.dst)
    return {g.nodes[n].file for n in dist if n in g.nodes and g.nodes[n].file}, len(test_files)


def untested_risky(g: CodeGraph, src: _Sources, tested: Optional[set] = None) -> List[Finding]:
    if tested is None:
        tested, _ = _tested_files(g)
    out = []
    for fid, fnode in g.nodes.items():
        f = fnode.file
        if (fnode.kind != "file" or not f or is_test(f) or f in tested or not g.by_file.get(f)
                or not f.endswith(_CODE_EXTS) or f.endswith(".d.ts")):
            continue
        dependents = {e.file or g.nodes[e.src].file
                      for nid in g.by_file[f] for e in g.in_edges(nid, _DEP_KINDS)}
        dependents.discard(f)
        dependents.discard("")
        text = src.lines(f)
        branches = sum(len(_BRANCH_WORDS.findall(ln)) for ln in text if not _COMMENT_RE.match(ln))
        if not dependents and branches < 5:
            continue
        score = 3 * len(dependents) + branches / 2 + len(text) / 100
        out.append(Finding(UNTESTED, score, f, "",
                           f"no test reaches it; {_n(len(dependents), 'dependent file')}, "
                           f"{_n(branches, 'branch', 'branches')}", [fid]))
    return out


def code_file_count(g: CodeGraph) -> int:
    return sum(1 for n in g.nodes.values() if n.kind == "file" and n.file.endswith(_CODE_EXTS))


def audit(g: CodeGraph, root: Path, tested: Optional[set] = None) -> List[Finding]:
    src = _Sources(root)
    found = dead_code(g) + near_duplicates(g, src) + untested_risky(g, src, tested)
    found.sort(key=lambda f: (-f.score, f.kind, f.title))
    return found


def render(findings: List[Finding], budget: int, limit: int = 20,
           test_files: Optional[int] = None, code_files: Optional[int] = None) -> str:
    out = _Out(budget)
    if code_files == 0:
        return ("The graph has no Kotlin, Swift or TypeScript/JavaScript code here, "
                "so there is nothing to audit.")
    counts = {k: sum(1 for f in findings if f.kind == k) for k in (DEAD, DUP, UNTESTED)}
    out.add(f"audit: {counts[DEAD]} dead, {counts[DUP]} near-duplicate pairs, "
            f"{counts[UNTESTED]} untested risky files (top {min(limit, len(findings))}, ranked):")
    if test_files == 0:
        out.add("  (no test files found in the graph: every file counts as untested)")
    for i, f in enumerate(findings[:limit], 1):
        out.add(f"{i:>3}. {f.line()}")
    if not findings:
        out.add("  nothing found")
    return out.text()


EXPLAIN_SYSTEM = """You review findings from a static code-graph audit.
For each numbered finding you get the code involved. In 1-2 sentences each, say
whether it is real (the graph can miss reflection, framework callbacks and
public API used by other repos) and the smallest fix: delete it, merge the two
functions, or add a test (name what to test). Do not invent code that is not
shown. Keep the numbering."""


def explain_prompt(findings: List[Finding], g: CodeGraph, root: Path, max_lines: int = 40) -> str:
    from .. import redact
    src = _Sources(root)
    parts = []
    for i, f in enumerate(findings, 1):
        parts.append(f"{i}. {f.line()}")
        for nid in f.nodes:
            n = g.nodes[nid]
            if redact.check_file(root, n.file):
                continue
            lines = src.lines(n.file)
            start = n.line - 1 if n.kind != "file" else 0
            end = n.end if n.kind != "file" else len(lines)
            snippet = lines[start:min(end, start + max_lines)]
            parts.append(f"{n.file}:{start + 1}\n```\n" + "\n".join(snippet) + "\n```")
    return "\n\n".join(parts)


def explain(findings: List[Finding], g: CodeGraph, root: Path, cfg=None) -> str:
    """Ask the model about the top findings. Raises LLMError on failure."""
    from ..agent import THINK_RE
    from ..config import Config
    from ..llm import chat
    cfg = cfg or Config(workdir=root)
    messages = [{"role": "system", "content": EXPLAIN_SYSTEM},
                {"role": "user", "content": explain_prompt(findings, g, root)}]
    reply = chat(cfg, messages)
    return THINK_RE.sub("", reply.content).strip()
