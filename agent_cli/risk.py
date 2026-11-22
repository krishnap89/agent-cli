"""The one-line risk note printed after coder and /micro edits.

Built from facts (did lint/compile pass? did tests run? which callers of a
changed signature live in files the edit didn't touch?) plus the model's own
`Risk:` line:

    Risk: lint ok, compile ok · tests not run · 1 caller not updated: MainActivity.onCreate (app/.../MainActivity.kt:23) · model: null ids now throw
"""
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_RISK_LINE_RE = re.compile(r"^[ \t]*\**[ \t]*risk[ \t]*\**[ \t]*:[ \t]*(.+?)[ \t]*$", re.I)
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_GRAPH_LANGS = ("kotlin", "swift", "ts")


@dataclass
class CheckFacts:
    """None = did not run."""
    lint: Optional[bool] = None
    compile: Optional[bool] = None
    tests: Optional[bool] = None


def _prose_risk_lines(lines: List[str]) -> List[int]:
    """Indexes of `Risk:` lines outside code fences and SEARCH/REPLACE blocks."""
    hits, in_fence, in_block = [], False, False
    for i, ln in enumerate(lines):
        if ln.startswith("<<<<<<< SEARCH"):
            in_block = True
        elif ln.startswith(">>>>>>> REPLACE"):
            in_block = False
        elif _FENCE_RE.match(ln) and not in_block:
            in_fence = not in_fence
        elif not in_fence and not in_block and _RISK_LINE_RE.match(ln):
            hits.append(i)
    return hits


def model_risk_line(text: str) -> str:
    """The last `Risk: ...` line a model wrote in prose (not in code), or ''."""
    lines = (text or "").splitlines()
    hits = _prose_risk_lines(lines)
    return _RISK_LINE_RE.match(lines[hits[-1]]).group(1).strip("*` ") if hits else ""


def strip_risk_line(text: str) -> str:
    """`text` without its prose `Risk:` lines (they are folded into the risk note)."""
    lines = (text or "").splitlines()
    drop = set(_prose_risk_lines(lines))
    return "\n".join(ln for i, ln in enumerate(lines) if i not in drop).rstrip()


def is_graph_file(rel: str) -> bool:
    """True for files the code graph parses as code (Kotlin, Swift, TS/JS)."""
    from .graph.sources import classify
    return classify(rel) in _GRAPH_LANGS


def _function_names(rel: str, text: str) -> Set[str]:
    """Function names declared in `text`, parsed as the language of `rel`."""
    from .graph import sources
    lang = sources.classify(rel)
    facts = (sources.parse_kotlin(text) if lang == "kotlin" else
             sources.parse_swift(text) if lang == "swift" else
             sources.parse_ts(text, rel) if lang == "ts" else {})
    return {d["name"] for d in facts.get("decls", ())
            if d.get("kind") == "fun" and not d["name"].startswith("<")}


def _calls_to_removed(graph, root: Path, rel: str, old: str, touched: Set[str]) -> List[Tuple[str, str, str]]:
    """Call sites, in untouched graph files, of functions the edit removed or
    renamed (a name in the old text that no function in the project has now)."""
    now = {graph.nodes[n].name for n in graph.by_file.get(rel, ())}
    gone = [name for name in _function_names(rel, old) - now
            if len(name) >= 4 and not any(graph.nodes[n].kind == "function"
                                          for n in graph.by_name.get(name, ()))]
    out = []
    for name in sorted(gone):
        call = re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
        for nid, node in graph.nodes.items():
            f = node.file
            if node.kind != "file" or f in touched or not is_graph_file(f):
                continue
            try:
                text = (root / f).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, ln in enumerate(text.splitlines(), 1):
                if call.search(ln):
                    out.append((f"{name} (removed)", f"call to removed {name}", f"{f}:{i}"))
    return out


def _changed_new_lines(old: str, new: str) -> Set[int]:
    """1-based line numbers in `new` that were added or replaced."""
    out: Set[int] = set()
    sm = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out.update(range(j1 + 1, j2 + 1))
    return out


def callers_not_updated(root: Path, changed: Dict[str, Optional[str]],
                        graph=None) -> List[Tuple[str, str, str]]:
    """(function, caller, path:line) for calls into functions whose declaration
    line changed (or that the edit removed or renamed), made from files this
    edit did not touch. Only languages the code graph parses (Kotlin, Swift,
    TS/JS); [] when the graph fails or no changed file is one of them."""
    if not any(is_graph_file(rel) for rel in changed):
        return []
    try:
        if graph is None:
            from .graph import load_graph
            graph, _ = load_graph(root)
        from .graph.facts import FUNC_KINDS, external_callers
    except Exception:
        return []
    touched = set(changed)
    out = []
    for rel, old in changed.items():
        if old is None or "file:" + rel not in graph.nodes:
            continue  # a new file has no existing callers
        p = root / rel
        new = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        lines = _changed_new_lines(old, new)
        for nid in graph.by_file.get(rel, ()):
            n = graph.nodes[nid]
            if n.kind in FUNC_KINDS and n.line in lines:
                for src, loc in external_callers(graph, nid, touched):
                    out.append((graph.short(nid), src, loc))
        out += _calls_to_removed(graph, root, rel, old, touched)
    return out


def _checks_text(c: CheckFacts) -> str:
    def word(name: str, v: Optional[bool]) -> str:
        return f"{name} {'ok' if v else 'FAILED'}"
    parts = [word(n, v) for n, v in (("lint", c.lint), ("compile", c.compile)) if v is not None]
    if not parts:
        return "not checked"
    if c.compile is None and c.lint:
        parts.append("not compiled")
    return ", ".join(parts)


def risk_note(checks: CheckFacts, missed: List[Tuple[str, str, str]], model_line: str) -> str:
    tests = "tests not run" if checks.tests is None else (
        "tests passed" if checks.tests else "tests FAILED")
    parts = [_checks_text(checks), tests]
    if missed:
        n = len(missed)
        shown = ", ".join(f"{src} ({loc})" for _f, src, loc in missed[:3])
        parts.append(f"{n} caller{'s' if n != 1 else ''} not updated: {shown}"
                     + (f" +{n - 3} more" if n > 3 else ""))
    parts.append(f"model: {model_line}" if model_line else "model: (no risk line)")
    return "Risk: " + " · ".join(parts)
