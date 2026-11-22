"""Graph facts handed to the micro Editor before it edits a step's files:

- callers (outside these files) of the functions the step is about to change,
- tests the graph says are affected (`impact`),
- up to 5 existing helpers elsewhere whose names resemble what the step
  describes ("reuse candidates"), so the model reuses instead of re-writing.

Plain text, capped; empty when the graph knows nothing about the files.
"""
import difflib
import re
from typing import List, Set, Tuple

from ..repomap import _split_words
from .model import CALLS, CodeGraph
from .queries import _group_targets, impact_files, is_test

FUNC_KINDS = ("function", "constructor")
_IDENT_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)`?")
# Words that say nothing about which helper a step needs.
_STOP = frozenset("""add the and for with from into this that make use new update change
fix call when then else return value values file files function functions method
methods class should could would also only each make sure code step""".split())


def _step_words(summary: str) -> Set[str]:
    return {w for w in _split_words(summary) if len(w) >= 3 and w not in _STOP}


def _step_idents(summary: str) -> List[str]:
    """Identifier-looking tokens in the step text: camelCase, snake_case or name()."""
    out = []
    for m in _IDENT_RE.finditer(summary):
        t = m.group(1)
        followed_by_call = summary[m.end():m.end() + 1] == "("
        if followed_by_call or "_" in t.strip("_") or re.search(r"[a-z][A-Z]", t):
            out.append(t)
    return out


def changed_functions(g: CodeGraph, files: List[str], summary: str) -> List[str]:
    """Functions in `files` the step is likely to change: those named in the
    step text, else (when none is named) every function in them."""
    funcs = [nid for f in files for nid in g.by_file.get(f, ())
             if g.nodes[nid].kind in FUNC_KINDS and not g.nodes[nid].name.startswith("<")]
    words = set(_split_words(summary)) | {t.lower() for t in _step_idents(summary)}
    named = [nid for nid in funcs if g.nodes[nid].name.lower() in words]
    return named or funcs


def external_callers(g: CodeGraph, nid: str, files: Set[str]) -> List[Tuple[str, str]]:
    """(caller short name, path:line) of calls to nid from outside `files`."""
    seen, out = set(), []
    for t in _group_targets(g, nid):
        for e in g.in_edges(t, (CALLS,)):
            if e.file in files or (e.src, e.loc) in seen:
                continue
            seen.add((e.src, e.loc))
            out.append((g.short(e.src), e.loc))
    return sorted(out, key=lambda c: c[1])


def reuse_candidates(g: CodeGraph, summary: str, exclude_files: Set[str],
                     limit: int = 5) -> List[Tuple[float, str]]:
    """Existing functions outside `exclude_files` whose names resemble the step."""
    words = _step_words(summary)
    idents = [t.lower() for t in _step_idents(summary)]
    if not words and not idents:
        return []
    scored = []
    for nid, n in g.nodes.items():
        if (n.kind != "function" or n.file in exclude_files or is_test(n.file)
                or n.name.startswith("<") or len(n.name) < 3):
            continue
        name_words = set(_split_words(n.name))
        overlap = len(name_words & words) / max(len(name_words), 1)
        ratio = 0.0
        for t in idents:
            sm = difflib.SequenceMatcher(None, n.name.lower(), t)
            if sm.real_quick_ratio() >= 0.7 and sm.quick_ratio() >= 0.7:
                ratio = max(ratio, sm.ratio())
        score = max(overlap, ratio if ratio >= 0.7 else 0.0)
        if score >= 0.5:
            scored.append((round(score, 2), nid))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return scored[:limit]


def edit_facts(g: CodeGraph, files: List[str], summary: str, budget: int = 1500) -> str:
    """The text block for the Editor, or '' when there is nothing to say."""
    in_graph = [f for f in files if "file:" + f in g.nodes]
    lines: List[str] = []
    fileset = set(files)
    if in_graph:
        caller_lines = []
        for nid in changed_functions(g, in_graph, summary)[:8]:
            calls = external_callers(g, nid, fileset)
            if calls:
                shown = ", ".join(f"{src} ({loc})" for src, loc in calls[:4])
                more = f" +{len(calls) - 4} more" if len(calls) > 4 else ""
                caller_lines.append(f"- {g.short(nid)}: {shown}{more}")
        if caller_lines:
            lines.append("Callers outside these files (keep them working, or update them):")
            lines += caller_lines
        by_file = impact_files(g, in_graph, depth=3)[0]
        tests = sorted(f for f in by_file if is_test(f))
        if tests:
            lines.append("Affected tests (graph impact): " + ", ".join(tests[:8])
                         + (f" +{len(tests) - 8} more" if len(tests) > 8 else ""))
    reuse = reuse_candidates(g, summary, fileset)
    if reuse:
        lines.append("Reuse candidates (existing helpers with similar names; use one instead of "
                     "writing a new one if it fits):")
        for _score, nid in reuse:
            n = g.nodes[nid]
            lines.append(f"- {g.short(nid)}  {n.loc}" + (f"  {n.sig}" if n.sig else ""))
    if not lines:
        return ""
    text = "Code graph facts:\n" + "\n".join(lines)
    return text if len(text) <= budget else text[:budget].rsplit("\n", 1)[0] + "\n..."
