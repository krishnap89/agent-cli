"""A read-only analysis pipeline for the micro-agent.

Flow (plain Python, no model deciding control flow):

    Locator -> RelevanceGate -> Analyst -> (Diagrammer, when asked) -> Writer

Unlike the edit pipeline in `loop.py`, this one NEVER modifies the files it
reads. It locates the relevant source, analyses it, optionally draws a Mermaid
class diagram, and writes the result to ONE new Markdown document. The created
document is recorded as a single undo snapshot, so `/undo` removes it.

Use this for "show/draw the class diagram", "explain how X works",
"document the structure of Y" -- requests that read code and produce a
document, rather than editing source.
"""
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console

from ..editblock import unified_diff
from .loop import RunLogger
from .modelpick import model_for
from .roles import (
    Locator, Keyworder, RelevanceGate, Analyst, Diagrammer, Writer, RoleError,
)

console = Console()

# The relevance gate (Locator result vs. request subject) is on by default so an
# off-target analysis stops instead of documenting the wrong files. Set
# AGENT_ANALYSIS_GATE=0 to skip it.
_GATE_ENV = "AGENT_ANALYSIS_GATE"

# An LLM proposes extra grep keywords (synonyms, likely identifiers) on top of
# the deterministic ones, so a request like "authentication" can find login/
# session/credential files. On by default; AGENT_ANALYSIS_LLM_KEYWORDS=0 skips
# the call and uses only request_keywords().
_LLM_KEYWORDS_ENV = "AGENT_ANALYSIS_LLM_KEYWORDS"

# Words in the request that mean "draw a diagram", so the Diagrammer runs.
_DIAGRAM_RE = re.compile(
    r"\b(diagram|chart|uml|mermaid|graph|draw|visuali[sz]e|flowchart)\b", re.I)

# Verbs that mean "read and produce a document", not "edit the code". A request
# led by one of these routes to the analysis pipeline instead of the edit loop.
_ANALYSIS_RE = re.compile(
    r"\b(show|draw|diagram|visuali[sz]e|explain|describe|summari[sz]e|document|"
    r"analy[sz]e|outline|map out|walk me through|overview of|how does|what does)\b",
    re.I)

# Where generated docs go, relative to the repo root.
_OUT_DIR = "docs/analysis"


def wants_diagram(request: str) -> bool:
    """True if the request asks for a visual diagram, not just prose."""
    return bool(_DIAGRAM_RE.search(request or ""))


def is_analysis_request(request: str) -> bool:
    """True if the request reads/describes code rather than asking to edit it.

    A heuristic used to route a /micro request to the read-only analysis
    pipeline. Deliberately conservative: it matches explicit read/explain/
    diagram verbs, so an ordinary edit request ("add a flag", "fix the bug")
    still goes to the edit loop.
    """
    return bool(_ANALYSIS_RE.search(request or ""))


def _slug(request: str) -> str:
    """A short filesystem-safe slug from the request, for the output filename."""
    words = re.findall(r"[a-z0-9]+", request.lower())
    slug = "-".join(words[:6]).strip("-")
    return slug or "analysis"


# Words that carry no subject on their own, stripped before the keyword search:
# the analysis verbs, generic code nouns, and ordinary stopwords. What's left is
# the domain terms ("authentication", "checkout") that are worth grepping for.
_STOPWORDS = frozenset(
    "the a an of for to in on and or with show draw diagram chart uml mermaid "
    "graph visualize visualise explain describe summarize summarise document "
    "analyze analyse outline overview how does what is are class classes "
    "structure relationship relationships model models code how its it this "
    "me walk through map out between all its their".split()
)

# How many pre-search hits to hand the Locator, and how many keywords to try.
_PRESEARCH_MAX_FILES = 12
_PRESEARCH_MAX_TERMS = 5


def request_keywords(request: str) -> List[str]:
    """Domain terms worth grepping for, drawn from the request.

    Drops the analysis verbs and stopwords, keeps distinctive words, and adds a
    short stem for long words so "authentication" also matches "auth"/"authMgr".
    Ordered by length (longer = more specific) so the most telling term greps
    first. Returns at most _PRESEARCH_MAX_TERMS.
    """
    seen: Dict[str, None] = {}
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", request.lower()):
        if len(w) < 3 or w in _STOPWORDS:
            continue
        seen.setdefault(w, None)
        if len(w) >= 7:  # a stem catches camelCase/compound identifiers
            seen.setdefault(w[:4], None)
    terms = sorted(seen, key=len, reverse=True)
    return terms[:_PRESEARCH_MAX_TERMS]


class AnalysisOrchestrator:
    """Drive one read-only analysis run against a Coder's files (no edits)."""

    def __init__(self, coder, call: Optional[Callable] = None):
        self.coder = coder
        self.cfg = coder.cfg
        self.root = coder.root
        self._call = call or coder._call_model
        self.logger = RunLogger(self.root)

        self.locator = Locator(self._run_role("Locator"))
        self.keyworder = Keyworder(self._run_role("Keyworder"))
        self.gate = RelevanceGate(self._run_role("RelevanceGate"))
        self.analyst = Analyst(self._run_role("Analyst"))
        self.diagrammer = Diagrammer(self._run_role("Diagrammer"))
        self.writer = Writer(self._run_role("Writer"))

    @property
    def _gate_on(self) -> bool:
        return os.environ.get(_GATE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")

    @property
    def _llm_keywords_on(self) -> bool:
        return os.environ.get(_LLM_KEYWORDS_ENV, "1").strip().lower() not in ("0", "false", "no", "off")

    # ---- role call wrapper (times, logs, and routes the model) ----

    def _run_role(self, role_name: str) -> Callable[[List[dict]], Optional[str]]:
        def call(messages: List[dict]) -> Optional[str]:
            size = sum(len(m.get("content", "")) for m in messages)
            start = time.monotonic()
            with self._role_model(role_name) as model:
                out = self._call(messages)
            self.logger.log(role_name, "", out or "", time.monotonic() - start,
                            extra={"input_chars": size, "model": model})
            return out
        return call

    @contextmanager
    def _role_model(self, role_name: str):
        model = model_for(self.cfg, role_name)
        previous = self.cfg.model
        if self.cfg.auto_model:
            console.print(f"[dim]  {role_name} \N{RIGHTWARDS ARROW} {model}[/dim]")
            if model != previous:
                self.cfg.model = model
        try:
            yield model
        finally:
            self.cfg.model = previous

    def _banner(self, role: str) -> None:
        console.print(f"[bold cyan]▸ {role}[/bold cyan]")

    # ---- entry point ----

    def run(self, request: str) -> bool:
        """Run the analysis for one request. Returns True if a document was written."""
        files = self._locate(request)
        if not files:
            return False

        self._banner("Analyst")
        try:
            analysis, _ = self.analyst.run(request=request, file_blocks=self._file_blocks(files))
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return False

        diagram = None
        if wants_diagram(request):
            self._banner("Diagrammer")
            try:
                diagram, _ = self.diagrammer.run(request=request, analysis=analysis)
            except RoleError as e:
                console.print(f"[yellow]Diagrammer produced no usable diagram ({e}); "
                              "writing the analysis without one.[/yellow]")

        self._banner("Writer")
        try:
            document, _ = self.writer.run(request=request, analysis=analysis, diagram=diagram)
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return False

        return self._write(request, document, files)

    # ---- locate (read-only: never adds editable files or creates placeholders) ----

    def _locate(self, request: str) -> List[str]:
        repo_map = self._repo_map_text(request)
        # Pre-search FIRST: grep the repo for the request's keywords (the
        # Keyworder expands them) so the Locator sees files that actually mention
        # the subject, even ones the truncated repo map left out. This is what
        # lets "authentication" find SupabaseAuthDataSource when it ranked below
        # the map budget. The Locator runs last, choosing from the map + hits.
        hits = self._presearch(request)
        if hits:
            console.print(f"[dim]Keyword search surfaced {len(hits)} file(s).[/dim]")
        self._banner("Locator")
        try:
            files, _ = self.locator.run(request=request, repo_map=repo_map,
                                        hits="\n".join(hits))
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return []
        # Keep only files that exist and are not our own generated docs. The
        # pipeline reads source; it never creates a source file (invented paths
        # are dropped) and must not analyse its own output under docs/analysis/.
        kept = []
        for f in files:
            if not (self.root / f).exists():
                console.print(f"[dim]Ignoring path that does not exist: {f}[/dim]")
            elif self._is_own_output(f):
                console.print(f"[dim]Ignoring our own analysis output: {f}[/dim]")
            else:
                kept.append(f)
        if not kept:
            console.print("[yellow]Locator found no existing files to analyse.[/yellow]")
            return []
        console.print(f"[dim]Files: {', '.join(kept)}[/dim]")
        if not self._is_relevant(request, kept):
            return []
        return kept

    def _is_relevant(self, request: str, files: List[str]) -> bool:
        """Gate: do the located files actually concern the request's subject?

        The Locator always returns a best guess, so for a topic request whose
        subject is not in the repo (e.g. "the authentication" in a project with
        no auth code) it returns unrelated files. The gate stops the run with a
        clear message rather than letting the Analyst document the wrong code.
        On by default; AGENT_ANALYSIS_GATE=0 skips it, and a gate that cannot
        produce a verdict is treated as a pass (never blocks on a model glitch).
        """
        if not self._gate_on:
            return True
        self._banner("RelevanceGate")
        try:
            (relevant, reason), _ = self.gate.run(
                request=request, file_outline=self._file_outline(files))
        except RoleError as e:
            console.print(f"[dim]Relevance gate gave no verdict ({e}); proceeding.[/dim]")
            return True
        if relevant:
            console.print("[dim]Relevance: OK[/dim]")
            return True
        note = f" {reason}" if reason else ""
        console.print(f"[yellow]The files found do not match the request.{note}[/yellow]")
        console.print("[dim]Nothing was written. Name the files or the exact area to analyse, "
                      "or set AGENT_ANALYSIS_GATE=0 to analyse them anyway.[/dim]")
        return False

    def _file_outline(self, files: List[str]) -> str:
        """Cheap per-file context for the gate: path + top-level declaration names.

        Uses the repo map's parsed defs when available so the gate judges on real
        symbols without being sent the full source. Falls back to just the paths.
        """
        rm = self.coder.repo_map
        lines = []
        if rm is not None:
            try:
                rm.build()
            except Exception:
                rm = None
        for rel in files:
            names: List[str] = []
            if rm is not None:
                info = rm.files.get(rel) or {}
                for d in info.get("defs", []):
                    # d = [lineno, depth, kind, name, sig]; top-level only.
                    if len(d) >= 4 and d[1] == 0 and d[3] not in names:
                        names.append(d[3])
            lines.append(f"- {rel}" + (f": {', '.join(names[:12])}" if names else ""))
        return "\n".join(lines)

    def _presearch(self, request: str) -> List[str]:
        """Grep the repo for the request's keywords; return matching file paths.

        Ranked by how many terms a file matches (so a file that hits several
        keywords sorts first), then by path. Read-only and best effort: any
        git/grep failure yields an empty list, and the Locator falls back to the
        repo map alone.
        """
        terms = self._search_terms(request)
        if not terms:
            return []
        counts: Dict[str, int] = {}
        for term in terms:
            for path in self._grep_files(term):
                counts[path] = counts.get(path, 0) + 1
        ranked = sorted(counts, key=lambda p: (-counts[p], p))
        return ranked[:_PRESEARCH_MAX_FILES]

    def _search_terms(self, request: str) -> List[str]:
        """The grep terms: the deterministic ones, unioned with LLM-proposed ones.

        request_keywords() is the floor -- always included -- so the LLM can only
        ADD reach (synonyms, likely identifiers), never remove a term the user
        typed. The Keyworder call is skipped when AGENT_ANALYSIS_LLM_KEYWORDS=0,
        and any failure/empty reply silently leaves just the deterministic terms.
        """
        terms = list(request_keywords(request))
        if not self._llm_keywords_on:
            return terms
        self._banner("Keyworder")
        try:
            extra, _ = self.keyworder.run(request=request)
        except RoleError as e:
            console.print(f"[dim]Keyworder gave no terms ({e}); using typed keywords only.[/dim]")
            return terms
        added = [t for t in extra if t not in terms]
        for t in added:
            terms.append(t)
        if added:
            console.print(f"[dim]Keyworder added: {', '.join(added)}[/dim]")
        # Cap the total so the grep sequence stays short even if both sources
        # are generous; deterministic terms come first, so they're never cut.
        return terms[:_PRESEARCH_MAX_TERMS + 10]

    def _grep_files(self, term: str) -> List[str]:
        """Repo-relative paths whose contents contain `term` (literal, case-insensitive).

        Uses `git grep -l` inside a repo (respects .gitignore, includes untracked)
        and falls back to `grep -rilI` otherwise. Returns [] on any error.
        """
        import subprocess
        try:
            in_git = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.root, capture_output=True, text=True, timeout=10,
            ).returncode == 0
            if in_git:
                cmd = ["git", "grep", "-l", "-i", "-I", "--untracked", "-F", "-e", term]
            else:
                cmd = ["grep", "-rilI", "-F", "-e", term, "."]
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        if r.returncode > 1:  # 1 = no matches; >1 = a real error
            return []
        out = []
        for line in r.stdout.splitlines():
            p = line[2:] if line.startswith("./") else line
            if p and not self._is_own_output(p):
                out.append(p)
        return out

    def _is_own_output(self, rel: str) -> bool:
        """True if `rel` is a doc this pipeline itself generated.

        The analysis docs under docs/analysis/ are prose about code and match the
        very keywords we grep for, so without this filter a prior run's output
        becomes the top 'source' for the next run -- the pipeline analysing its
        own output. They are output, never input.
        """
        norm = rel.replace("\\", "/")
        return norm == _OUT_DIR or norm.startswith(_OUT_DIR + "/")

    def _repo_map_text(self, request: str) -> str:
        if self.coder.repo_map is None:
            return "(repo map is off)"
        self.coder.repo_map.build()
        in_chat = set(self.coder.editable) | set(self.coder.read_only)
        return self.coder.repo_map.render(query=request, budget=self.cfg.map_chars, exclude=in_chat)

    # ---- write the one output document (and record it for /undo) ----

    def _write(self, request: str, document: str, sources: List[str]) -> bool:
        rel = self._output_path(request)
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        body = document if document.endswith("\n") else document + "\n"
        p.write_text(body, encoding="utf-8")

        # One undo snapshot for the whole run, same shape the edit pipeline uses.
        changed: Dict[str, Optional[str]] = {rel: old}
        self.coder.undo_stack.append(changed)
        self.coder.last_diffs = []
        d = unified_diff(rel, old or "", body)
        if d:
            self.coder.last_diffs.append(d)
        if self.coder.repo_map is not None:
            self.coder._all_files = None

        verb = "Updated" if old is not None else "Wrote"
        console.print(f"\n[green]{verb} {rel} from {len(sources)} file(s).[/green]")
        console.print(f"[dim]{rel} — /undo to remove it.[/dim]")
        self.coder.history.append({"role": "user", "content": f"/analyze {request}"})
        self.coder.history.append(
            {"role": "assistant", "content": f"{verb} analysis document {rel}."})
        return True

    def _output_path(self, request: str) -> str:
        return f"{_OUT_DIR}/{_slug(request)}.md"

    def _file_blocks(self, files: List[str]) -> str:
        return "\n".join(self.coder._file_block(f) for f in files if (self.root / f).exists())
