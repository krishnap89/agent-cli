"""The micro-agent roles: one small class per job.

Every LLM-based role follows the same shape:

- `system` and a `build(...)` that turns the role's own inputs into the user
  message. A role is given ONLY its own inputs -- there is no shared chat
  history, so a mistake in one role can't poison the others.
- `parse(text)` turns the reply into a typed result or raises `RoleError` with
  a message the model could act on. `run(...)` calls the model and, on a
  `RoleError`, retries exactly once (re-stating the format), then gives up.

The non-LLM Checker lives in `checker.py`; it is not a Role.
"""
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ..editblock import EditBlock, parse_edit_blocks

# A model-caller: takes a messages list, returns the reply text, or None if the
# request itself failed (network/empty). The orchestrator passes Coder._call_model
# in production and a fake in tests.
CallFn = Callable[[List[dict]], Optional[str]]


class RoleError(Exception):
    """A role's reply did not match its required format."""


@dataclass
class PlanStep:
    """One step of the Planner's plan: a short instruction plus the files it touches."""
    summary: str
    files: List[str]


class Role:
    """Base class: one strict-format LLM call with a single retry on bad format."""

    name = "role"
    system = ""

    def __init__(self, call: CallFn):
        self._call = call

    def build(self, **inputs) -> str:
        """Return the user-message text for this role from its own inputs."""
        raise NotImplementedError

    def parse(self, text: str):
        """Turn a reply into this role's typed result, or raise RoleError."""
        raise NotImplementedError

    def run(self, **inputs):
        """Call the model, parse, and retry once on a RoleError.

        Returns (result, user_message). Raises RoleError if the second reply is
        still malformed, or RoleError("no reply ...") if the model call failed.
        """
        user = self.build(**inputs)
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]
        text = self._call(messages)
        if text is None:
            raise RoleError(f"{self.name}: the model request failed")
        try:
            return self.parse(text), user
        except RoleError as first:
            # One retry: show the model its reply and restate the format.
            retry_messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That reply was not in the required format: {first}\n"
                                             "Reply again, in the exact format described above, and nothing else."},
            ]
            text2 = self._call(retry_messages)
            if text2 is None:
                raise RoleError(f"{self.name}: the model request failed on retry")
            return self.parse(text2), user  # a second RoleError propagates


# ---- Locator -------------------------------------------------------------------------

class Locator(Role):
    """Pick the files that need editing, out of a repo map / candidate list."""

    name = "Locator"
    system = (
        "You locate the files that must be opened to satisfy a code request.\n"
        "You are given the request, a map of the repository, and (sometimes) a\n"
        "list of files whose contents already match the request's keywords --\n"
        "prefer those, they were found by searching the code. Reply with ONLY a\n"
        "list of repo-relative file paths, one per line, no prose, no numbering,\n"
        "no code fences. List only files that must be opened or changed, most\n"
        "important first, at most 6. If you cannot tell, list your single best guess."
    )

    def build(self, *, request: str, repo_map: str, hits: str = "") -> str:
        hits_section = f"\n\nFiles whose contents match the request (from a keyword search):\n{hits}" if hits else ""
        return (f"Request:\n{request}\n\nRepository map:\n{repo_map}{hits_section}\n\n"
                "List the files to open:")

    def parse(self, text: str) -> List[str]:
        files = []
        for line in text.splitlines():
            s = line.strip().strip("`-*0123456789. \t")
            s = s.strip("`\"' ")
            if not s or " " in s:
                continue
            if "/" in s or "." in s:
                files.append(s)
        if not files:
            raise RoleError("no file paths found; list one repo-relative path per line")
        return files[:6]


# ---- Planner -------------------------------------------------------------------------

_STEP_RE = re.compile(r"^\s*STEP\s*:\s*(.+?)\s*$", re.I)
_FILES_RE = re.compile(r"^\s*FILES\s*:\s*(.+?)\s*$", re.I)


class Planner(Role):
    """Break the request into an ordered list of small, single-purpose steps."""

    name = "Planner"
    system = (
        "You turn a code request into a short, ordered plan of edit steps.\n"
        "You are given the request and the files available to edit.\n"
        "Reply with one or more steps, each EXACTLY two lines:\n"
        "STEP: <one sentence describing the change>\n"
        "FILES: <comma-separated repo-relative paths this step edits>\n"
        "Keep steps small and independent where possible. No other text, no blank\n"
        "lines inside a step, no numbering, no code."
    )

    def build(self, *, request: str, files: List[str]) -> str:
        listing = "\n".join(files) if files else "(none yet)"
        return f"Request:\n{request}\n\nFiles available to edit:\n{listing}\n\nWrite the plan:"

    def parse(self, text: str) -> List[PlanStep]:
        steps: List[PlanStep] = []
        pending_summary: Optional[str] = None
        for line in text.splitlines():
            m = _STEP_RE.match(line)
            if m:
                pending_summary = m.group(1).strip()
                continue
            m = _FILES_RE.match(line)
            if m and pending_summary is not None:
                files = [p.strip().strip("`\"'") for p in m.group(1).split(",")]
                files = [p for p in files if p]
                steps.append(PlanStep(pending_summary, files))
                pending_summary = None
        if not steps:
            raise RoleError("no steps found; each step is 'STEP: ...' then 'FILES: ...'")
        return steps


# ---- Editor / Fixer ------------------------------------------------------------------

# The edit prompt reuses the same SEARCH/REPLACE contract as the main coder so a
# model that can drive one can drive the other. Kept terse; roles never see the
# long CODE_SYSTEM preamble or any history.
_EDIT_FORMAT = (
    "Make the change with SEARCH/REPLACE blocks in this shape (the first line is\n"
    "the real repo path of the file you are editing, NOT a literal to copy):\n"
    "\n"
    "<the file's path>\n"
    "```\n"
    "<<<<<<< SEARCH\n"
    "exact existing lines\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n"
    "```\n"
    "\n"
    "The path on the first line MUST be one of the files listed above, copied\n"
    "exactly. Never invent a path and never write a placeholder like 'path/to/file'.\n"
    "The SEARCH lines must match the file EXACTLY. Keep blocks small (changed lines\n"
    "plus 1-3 around them). Empty REPLACE deletes code.\n"
    "To CREATE a new file or fill an empty one, leave the SEARCH section EMPTY\n"
    "(nothing between <<<<<<< SEARCH and =======) and put the whole file in REPLACE.\n"
    "Only edit the files shown. Output the blocks and nothing else."
)


def _parse_edits(text: str, default_path: Optional[str]) -> Tuple[List[EditBlock], List[str]]:
    blocks, problems = parse_edit_blocks(text, default_path=default_path)
    if not blocks and not problems:
        raise RoleError("no SEARCH/REPLACE block found")
    return blocks, problems


class Editor(Role):
    """Carry out one plan step by writing SEARCH/REPLACE blocks for its files."""

    name = "Editor"
    system = "You are a careful software engineer applying one small, specific change.\n\n" + _EDIT_FORMAT

    def __init__(self, call: CallFn):
        super().__init__(call)
        self._default_path: Optional[str] = None

    def build(self, *, step: PlanStep, file_blocks: str,
              new_files: Optional[List[str]] = None, symbols: str = "") -> str:
        self._default_path = step.files[0] if len(step.files) == 1 else None
        new_note = ""
        if new_files:
            listing = ", ".join(new_files)
            new_note = (f"\nThese files do not exist yet and must be CREATED with an empty "
                        f"SEARCH section: {listing}\n")
        sym_note = f"\n{symbols}\n" if symbols else ""
        return (f"Step to implement: {step.summary}\n{new_note}{sym_note}\n"
                f"Current contents of the files you may edit:\n\n{file_blocks}\n\n"
                "Write the SEARCH/REPLACE blocks for this step:")

    def parse(self, text: str):
        return _parse_edits(text, self._default_path)


class Fixer(Role):
    """Repair the files after a check failed, using only the error and current contents."""

    name = "Fixer"
    system = ("You fix code that failed a check (lint, compile, or tests).\n"
              "You are given the check output and the current file contents.\n\n" + _EDIT_FORMAT)

    def __init__(self, call: CallFn):
        super().__init__(call)
        self._default_path: Optional[str] = None

    def build(self, *, error: str, file_blocks: str,
              default_path: Optional[str] = None, symbols: str = "") -> str:
        self._default_path = default_path
        sym_note = f"\n{symbols}\n" if symbols else ""
        return (f"A check failed with this output:\n\n{error}\n{sym_note}\n"
                f"Current contents of the files you may edit:\n\n{file_blocks}\n\n"
                "Fix the problem. Write only the SEARCH/REPLACE blocks that are needed:")

    def parse(self, text: str):
        return _parse_edits(text, self._default_path)


# ---- Reviewer ------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(OK|CHANGES)\b", re.I)
_NOTES_RE = re.compile(r"^\s*NOTES\s*:\s*(.*)$", re.I)


class Reviewer(Role):
    """Judge whether a finished step's diff matches its intent. Advisory only."""

    name = "Reviewer"
    system = (
        "You review a code change against the step it was meant to implement.\n"
        "You are given the step and a unified diff. Reply EXACTLY two lines:\n"
        "VERDICT: OK   (the diff implements the step and introduces no obvious bug)\n"
        "or\n"
        "VERDICT: CHANGES\n"
        "NOTES: <one sentence on what is wrong>\n"
        "No other text."
    )

    def build(self, *, step: PlanStep, diff: str) -> str:
        return f"Step: {step.summary}\n\nDiff:\n{diff or '(no changes were made)'}\n\nReview it:"

    def parse(self, text: str) -> Tuple[bool, str]:
        verdict = None
        notes = ""
        for line in text.splitlines():
            m = _VERDICT_RE.match(line)
            if m:
                verdict = m.group(1).upper() == "OK"
            m = _NOTES_RE.match(line)
            if m:
                notes = m.group(1).strip()
        if verdict is None:
            raise RoleError("no 'VERDICT: OK' or 'VERDICT: CHANGES' line found")
        return verdict, notes


# ---- Keyworder (read-only analysis pipeline) -----------------------------------------

class Keyworder(Role):
    """Expand a request into code-search keywords (synonyms, likely identifiers).

    The deterministic extractor can only grep words the user literally typed, so
    "authentication" never finds a `login`/`session`/`credential` file. This role
    proposes the vocabulary the code is likely to use. It guesses blind (it does
    not see the repo); grep then validates the guesses against the real files, so
    an invented term simply matches nothing. Advisory: its terms are UNIONED with
    the deterministic ones, never a replacement.
    """

    name = "Keyworder"
    system = (
        "You turn a request about a codebase into search keywords.\n"
        "Given the request, list the words and short identifiers most likely to\n"
        "appear IN THE CODE for that subject -- the concept itself plus common\n"
        "synonyms and naming conventions. For 'authentication' that might be:\n"
        "auth, login, signin, session, credential, token, oauth.\n"
        "Reply with ONLY the keywords, comma-separated, lowercase, no prose, at\n"
        "most 10. Prefer short stems that match many identifiers (auth, not\n"
        "authentication). No file paths, no explanations."
    )

    def build(self, *, request: str) -> str:
        return f"Request:\n{request}\n\nSearch keywords:"

    def parse(self, text: str) -> List[str]:
        # Accept commas or newlines; strip bullets/quotes; keep word-ish tokens.
        raw = re.split(r"[,\n]", text)
        terms: List[str] = []
        for tok in raw:
            t = tok.strip().strip("`\"'-*0123456789. \t").lower()
            if t and re.fullmatch(r"[a-z][a-z0-9_]{1,}", t) and t not in terms:
                terms.append(t)
        if not terms:
            raise RoleError("no keywords found; reply with comma-separated words")
        return terms[:10]


# ---- Relevance gate (read-only analysis pipeline) ------------------------------------

_RELEVANT_RE = re.compile(r"^\s*RELEVANT\s*:\s*(YES|NO)\b", re.I)


class RelevanceGate(Role):
    """Judge whether the located files are actually about what the request asked.

    The Locator always returns its best guess, even when nothing in the repo
    matches -- fine for editing (the user named the change) but bad for analysis,
    where confidently documenting the wrong files is worse than admitting the
    code was not found. This gate sees only the request and the file list (paths
    and a one-line outline), not the full source, so it stays cheap.
    """

    name = "RelevanceGate"
    system = (
        "You decide whether a set of files is relevant to an analysis request.\n"
        "You are given the request and a list of candidate files (paths and a\n"
        "short outline of each). Reply EXACTLY one or two lines:\n"
        "RELEVANT: YES   (at least one file is about the subject of the request)\n"
        "or\n"
        "RELEVANT: NO\n"
        "REASON: <one short sentence on what the request asked for vs. what the files are>\n"
        "Judge by subject only. If the request names a topic (e.g. authentication)\n"
        "and none of the files concern it, answer NO. No other text."
    )

    def build(self, *, request: str, file_outline: str) -> str:
        return (f"Request:\n{request}\n\nCandidate files:\n{file_outline}\n\n"
                "Are these files relevant to the request?")

    def parse(self, text: str) -> Tuple[bool, str]:
        relevant = None
        reason = ""
        for line in text.splitlines():
            m = _RELEVANT_RE.match(line)
            if m:
                relevant = m.group(1).upper() == "YES"
            m = re.match(r"^\s*REASON\s*:\s*(.*)$", line, re.I)
            if m:
                reason = m.group(1).strip()
        if relevant is None:
            raise RoleError("no 'RELEVANT: YES' or 'RELEVANT: NO' line found")
        return relevant, reason


# ---- Analyst / Diagrammer / Writer (read-only analysis pipeline) ---------------------

class Analyst(Role):
    """Read the located files and describe their structure, without changing them.

    This role never edits code. It produces a plain-text analysis of the classes,
    their attributes, and how they relate -- the raw material the Writer turns
    into a document (and the Diagrammer, when asked, into a diagram).
    """

    name = "Analyst"
    system = (
        "You analyse source code and explain its structure. You NEVER modify code\n"
        "and you NEVER output SEARCH/REPLACE blocks or file edits.\n"
        "You are given a request and the full contents of the relevant files.\n"
        "Describe only what the code actually contains: the classes/types, their\n"
        "key attributes, and the relationships between them (inheritance, usage,\n"
        "composition). Be concrete and name real identifiers. Plain prose and short\n"
        "bullet lists only."
    )

    def build(self, *, request: str, file_blocks: str) -> str:
        return (f"Request:\n{request}\n\nFiles to analyse (read-only):\n\n{file_blocks}\n\n"
                "Write the analysis:")

    def parse(self, text: str) -> str:
        analysis = text.strip()
        if not analysis:
            raise RoleError("empty analysis")
        return analysis


class Diagrammer(Role):
    """Turn the Analyst's findings into a Mermaid classDiagram. Output only the diagram."""

    name = "Diagrammer"
    system = (
        "You turn a code analysis into a single Mermaid class diagram.\n"
        "Reply with ONE fenced Mermaid block and nothing else:\n"
        "```mermaid\n"
        "classDiagram\n"
        "  ...\n"
        "```\n"
        "Use classDiagram syntax: `ClassName` nodes, attributes/methods inside a\n"
        "class block, and relationship arrows (<|-- inheritance, --> association,\n"
        "*-- composition, o-- aggregation). Only include classes and relationships\n"
        "named in the analysis. No prose before or after the block."
    )

    def build(self, *, request: str, analysis: str) -> str:
        return (f"Request:\n{request}\n\nAnalysis to diagram:\n\n{analysis}\n\n"
                "Write the Mermaid class diagram:")

    def parse(self, text: str) -> str:
        m = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        body = m.group(1).strip() if m else text.strip()
        if not body:
            raise RoleError("no mermaid diagram found")
        if "classDiagram" not in body and "graph" not in body:
            raise RoleError("diagram is not a Mermaid classDiagram/graph")
        return body


class Writer(Role):
    """Assemble the final analysis document (prose + optional diagram) as Markdown.

    The Writer produces document text only -- the orchestrator writes it to the
    output file. It never emits SEARCH/REPLACE blocks or touches source files.
    """

    name = "Writer"
    system = (
        "You write a clear Markdown document from an analysis of some code.\n"
        "You are given the request, the analysis, and (optionally) a Mermaid\n"
        "diagram to embed. Produce a well-structured Markdown document: a title,\n"
        "the diagram in a ```mermaid block if one was given, then the explanation.\n"
        "Describe only what the analysis contains. Output the Markdown document\n"
        "and nothing else -- no SEARCH/REPLACE blocks, no commentary about files."
    )

    def build(self, *, request: str, analysis: str, diagram: Optional[str] = None) -> str:
        diagram_note = ""
        if diagram:
            diagram_note = (f"\nEmbed this diagram verbatim inside a ```mermaid code block:\n\n"
                            f"{diagram}\n")
        return (f"Request:\n{request}\n\nAnalysis:\n\n{analysis}\n{diagram_note}\n"
                "Write the Markdown document:")

    def parse(self, text: str) -> str:
        doc = text.strip()
        if not doc:
            raise RoleError("empty document")
        return doc


# ---- Summarizer ----------------------------------------------------------------------

class Summarizer(Role):
    """Write the one-paragraph summary of everything the run changed."""

    name = "Summarizer"
    system = (
        "You summarise a completed set of code edits for the developer.\n"
        "You are given the original request and the combined diff.\n"
        "Describe ONLY what the diff actually changed -- the added and removed\n"
        "lines. Do not mention files, classes, or behaviour that are not in the\n"
        "diff, and do not describe what the request asked for unless the diff did it.\n"
        "Reply with a short plain-text summary: 1-4 sentences. No code, no markdown\n"
        "headings, no bullet lists. If the diff is empty, say nothing changed."
    )

    def build(self, *, request: str, diff: str) -> str:
        return f"Original request:\n{request}\n\nCombined diff of all edits:\n{diff}\n\nSummarise what changed:"

    def parse(self, text: str) -> str:
        summary = text.strip()
        if not summary:
            raise RoleError("empty summary")
        return summary
