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
        "You locate the files that must be edited to satisfy a code request.\n"
        "You are given the request and a map of the repository.\n"
        "Reply with ONLY a list of repo-relative file paths, one per line, no prose,\n"
        "no numbering, no code fences. List only files that must be opened or changed,\n"
        "most important first, at most 6. If you cannot tell, list your single best guess."
    )

    def build(self, *, request: str, repo_map: str) -> str:
        return f"Request:\n{request}\n\nRepository map:\n{repo_map}\n\nList the files to edit:"

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
