"""micro: a refactor of the coder edit flow into a small multi-role agent loop.

Instead of one long chat where the model both plans and edits with the whole
history in front of it, the work is split across focused roles that each see
only their own inputs:

    Locator -> Planner -> for each step:
        Editor -> Checker -> (Fixer -> Checker)* -> Reviewer
    -> Summarizer

Each LLM role (Locator, Planner, Editor, Fixer, Reviewer, Summarizer) has its
own short prompt and a strict output format parsed in Python; one bad-format
reply is retried once, then the run stops with a clear message. The Checker
uses no model: it runs the existing built-in lint and (on Android projects)
the Gradle compile/test checks. The orchestrator in `loop.py` wires them
together, logs every role call to .agent-cache/runs/<timestamp>.jsonl, and
records one undo snapshot for the whole run.
"""
from .roles import (
    Locator, Planner, Editor, Fixer, Reviewer, Summarizer,
    Keyworder, RelevanceGate, Analyst, Diagrammer, Writer,
    Role, RoleError, PlanStep,
)
from .checker import Checker, CheckResult
from .loop import MicroOrchestrator, RunLogger
from .analysis import AnalysisOrchestrator, wants_diagram, is_analysis_request

__all__ = [
    "Locator", "Planner", "Editor", "Fixer", "Reviewer", "Summarizer",
    "Keyworder", "RelevanceGate", "Analyst", "Diagrammer", "Writer",
    "Role", "RoleError", "PlanStep",
    "Checker", "CheckResult",
    "MicroOrchestrator", "RunLogger",
    "AnalysisOrchestrator", "wants_diagram", "is_analysis_request",
]
