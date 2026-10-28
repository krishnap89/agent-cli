"""Read-only pipeline: a Jira issue -> an ordered list of subtasks.

Flow (plain Python; no model decides control flow):

    fetch issue -> Locator (keyword presearch + repo map) -> Planner -> print

It reads the ticket from Jira and the relevant source from the repo, then has
the existing Planner role break the work into small, ordered subtasks, each
naming the files it would touch. Nothing is edited: no source is changed, no
document is written, and nothing is sent back to Jira. It is the planning half
of /micro, seeded from a ticket instead of a typed request.

The locate step (keyword presearch + Locator + relevance gate) is reused
wholesale from AnalysisOrchestrator, so this module only fetches the issue,
composes it into a request, runs the Planner, and prints the result.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from rich.console import Console

from .analysis import AnalysisOrchestrator
from .roles import Planner, PlanStep, RoleError

console = Console()


def _issue_to_request(fields: dict) -> str:
    """Compose the fetched issue fields into one request string for the roles.

    Keeps the summary, description and any comments; these carry the subject
    terms the keyword presearch and Locator need, and the acceptance criteria
    the Planner should break down.
    """
    parts = [f"{fields['key']}: {fields['summary']}".strip()]
    desc = (fields.get("description") or "").strip()
    if desc:
        parts.append("Description:\n" + desc)
    comments = [c.strip() for c in fields.get("comments") or [] if c.strip()]
    if comments:
        parts.append("Comments:\n" + "\n---\n".join(comments))
    return "\n\n".join(parts)


class JiraPlanOrchestrator:
    """Turn one Jira issue into an ordered subtask plan (read-only)."""

    def __init__(self, coder, call: Optional[Callable] = None):
        self.coder = coder
        self.cfg = coder.cfg
        # Reuse the analysis pipeline's locate machinery (presearch + Locator +
        # relevance gate + banners/logging) rather than re-implementing it.
        self._analysis = AnalysisOrchestrator(coder, call=call)
        self.planner = Planner(self._analysis._run_role("Planner"))

    def run(self, key: str) -> bool:
        """Fetch issue `key`, plan it, and print the subtasks. Returns True on success."""
        from .. import jira
        if not jira.is_configured(self.cfg):
            console.print("[yellow]Jira is not configured. Set AGENT_JIRA_URL, "
                          "AGENT_JIRA_EMAIL and AGENT_JIRA_TOKEN.[/yellow]")
            return False

        console.print(f"[dim]Fetching {key.strip().upper()} from Jira...[/dim]")
        try:
            fields = jira.get_issue_fields(self.cfg, key)
        except jira.JiraError as e:
            console.print(f"[red]Jira error: {e}[/red]")
            return False

        console.print(f"[bold]{fields['key']}[/bold]: {fields['summary']} "
                      f"[dim]({fields['status'] or 'no status'})[/dim]")
        request = _issue_to_request(fields)

        # Locate the code the ticket concerns. Reuses presearch + Locator + the
        # relevance gate; returns [] (with a printed reason) if nothing matches.
        files = self._analysis._locate(request)
        # Unlike analysis, a plan is still useful with no located files (a brand
        # new feature may touch no existing file), so we plan either way and just
        # note it when the repo gave us nothing to anchor on.
        if not files:
            console.print("[dim]No existing files matched the ticket; planning without "
                          "file anchors.[/dim]")

        self._analysis._banner("Planner")
        try:
            steps, _ = self.planner.run(request=request, files=files)
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return False

        self._print_plan(fields["key"], steps)
        return True

    def _print_plan(self, key: str, steps: List[PlanStep]) -> None:
        console.print(f"\n[green]{len(steps)} subtask(s) for {key}:[/green]")
        for i, step in enumerate(steps, 1):
            console.print(f"[bold]{i}.[/bold] {step.summary}")
            if step.files:
                console.print(f"   [dim]{', '.join(step.files)}[/dim]")
        console.print(f"\n[dim]Read-only plan. Run /micro to implement a step, or "
                      f"jira_issue(key=\"{key}\") in agent mode for the full ticket.[/dim]")
