"""The orchestrator that wires the micro-agent roles into one run.

Flow (plain Python, no model deciding control flow):

    Locator -> Planner -> for each step:
        Editor -> Checker -> (Fixer -> Checker, up to AGENT_FIX_ROUNDS) -> Reviewer
    -> Summarizer

- Fast path: if the request names exactly one file already in the chat, skip
  the Locator and the Planner and run a single step on that file.
- The loop stops a step early if the SAME check error comes back twice.
- Every role call is logged to .agent-cache/runs/<timestamp>.jsonl with its
  input size, output and duration.
- The whole run is one undo snapshot: one /undo reverts everything.
"""
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console

from ..editblock import unified_diff
from .checker import Checker, CheckResult
from .modelpick import model_for
from .roles import (
    Locator, Planner, Editor, Fixer, Reviewer, Summarizer,
    PlanStep, RoleError,
)

console = Console()

# Max Fixer rounds per step after the first check. Read from the environment so
# it sits with the other AGENT_* knobs; falls back to cfg.max_fix_attempts.
_FIX_ROUNDS_ENV = "AGENT_FIX_ROUNDS"
# Extra fix passes driven by a Reviewer CHANGES verdict (not a compile/lint fail).
_REVIEW_ROUNDS_ENV = "AGENT_REVIEW_ROUNDS"

# Extensions the Locator may name as a brand-new file to create. Kept to source
# files so a stray path like "path/to/file" (no extension) is never creatable.
_CREATABLE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".go", ".rs",
    ".rb", ".php", ".cs", ".c", ".h", ".cpp", ".swift", ".sh", ".md", ".json",
    ".yml", ".yaml", ".html", ".css", ".sql", ".toml", ".xml", ".gradle",
}


def _apply_signature(detail: str) -> str:
    """Fingerprint an edit-apply failure so a repeat of the same one stops the step.

    Like the Checker's signature, this ignores line numbers and whitespace,
    which jitter between otherwise-identical failures.
    """
    norm = re.sub(r"\d+", "#", detail)
    norm = re.sub(r"\s+", " ", norm).strip()
    return f"apply:{norm[:400]}"


class RunLogger:
    """Append one JSON line per role call to .agent-cache/runs/<timestamp>.jsonl."""

    def __init__(self, root: Path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = root / ".agent-cache" / "runs" / f"{ts}.jsonl"
        self._ok = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False  # logging must never break a run

    def log(self, role: str, input_text: str, output: str, duration: float, extra: Optional[dict] = None) -> None:
        if not self._ok:
            return
        rec = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "role": role,
            "input_chars": len(input_text or ""),
            "output": (output or "")[:4000],
            "duration_s": round(duration, 3),
        }
        if extra:
            rec.update(extra)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            self._ok = False


class MicroOrchestrator:
    """Drive one micro-agent run against a Coder's files and checks."""

    def __init__(self, coder, call: Optional[Callable] = None, checker: Optional[Checker] = None):
        self.coder = coder
        self.cfg = coder.cfg
        self.root = coder.root
        # Reuse the coder's model caller (handles streaming, retries, think-strip)
        # unless a fake is injected for tests.
        self._call = call or coder._call_model
        self.checker = checker or Checker(
            self.cfg, self.root, coder.android_info, coder.run_lint,
        )
        self.logger = RunLogger(self.root)
        # New files this run created as empty placeholders, so we can remove any
        # the Editor never actually filled.
        self._created: List[str] = []

        self.locator = Locator(self._run_role("Locator"))
        self.planner = Planner(self._run_role("Planner"))
        self.editor = Editor(self._run_role("Editor"))
        self.fixer = Fixer(self._run_role("Fixer"))
        self.reviewer = Reviewer(self._run_role("Reviewer"))
        self.summarizer = Summarizer(self._run_role("Summarizer"))

    # ---- role call wrapper: times and logs every model call ----

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
        """Point cfg.model at this role's auto-selected model for one call.

        build_payload reads cfg.model at request time, so overriding it here (and
        restoring it in finally) is all it takes to route a single role to a
        different model. A no-op when auto_model is off: model == cfg.model.
        """
        model = model_for(self.cfg, role_name)
        previous = self.cfg.model
        if self.cfg.auto_model:
            # Show which model this role runs on, every role, so the routing is
            # visible even when consecutive roles share a model.
            self._banner_model(role_name, model)
            if model != previous:
                self.cfg.model = model
        try:
            yield model
        finally:
            self.cfg.model = previous

    def _banner_model(self, role_name: str, model: str) -> None:
        console.print(f"[dim]  {role_name} \N{RIGHTWARDS ARROW} {model}[/dim]")

    @property
    def _fix_rounds(self) -> int:
        raw = os.environ.get(_FIX_ROUNDS_ENV, "")
        if raw.isdigit():
            return int(raw)
        return self.cfg.max_fix_attempts

    @property
    def _review_rounds(self) -> int:
        """Extra Fixer passes driven by a Reviewer CHANGES verdict (0 disables)."""
        raw = os.environ.get(_REVIEW_ROUNDS_ENV, "")
        return int(raw) if raw.isdigit() else 1

    # ---- display ----

    def _banner(self, role: str, round_no: Optional[int] = None, extra: str = "") -> None:
        rnd = f" · round {round_no}" if round_no is not None else ""
        tail = f" {extra}" if extra else ""
        console.print(f"[bold cyan]▸ {role}[/bold cyan][dim]{rnd}{tail}[/dim]")

    # ---- entry point ----

    def run(self, request: str) -> bool:
        """Run the loop for one request. Returns True if any file changed."""
        changed: Dict[str, Optional[str]] = {}
        try:
            steps = self._locate_and_plan(request)
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return False
        if steps is None:
            return False

        for i, step in enumerate(steps, 1):
            console.print(f"[dim]Step {i}/{len(steps)}: {step.summary}[/dim]")
            if not self._run_step(step, changed):
                break  # a hard stop inside the step (bad format or repeated error)

        self._cleanup_empty_created()

        if not changed:
            console.print("[yellow]No files were changed.[/yellow]")
            return False

        self._finish(request, changed)
        return True

    def _cleanup_empty_created(self) -> None:
        """Remove placeholder files we created that the Editor left empty."""
        for f in self._created:
            p = self.root / f
            try:
                if p.exists() and p.read_text(encoding="utf-8", errors="replace").strip() == "":
                    p.unlink()
                    self.coder.editable.pop(f, None)
                    console.print(f"[dim]Removed unused empty file {f}[/dim]")
            except OSError:
                pass

    # ---- locate + plan (or fast path) ----

    def _locate_and_plan(self, request: str) -> Optional[List[PlanStep]]:
        fast = self._fast_path_file(request)
        if fast is not None:
            console.print(f"[dim]Fast path: editing {fast} (skipping Locator and Planner)[/dim]")
            return [PlanStep(summary=request, files=[fast])]

        self._banner("Locator")
        repo_map = self._repo_map_text(request)
        files, _ = self.locator.run(request=request, repo_map=repo_map)
        # Keep files that exist, are already in the chat, or are a plausible NEW
        # path to create (a feature request often needs a new file). Drop wild
        # paths that are neither real nor plausibly creatable.
        kept = []
        for f in files:
            if (self.root / f).exists() or f in self.coder.editable:
                kept.append(f)
            elif self._plausible_new_path(f):
                console.print(f"[dim]New file to create: {f}[/dim]")
                kept.append(f)
            else:
                console.print(f"[dim]Ignoring implausible path from Locator: {f}[/dim]")
        files = kept
        if not files:
            console.print("[yellow]Locator found no files to edit or create.[/yellow]")
            return None
        for f in files:
            # Create the file ourselves (empty) so coder.add never has to prompt
            # for it; the Editor then fills it with an empty-SEARCH block.
            p = self.root / f
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                self._created.append(f)
            if f not in self.coder.editable:
                self.coder.add([f])
        console.print(f"[dim]Files: {', '.join(files)}[/dim]")

        self._banner("Planner")
        steps, _ = self.planner.run(request=request, files=files)
        steps = self._validate_steps(steps, files, request)
        if not steps:
            console.print("[yellow]No usable plan steps after validation.[/yellow]")
            return None
        console.print(f"[dim]{len(steps)} step(s) planned.[/dim]")
        return steps

    def _validate_steps(self, steps: List[PlanStep], located: List[str],
                        request: str) -> List[PlanStep]:
        """Reconcile planned step files against the files the Locator approved.

        The Planner may only touch files the Locator chose (existing ones, or
        new files it named for creation). A step that references some other path
        -- a hallucinated or mistyped one -- has that path dropped; if a step is
        left with no valid file, the Planner is asked once to redo the plan
        against the exact located files, and whatever survives is used.
        """
        allowed = {self._norm_one(f) for f in located}

        def clean(step_list: List[PlanStep]) -> List[PlanStep]:
            out = []
            for s in step_list:
                kept, dropped = [], []
                for f in s.files:
                    (kept if self._norm_one(f) in allowed else dropped).append(f)
                if dropped:
                    console.print(f"[dim]Planner referenced unknown path(s): "
                                  f"{', '.join(dropped)} — not among the located files.[/dim]")
                if kept:
                    out.append(PlanStep(summary=s.summary, files=kept))
            return out

        cleaned = clean(steps)
        if cleaned:
            return cleaned

        # Every step named only unknown files. Give the Planner one corrective
        # pass with the located files spelled out, then take whatever is valid.
        console.print("[yellow]Plan referenced no located files; asking the Planner "
                      "to re-plan against them.[/yellow]")
        try:
            retry_steps, _ = self.planner.run(
                request=f"{request}\n\nUse ONLY these exact file paths: {', '.join(located)}",
                files=located,
            )
        except RoleError:
            return []
        return clean(retry_steps)

    def _fast_path_file(self, request: str) -> Optional[str]:
        """If the request names exactly one file already in the chat, return it."""
        in_chat = list(self.coder.editable)
        named = []
        for f in in_chat:
            name = Path(f).name
            # Match the full path or the bare filename as a whole word.
            if f in request or re.search(rf"\b{re.escape(name)}\b", request):
                named.append(f)
        return named[0] if len(named) == 1 else None

    def _repo_map_text(self, request: str) -> str:
        if self.coder.repo_map is None:
            return "(repo map is off)"
        self.coder.repo_map.build()
        in_chat = set(self.coder.editable) | set(self.coder.read_only)
        return self.coder.repo_map.render(query=request, budget=self.cfg.map_chars, exclude=in_chat)

    # ---- one step: Editor -> Checker -> (Fixer -> Checker)* -> Reviewer ----

    def _run_step(self, step: PlanStep, changed: Dict[str, Optional[str]]) -> bool:
        self._banner("Editor")
        new_files = [f for f in step.files if self._is_new_file(f)]
        try:
            (blocks, problems), _ = self.editor.run(
                step=step, file_blocks=self._file_blocks(step.files),
                new_files=new_files, symbols=self._symbol_context(step),
            )
        except RoleError as e:
            console.print(f"[red]Stopped: {e}[/red]")
            return False
        allowed = self._normalize(step.files)
        apply_errors = self._apply_scoped(blocks, changed, allowed) + problems
        step_files = self._touched(step, changed)

        # Run the check/fix loop; then the Reviewer. A CHANGES verdict re-enters
        # the fix loop with the Reviewer's note as the "error", up to
        # AGENT_REVIEW_ROUNDS extra passes -- so "compiles but doesn't match the
        # step" is caught, not just "doesn't compile".
        for review_round in range(self._review_rounds + 1):
            if not self._fix_until_ok(step, step_files, changed, allowed, apply_errors):
                return False  # hard stop inside the fix loop (bad format / repeated error)
            step_files = self._touched(step, changed)

            ok, notes = self._review(step, step_files, changed)
            if ok or notes is None:
                return True
            if review_round >= self._review_rounds:
                console.print(f"[yellow]Reviewer still flags this step after "
                              f"{self._review_rounds} review round(s); keeping the edits.[/yellow]")
                return True

            # Feed the Reviewer's note back as a fix target.
            self._banner("Fixer", extra="(review)")
            default = step_files[0] if len(step_files) == 1 else None
            feedback = (f"The change compiles but a reviewer says it does not correctly implement "
                        f"the step '{step.summary}': {notes}\nAdjust the code to satisfy the step.")
            try:
                (fix_blocks, fix_problems), _ = self.fixer.run(
                    error=feedback, file_blocks=self._file_blocks(step_files),
                    default_path=default, symbols=self._symbol_context(step),
                )
            except RoleError as e:
                console.print(f"[red]Stopped: {e}[/red]")
                return True  # keep what we have; the edits already compile
            apply_errors = self._apply_scoped(fix_blocks, changed, allowed) + fix_problems
        return True

    def _fix_until_ok(self, step, step_files, changed, allowed, apply_errors) -> bool:
        """The check/fix loop: Checker -> (Fixer -> Checker)* up to AGENT_FIX_ROUNDS.

        Returns False only on a hard stop (bad format or the same error twice);
        returns True when checks pass OR the round budget is spent (the edits
        stay applied either way).
        """
        seen_signatures: set = set()
        for round_no in range(self._fix_rounds + 1):
            # A block that didn't apply never reached the file, so the Checker
            # would see unchanged code and pass vacuously. Treat the apply
            # failure itself as the thing to fix, ahead of lint/compile.
            if apply_errors:
                detail = "\n\n".join(apply_errors)
                result = CheckResult(ok=False, signature=_apply_signature(detail), detail=detail)
                console.print(f"[yellow]{len(apply_errors)} edit(s) did not apply.[/yellow]")
            else:
                self._banner("Checker", round_no=round_no)
                result = self.checker.check(step_files)
            if result.ok:
                console.print("[dim]Checks passed.[/dim]")
                return True
            if not apply_errors:
                console.print(f"[yellow]Check failed:[/yellow]\n{result.detail[-1500:]}")

            if result.signature in seen_signatures:
                console.print("[red]The same error came back; stopping this step.[/red]")
                return False
            seen_signatures.add(result.signature)

            if round_no >= self._fix_rounds:
                console.print(f"[red]Still failing after {self._fix_rounds} fix round(s).[/red]")
                return True  # keep the edits; the developer can /undo

            self._banner("Fixer", round_no=round_no + 1)
            default = step_files[0] if len(step_files) == 1 else None
            try:
                (fix_blocks, fix_problems), _ = self.fixer.run(
                    error=result.detail, file_blocks=self._file_blocks(step_files),
                    default_path=default, symbols=self._symbol_context(step),
                )
            except RoleError as e:
                console.print(f"[red]Stopped: {e}[/red]")
                return False
            # Carry any still-failing blocks into the next iteration so the loop
            # keeps fixing the apply failure; an empty list means the Fixer's
            # edits landed and the next round runs the real lint/compile check.
            apply_errors = self._apply_scoped(fix_blocks, changed, allowed) + fix_problems
            step_files[:] = self._touched(step, changed)
        return True

    def _review(self, step: PlanStep, step_files: List[str],
                changed: Dict[str, Optional[str]]):
        """Run the Reviewer. Returns (ok, notes): notes is None when unusable."""
        self._banner("Reviewer")
        diff = self._step_diff(step_files, changed)
        try:
            (ok, notes), _ = self.reviewer.run(step=step, diff=diff)
        except RoleError as e:
            console.print(f"[dim]Reviewer gave no usable verdict ({e}); keeping the edits.[/dim]")
            return True, None
        if ok:
            console.print("[dim]Reviewer: OK[/dim]")
        else:
            console.print(f"[yellow]Reviewer flagged: {notes or '(no note)'}[/yellow]")
        return ok, notes

    # ---- finish: undo snapshot + Summarizer ----

    def _finish(self, request: str, changed: Dict[str, Optional[str]]) -> None:
        self.coder.undo_stack.append(changed)
        self.coder.last_diffs = []
        all_diffs = []
        for rel, old in changed.items():
            new = self._read(rel)
            d = unified_diff(rel, old or "", new)
            if d:
                self.coder.last_diffs.append(d)
                all_diffs.append(d)
        if self.coder.repo_map is not None:
            self.coder._all_files = None

        self._banner("Summarizer")
        combined = "\n".join(all_diffs)
        try:
            summary, _ = self.summarizer.run(request=request, diff=combined)
        except RoleError:
            summary = f"Changed {len(changed)} file(s): {', '.join(changed)}."
        console.print(f"\n[green]{summary}[/green]")
        console.print(f"[dim]Changed {', '.join(changed)} — /undo to revert the whole run.[/dim]")
        self.coder.history.append({"role": "user", "content": f"/micro {request}"})
        self.coder.history.append({"role": "assistant", "content": summary})

    # ---- helpers ----

    def _touched(self, step: PlanStep, changed: Dict[str, Optional[str]]) -> List[str]:
        """Files this step is working on: its declared files plus any actually changed."""
        files = list(dict.fromkeys(list(step.files) + list(changed.keys())))
        return [f for f in files if (self.root / f).exists()]

    def _symbol_context(self, step: PlanStep) -> str:
        """Real symbol names from the step's files, so the model doesn't invent them.

        Lists each editable file's own top-level declarations (class/object/
        interface/fun names) from the repo map. Giving the Editor and Fixer the
        actual names of the types they are wiring together cuts down on invented
        identifiers like a nonexistent 'NewInterceptor'. Empty string if the
        repo map is off or has nothing for these files.
        """
        rm = self.coder.repo_map
        if rm is None:
            return ""
        try:
            rm.build()
        except Exception:
            return ""
        lines = []
        for rel in step.files:
            info = rm.files.get(rel)
            if not info:
                continue
            names = []
            for d in info.get("defs", []):
                # d = [lineno, depth, kind, name, sig]; top-level only, dedup.
                if len(d) >= 4 and d[1] == 0 and d[3] not in names:
                    names.append(d[3])
            if names:
                lines.append(f"{rel}: {', '.join(names[:20])}")
        if not lines:
            return ""
        return ("Real declarations in these files (use these exact names; do not "
                "invent types):\n" + "\n".join(lines))

    def _plausible_new_path(self, rel: str) -> bool:
        """True if `rel` is a sane new file to create, not an invented path.

        Requires: inside the repo, a known source extension, and a parent
        directory that already exists somewhere up the tree (so the model can
        create a file in an existing package, but not conjure a whole new tree).
        """
        try:
            p = (self.root / rel).resolve()
        except (OSError, ValueError):
            return False
        if not p.is_relative_to(self.root):
            return False
        if p.suffix.lower() not in _CREATABLE_EXTS:
            return False
        ancestor = p.parent
        while ancestor != self.root and ancestor.is_relative_to(self.root):
            if ancestor.exists():
                return True
            ancestor = ancestor.parent
        return ancestor == self.root  # at worst, created directly under the repo root

    def _norm_one(self, f: str) -> str:
        """Resolve one path to repo-relative form (falls back to the stripped path)."""
        try:
            return self.coder._rel(f)
        except ValueError:
            return f.strip()

    def _normalize(self, files: List[str]) -> set:
        """Resolve step paths to repo-relative form so scope checks compare equal."""
        return {self._norm_one(f) for f in files}

    def _apply_scoped(self, blocks, changed: Dict[str, Optional[str]], allowed: set) -> List[str]:
        """Apply only blocks whose path is one of the step's files; reject the rest.

        Weaker models sometimes copy the `path/to/file` placeholder from the edit
        format, or invent a path, which would otherwise create a junk file. A
        block aimed anywhere outside the step's declared files is turned into an
        apply error (fed back to the Fixer) instead of being written.
        """
        in_scope, errors = [], []
        for b in blocks:
            raw = b.path.strip()
            try:
                rel = self.coder._rel(raw)
            except ValueError:
                rel = raw
            if rel in allowed:
                in_scope.append(b)
            else:
                errors.append(
                    f"Edit for '{raw}' was rejected: this step may only edit "
                    f"{', '.join(sorted(allowed))}. Use one of those exact paths "
                    "(do not invent a path or copy the example 'path/to/file')."
                )
        return self.coder._apply(in_scope, changed) + errors

    def _is_new_file(self, rel: str) -> bool:
        """True if the file must be created: it is missing or exists but is empty."""
        p = self.root / rel
        if not p.exists():
            return True
        try:
            return p.read_text(encoding="utf-8", errors="replace").strip() == ""
        except OSError:
            return False

    def _file_blocks(self, files: List[str]) -> str:
        return "\n".join(self.coder._file_block(f) for f in files if (self.root / f).exists())

    def _step_diff(self, files: List[str], changed: Dict[str, Optional[str]]) -> str:
        """Unified diff of this step's files against their pre-run contents.

        `changed[rel]` holds each file's content before the first edit of the
        run (None for a newly created file), so this shows what the run has done
        to that file so far -- the right thing for the Reviewer to judge.
        """
        diffs = []
        for rel in files:
            if rel not in changed:
                continue
            d = unified_diff(rel, changed[rel] or "", self._read(rel))
            if d:
                diffs.append(d)
        return "\n".join(diffs)

    def _read(self, rel: str) -> str:
        p = self.root / rel
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
