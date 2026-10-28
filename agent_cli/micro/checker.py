"""Checker: the one role with no model behind it.

It reuses the existing checks (built-in syntax lint, any custom/per-ext lint
command, and on Android projects the Gradle compile check) and returns a
`CheckResult`. The orchestrator feeds a failing result's text back to the
Fixer. Kept as a plain callable-friendly class so tests can inject a fake in
its place.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..android import (
    compile_tasks, run_gradle, parse_kotlin_errors, format_kotlin_errors,
    is_offline_dependency_error, run_ktlint,
)
from ..checks import truncate_output
from ..config import Config


@dataclass
class CheckResult:
    ok: bool
    # A short, stable signature of the failure, used to detect the same error
    # repeating (so the loop can stop). Empty when ok.
    signature: str = ""
    # Full human/model-readable error text. Empty when ok.
    detail: str = ""


class Checker:
    """Run lint + Gradle compile on the changed files. No LLM."""

    def __init__(self, cfg: Config, root: Path, android_info: Optional[dict], run_lint):
        self.cfg = cfg
        self.root = root
        self.android_info = android_info
        # Coder.run_lint(files) -> List[str]; reused as-is so built-in + custom
        # + per-extension lint all behave exactly like the normal edit flow.
        self._run_lint = run_lint

    def check(self, files: List[str]) -> CheckResult:
        if not self.cfg.lint_enabled:
            return CheckResult(ok=True)

        lint_errors = self._run_lint(files)
        if not lint_errors and self.android_info is not None:
            lint_errors.extend(run_ktlint(self.root, files, self.cfg.ktlint_mode))
        if lint_errors:
            text = "\n".join(lint_errors)
            return CheckResult(ok=False, signature=_signature("lint", text),
                               detail=truncate_output(text, self.cfg.check_output_chars))

        compile_detail = self._compile(files)
        if compile_detail is not None:
            return CheckResult(ok=False, signature=_signature("compile", compile_detail),
                               detail=truncate_output(compile_detail, self.cfg.check_output_chars))

        return CheckResult(ok=True)

    def _compile(self, files: List[str]) -> Optional[str]:
        """Run the Gradle compile check. Returns error text, or None if it passed/skipped."""
        if self.cfg.compile_enabled != "on" or self.android_info is None:
            return None
        modules = self.android_info["modules"]
        tasks, _ = compile_tasks(files, modules, self.cfg.android_variant)
        if not tasks:
            return None
        passed, output = run_gradle(self.root, tasks, self.cfg.gradle_args, self.cfg.compile_timeout)
        if passed:
            return None
        if is_offline_dependency_error(output):
            # Not the edit's fault; don't loop on it.
            return None
        errors, all_pre_existing = parse_kotlin_errors(output, self.root, set(files))
        if all_pre_existing and errors:
            return None  # errors are all in files we didn't change
        if errors:
            return format_kotlin_errors(errors, self.root)
        return "\n".join(output.splitlines()[-60:])


def _signature(kind: str, text: str) -> str:
    """A compact, order-insensitive fingerprint of a failure.

    Used only to notice the SAME error coming back twice so the loop can stop;
    it deliberately ignores line numbers and surrounding whitespace, which jitter
    between otherwise-identical failures.
    """
    import re
    norm = re.sub(r"\d+", "#", text)          # drop line/column numbers
    norm = re.sub(r"\s+", " ", norm).strip()
    return f"{kind}:{norm[:400]}"
