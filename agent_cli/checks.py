"""Automatic checks after edits: built-in lint (syntax) and custom lint/test commands.

Used by coder after applying edits to catch errors before the user sees broken code.
"""
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def builtin_lint(root: Path, files: List[str]) -> List[str]:
    """Run built-in syntax checks on changed files. Returns a list of error strings."""
    errors = []
    for rel in files:
        path = root / rel
        if not path.exists():
            continue
        ext = path.suffix.lower()
        if ext == ".py":
            err = _check_python(path, rel)
            if err:
                errors.append(err)
        elif ext == ".json":
            err = _check_json(path, rel)
            if err:
                errors.append(err)
    return errors


def _check_python(path: Path, rel: str) -> Optional[str]:
    try:
        source = path.read_text(errors="replace")
        compile(source, rel, "exec")
        return None
    except SyntaxError as e:
        lines = source.splitlines()
        lineno = e.lineno or 1
        start = max(lineno - 3, 0)
        end = min(lineno + 2, len(lines))
        context_lines = []
        for i in range(start, end):
            marker = " >> " if i + 1 == lineno else "    "
            context_lines.append(f"{marker}{i + 1:>4} {lines[i]}")
        context = "\n".join(context_lines)
        return f"{rel} line {lineno}: SyntaxError: {e.msg}\n{context}"


def _check_json(path: Path, rel: str) -> Optional[str]:
    try:
        json.loads(path.read_text(errors="replace"))
        return None
    except json.JSONDecodeError as e:
        return f"{rel} line {e.lineno} col {e.colno}: JSONDecodeError: {e.msg}"


def custom_lint(root: Path, files: List[str], cmd: str, timeout: int = 60) -> Optional[str]:
    """Run a custom lint command. Returns error output on failure, None on success."""
    quoted = " ".join(shlex.quote(f) for f in files)
    full_cmd = cmd.replace("{files}", quoted)
    try:
        r = subprocess.run(
            full_cmd, shell=True, cwd=root,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Lint command timed out after {timeout}s: {full_cmd}"
    if r.returncode != 0:
        return (r.stdout + r.stderr).strip()
    return None


def run_tests(root: Path, cmd: str, timeout: int = 600) -> Tuple[bool, str]:
    """Run the test command. Returns (passed, output)."""
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=root,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Test command timed out after {timeout}s: {cmd}"
    output = (r.stdout + r.stderr).strip()
    return r.returncode == 0, output


def truncate_output(text: str, max_chars: int) -> str:
    """Keep the END of the output (where errors/tracebacks are), truncating the start."""
    if len(text) <= max_chars:
        return text
    return "...(truncated)...\n" + text[-max_chars:]
