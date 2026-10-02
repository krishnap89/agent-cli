"""Shared logic for the /review command.

Collects material (git diffs, untracked files, or whole files), splits by
budget, and sends to the model with a dedicated review prompt.
Used by both coder.py and main.py (agent).
"""
import fnmatch
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .checks import truncate_output

REVIEW_SYSTEM = """You are an expert code reviewer. Review ONLY the changes shown.

Checklist — check each, but report only real problems you are confident about:
- Correctness and logic errors
- Edge cases: empty, None/null, zero, large inputs
- Error handling
- Security: injection, secrets in code, unsafe input handling
- Resource leaks and performance problems
- Readability and naming
- Missing or outdated tests

Rules:
- Do NOT invent problems. If nothing is wrong, say "No issues found."
- Do NOT suggest SEARCH/REPLACE blocks or edit the code.
- Review the changed lines; use surrounding context to judge them.

Output format — use EXACTLY this:

1. [high|medium|low] path/to/file.py:42 - Short title
   Problem: one or two sentences.
   Fix: one or two sentences, or a short code snippet.

End with a one-line overall summary.
"""

MERGE_PROMPT = """I reviewed several files separately and got these per-file findings.
Merge them into a single numbered list: remove duplicates, re-number, and keep
the same format. End with a one-line overall summary.

{findings}
"""


def _run_git(root: Path, args: List[str], timeout: int = 30) -> Tuple[int, str]:
    """Run a git command and return (returncode, stdout)."""
    try:
        r = subprocess.run(
            ["git"] + args, cwd=root,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def _is_git_repo(root: Path) -> bool:
    code, _ = _run_git(root, ["rev-parse", "--git-dir"])
    return code == 0


def _is_branch(root: Path, name: str) -> bool:
    code, _ = _run_git(root, ["rev-parse", "--verify", name])
    return code == 0


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def collect_material(
    root: Path,
    target: str = "",
    context_chars: int = 48000,
) -> Tuple[str, List[str], str]:
    """Collect review material based on the target.

    Returns (material_text, list_of_file_paths, description).
    Raises ValueError on errors (not a git repo, unknown target, etc.).
    """
    target = target.strip()

    if not target:
        return _collect_uncommitted(root)
    if target == "staged":
        return _collect_staged(root)

    # Check if it's file paths/globs
    expanded = _expand_file_args(root, target)
    if expanded:
        return _collect_files(root, expanded)

    # Check if it's a branch/ref
    if _is_git_repo(root) and _is_branch(root, target):
        return _collect_branch(root, target)

    raise ValueError(
        f"Unknown review target: {target}\n"
        "Use: /review (uncommitted), /review staged, /review <branch>, "
        "or /review <file> [file]"
    )


def _expand_file_args(root: Path, target: str) -> List[str]:
    """Expand file arguments (space-separated paths/globs). Returns repo-relative paths."""
    parts = target.split()
    result = []  # type: List[str]
    for pat in parts:
        matches = sorted(glob.glob(str(root / pat), recursive=True))
        matches = [m for m in matches if Path(m).is_file() and not _is_binary(Path(m))]
        for m in matches:
            p = Path(m).resolve()
            if p.is_relative_to(root.resolve()):
                rel = p.relative_to(root.resolve()).as_posix()
                if rel not in result:
                    result.append(rel)
    return result


def _collect_uncommitted(root: Path) -> Tuple[str, List[str], str]:
    """Uncommitted changes: git diff HEAD + untracked files."""
    if not _is_git_repo(root):
        raise ValueError("Not a git repository. Use /review <file> to review specific files.")

    parts = []  # type: List[str]
    files = []  # type: List[str]

    # Tracked changes
    code, diff = _run_git(root, ["diff", "-U10", "HEAD"])
    if diff.strip():
        changed = _extract_files_from_diff(diff)
        files.extend(changed)
        parts.append(diff.strip())

    # Untracked files
    code, untracked_out = _run_git(root, [
        "ls-files", "--others", "--exclude-standard"
    ])
    for line in untracked_out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        path = root / line
        if path.is_file() and not _is_binary(path):
            try:
                content = path.read_text(errors="replace")
            except OSError:
                continue
            files.append(line)
            parts.append(f"new file: {line}\n{_numbered(content)}")

    if not parts:
        raise ValueError("No changes to review.")

    material = "\n\n".join(parts)
    desc = f"uncommitted changes ({len(files)} file{'s' if len(files) != 1 else ''})"
    return material, files, desc


def _collect_staged(root: Path) -> Tuple[str, List[str], str]:
    if not _is_git_repo(root):
        raise ValueError("Not a git repository.")
    code, diff = _run_git(root, ["diff", "-U10", "--cached"])
    if not diff.strip():
        raise ValueError("No staged changes to review.")
    files = _extract_files_from_diff(diff)
    desc = f"staged changes ({len(files)} file{'s' if len(files) != 1 else ''})"
    return diff.strip(), files, desc


def _collect_branch(root: Path, branch: str) -> Tuple[str, List[str], str]:
    code, diff = _run_git(root, ["diff", "-U10", f"{branch}...HEAD"])
    if not diff.strip():
        raise ValueError(f"No changes between {branch} and HEAD.")
    files = _extract_files_from_diff(diff)
    desc = f"changes vs {branch} ({len(files)} file{'s' if len(files) != 1 else ''})"
    return diff.strip(), files, desc


def _collect_files(root: Path, file_list: List[str]) -> Tuple[str, List[str], str]:
    """Collect whole file contents with line numbers."""
    parts = []  # type: List[str]
    for rel in file_list:
        path = root / rel
        try:
            content = path.read_text(errors="replace")
        except OSError as e:
            parts.append(f"{rel}: (could not read: {e})")
            continue
        parts.append(f"{rel}\n{_numbered(content)}")
    desc = f"{len(file_list)} file{'s' if len(file_list) != 1 else ''}"
    return "\n\n".join(parts), file_list, desc


def _numbered(text: str) -> str:
    """Add line numbers to text."""
    lines = text.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{width}} {line}" for i, line in enumerate(lines))


def _extract_files_from_diff(diff: str) -> List[str]:
    """Extract file paths from a unified diff."""
    files = []  # type: List[str]
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            f = line[6:]
            if f not in files:
                files.append(f)
    return files


def split_material_by_file(material: str) -> List[Tuple[str, str]]:
    """Split a unified diff into per-file chunks. Returns [(filename, chunk)]."""
    chunks = []  # type: List[Tuple[str, str]]
    current_file = None  # type: Optional[str]
    current_lines = []  # type: List[str]

    for line in material.splitlines():
        if line.startswith("diff --git "):
            if current_file is not None:
                chunks.append((current_file, "\n".join(current_lines)))
            current_lines = [line]
            current_file = None
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            current_lines.append(line)
        elif line.startswith("new file: "):
            if current_file is not None:
                chunks.append((current_file, "\n".join(current_lines)))
            current_file = line[len("new file: "):]
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_file is not None:
        chunks.append((current_file, "\n".join(current_lines)))

    return chunks


def build_review_messages(
    material: str,
    desc: str,
) -> List[dict]:
    """Build messages for a single review request."""
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": f"Review these changes ({desc}):\n\n{material}"},
    ]


def build_merge_messages(per_file_findings: List[str]) -> List[dict]:
    """Build messages for merging per-file review findings."""
    combined = "\n\n---\n\n".join(per_file_findings)
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": MERGE_PROMPT.format(findings=combined)},
    ]


def needs_split(material: str, context_chars: int) -> bool:
    """Check if material needs per-file splitting (exceeds 50% of budget)."""
    return len(material) > context_chars // 2
