"""Project notes file: load and manage AGENT.md / CONVENTIONS.md / .agent/notes.md.

The notes file describes the project (architecture, conventions, commands) and
is automatically included in every request so the model stops guessing.
"""
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console

console = Console()

CANDIDATES = ["AGENT.md", "CONVENTIONS.md", ".agent/notes.md"]

INIT_PROMPT = """Based on the repository information below, write a concise project notes file
(max ~60 lines) with these sections:

## Overview
A one-paragraph summary of what this project does.

## Layout
Main folders and what's in them (short bullet list).

## Conventions
Coding conventions evident from the code (naming, patterns, style). Only what you can see, don't invent.

## Commands
Build, test, and run commands found in the config files.

## Notes for the assistant
<!-- TODO: The developer should fill this in with any special instructions. -->

---
Repository information:

{info}
"""


def find_notes(workdir: Path, env_path: str = "") -> Optional[Path]:
    """Find the project notes file, respecting priority order."""
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = workdir / p
        return p if p.is_file() else None
    for name in CANDIDATES:
        p = workdir / name
        if p.is_file():
            return p
    return None


def load_notes(workdir: Path, env_path: str = "", max_chars: int = 3000,
               warn_once: bool = True) -> Tuple[Optional[Path], str]:
    """Load the notes file content, truncating if needed.

    Returns (path_or_None, content). Content is empty string if no file found.
    """
    path = find_notes(workdir, env_path)
    if path is None:
        return None, ""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None, ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)..."
        if warn_once:
            console.print(f"[yellow]Project notes file is too long ({len(path.read_text())} chars), "
                          f"truncated to {max_chars}. Consider shortening it.[/yellow]")
    return path, text


def notes_exist(workdir: Path, env_path: str = "") -> bool:
    """Check if any notes file exists."""
    return find_notes(workdir, env_path) is not None


def gather_init_info(workdir: Path, map_text: str = "") -> str:
    """Gather project info for --init-notes: repo map, README, config files."""
    parts = []
    if map_text:
        parts.append(f"### Repo map\n{map_text}")

    for pattern in ["README*", "README.*"]:
        for p in sorted(workdir.glob(pattern)):
            if p.is_file():
                try:
                    text = p.read_text(errors="replace")[:3000]
                    parts.append(f"### {p.name}\n{text}")
                except OSError:
                    pass
                break

    config_files = [
        "pyproject.toml", "package.json", "pom.xml", "build.gradle",
        "go.mod", "Cargo.toml", "Makefile",
    ]
    for name in config_files:
        p = workdir / name
        if p.is_file():
            try:
                text = p.read_text(errors="replace")[:2000]
                parts.append(f"### {name}\n{text}")
            except OSError:
                pass

    return "\n\n".join(parts) if parts else "(no project information found)"
