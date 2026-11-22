"""/cd for coder and agent: one in-memory session per folder, and a recent list.

The process never changes its working directory: each session has its own Config
whose workdir everything else already uses.
"""
import dataclasses
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, TypeVar

from rich.console import Console

from . import redact
from .config import Config
from .term import safe_markup

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore

console = Console(emoji=False, highlight=False)

MAX_RECENT = 10
# Settings changed by a command that belong to the process, not to one folder
SHARED_SETTINGS = ("model", "auto_test")

S = TypeVar("S")  # a session: anything with a .cfg


def store_path() -> Path:
    return Path.home() / ".config" / "agent-cli" / "folders.json"


def load_recent() -> List[str]:
    """Recently used folders, newest first. A missing or broken file means none."""
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):  # RuntimeError: no home folder
        return []
    if not isinstance(data, list):
        return []
    valid = [p for p in data if isinstance(p, str) and os.path.isabs(p)]
    return list(dict.fromkeys(valid))[:MAX_RECENT]


def save_recent(folder: Path) -> None:
    """Move `folder` to the top of the list. Re-reads the file first, as coder
    and agent in other terminals write to it too."""
    paths = [str(folder)] + [p for p in load_recent() if p != str(folder)]
    try:
        store = store_path()
        store.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(store.parent), prefix=".folders-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(paths[:MAX_RECENT], f, indent=2)
            os.replace(tmp, store)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except (OSError, RuntimeError) as e:
        console.print(f"[dim]Could not save the recent folders: {safe_markup(str(e))}[/dim]")


def real_folder(path: Path) -> Path:
    """The one spelling of a folder. resolve() keeps the typed case on macOS, which
    would give a second session and paths that relative_paths() doesn't shorten."""
    real = Path(path).resolve()
    getpath = getattr(fcntl, "F_GETPATH", None)
    if getpath is None:
        return real
    try:
        fd = os.open(real, os.O_RDONLY)
        try:
            return Path(os.fsdecode(fcntl.fcntl(fd, getpath, bytes(1024)).split(b"\0")[0]))
        finally:
            os.close(fd)
    except OSError:
        return real


def _spellings(arg: str) -> List[str]:
    """The argument as typed, and without shell quotes or escapes (drag and drop)."""
    try:
        words = shlex.split(arg)
    except ValueError:
        words = []
    return list(dict.fromkeys([arg] + (words if len(words) == 1 else [])))


def _warn_parent_rules(folder: Path) -> None:
    """Masking rules are read from the active folder only, so say when a parent has some."""
    for parent in folder.parents:
        for name in (".agentignore", ".agent/redact.json", ".agent/redact-names.txt"):
            rules = parent / name
            if os.path.isfile(rules):
                console.print(f"[yellow]{safe_markup(str(rules))} does not apply in "
                              f"{safe_markup(folder.name)}: rules are read from the active "
                              "folder only.[/yellow]", soft_wrap=True)


class Folders(Generic[S]):
    """The sessions of one REPL and which of them is active."""

    def __init__(self, base: Config, first: S, create: Callable[[Config], S]):
        """`base` is the startup config, untouched by any session."""
        self._base = base
        self._create = create
        self.root = real_folder(base.workdir)
        self.active = first
        self._sessions: Dict[Path, S] = {self.root: first}
        self._unchecked: Dict[Path, List[str]] = {}  # strict-masking queue of inactive folders
        self._listed: List[str] = []  # what the numbers on screen refer to
        if os.path.isdir(self.root):
            save_recent(self.root)

    @property
    def label(self) -> str:
        """The active folder's name, safe inside a rich prompt."""
        return safe_markup(self.root.name or str(self.root)).replace("\n", " ")

    @property
    def short_path(self) -> str:
        """The active folder's path, with ~ for home. Plain text, not markup."""
        try:
            return str(Path("~") / self.root.relative_to(real_folder(Path.home())))
        except (ValueError, OSError, RuntimeError):  # not under home, or no home
            return str(self.root)

    def is_missing(self) -> bool:
        if os.path.isdir(self.active.cfg.workdir):
            return False
        console.print(f"[yellow]{safe_markup(str(self.root))} no longer exists. "
                      "Use /cd to switch folder.[/yellow]", soft_wrap=True)
        return True

    def cd(self, arg: str) -> Optional[S]:
        """Handle /cd. Returns the session switched to, or None if nothing changed."""
        arg = arg.strip()
        if not arg:
            self._show_recent()
            return None
        found = self._find(arg)
        if found is None:
            return None
        if not os.access(found, os.R_OK | os.X_OK):
            console.print(f"[red]Cannot open {safe_markup(found)}: permission denied.[/red]",
                          soft_wrap=True)
            return None
        root = real_folder(Path(found))
        if root == self.root:
            console.print(f"[dim]Already in {safe_markup(str(root))}[/dim]", soft_wrap=True)
            return None

        session = self._sessions.get(root)
        is_new = session is None
        current = self.active.cfg
        shared = {name: getattr(current, name) for name in SHARED_SETTINGS}
        if is_new:
            cfg = dataclasses.replace(self._base, workdir=root, **shared)
            try:
                session = self._create(cfg)
            except OSError as e:
                console.print(f"[red]Cannot open {safe_markup(str(root))}: "
                              f"{safe_markup(str(e))}[/red]", soft_wrap=True)
                return None
        else:
            for name, value in shared.items():
                setattr(session.cfg, name, value)

        # All that can be interrupted comes first, so Ctrl-C can't leave a half switch
        if is_new:
            redact.startup_notice(session.cfg)
            _warn_parent_rules(root)
        state = "new session" if is_new else "session resumed"
        save_recent(root)
        console.print(f"[green]Folder: {safe_markup(str(root))}[/green] [dim]({state})[/dim]",
                      soft_wrap=True)
        self._unchecked[self.root] = redact.swap_pending(self._unchecked.pop(root, []))
        self._sessions[root] = session
        self.root, self.active = root, session
        return session

    def _show_recent(self) -> None:
        gone = "" if os.path.isdir(self.root) else " [yellow](missing)[/yellow]"
        console.print(f"Current: {safe_markup(str(self.root))}{gone}", soft_wrap=True)
        self._listed = load_recent()
        if not self._listed:
            console.print("[dim]No recent folders. Use /cd <path>.[/dim]")
            return
        for number, path in enumerate(self._listed, 1):
            marker = "*" if path == str(self.root) else " "
            note = "" if os.path.isdir(path) else " [yellow](missing)[/yellow]"
            console.print(f"{marker}{number:>3}. {safe_markup(Path(path).name or path)}  "
                          f"[dim]{safe_markup(path)}[/dim]{note}", soft_wrap=True)
        console.print("[dim]Switch with /cd <number>, /cd <name> or /cd <path>.[/dim]")

    def _find(self, arg: str) -> Optional[str]:
        """A number from the list, else a path from the active folder, else a folder name."""
        recent = load_recent()
        numbered = self._listed or recent  # the list moves on every switch; keep to what was shown
        by_number = {str(i): p for i, p in enumerate(numbered, 1)}
        if arg in by_number:
            path = by_number[arg]
            if os.path.isdir(path):
                return path
            console.print(f"[red]{safe_markup(path)} no longer exists.[/red]", soft_wrap=True)
            return None

        spellings = _spellings(arg)
        for text in spellings:
            path = os.path.join(self.root, os.path.expanduser(text))
            if os.path.isdir(path):
                return path

        known = dict.fromkeys(recent + [str(p) for p in self._sessions])
        existing = [p for p in known if os.path.isdir(p)]
        names = [text.rstrip("/") for text in spellings]
        matches = ([p for p in existing if Path(p).name in names]
                   or [p for p in existing if Path(p).name.lower() in [n.lower() for n in names]])
        if len(matches) == 1:
            return matches[0]
        if matches:
            console.print(f"[yellow]Several folders are named {safe_markup(arg)}. Use the path:[/yellow]")
            for path in matches:
                console.print(f"  {safe_markup(path)}", soft_wrap=True)
        else:
            console.print(f"[red]No folder {safe_markup(arg)}.[/red] "
                          "[dim]/cd lists recent folders.[/dim]", soft_wrap=True)
        return None
