import difflib
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from .config import Config
from .repomap import RepoMap

console = Console()

# ---- JSON schemas sent to the model ----------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "list_dir",
        "description": "List files and folders at a path (relative to the working directory).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "repo_map",
        "description": "Ranked outline of the codebase (files, classes, functions, line numbers). "
                       "Give a query like 'tax calculation' to focus on matching code, and/or a "
                       "path to limit it to a folder.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
        },
    },
    {
        "name": "outline",
        "description": "List all classes and functions in one file with line numbers. "
                       "Use it before read_file on large files.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file with line numbers. For large files pass start_line and "
                       "end_line to read only the part you need (max 400 lines per call).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact, unique string in a file with a new string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "search",
        "description": "Find where a class, function or text appears in the code. Plain text by "
                       "default (e.g. 'LoginViewModel' or 'fun login('); set regex=true for a regex. "
                       "Optional path limits it to a folder. Skips build output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "regex": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "dependency_graph",
        "description": "Show what a file depends on and what depends on it. "
                       "Returns incoming and outgoing file dependencies with the shared symbols.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply several edits across one or more files in a single atomic step: "
                       "every edit is checked first, and nothing is written unless all of them "
                       "match. Each edit replaces an exact, unique string (old_str) in a file with "
                       "new_str; set old_str to \"\" to create a new file from new_str. Prefer this "
                       "over repeated edit_file calls when a change touches multiple places.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"},
                        },
                        "required": ["path", "new_str"],
                    },
                },
            },
            "required": ["edits"],
        },
    },
    {
        "name": "todo",
        "description": "Track a multi-step task as a checklist that persists across turns. "
                       "action='set' replaces the whole list from items (each {text, status} with "
                       "status one of pending|in_progress|done); action='list' shows the current "
                       "list. Keep exactly one item in_progress. Use it to plan and show progress "
                       "on tasks with several steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "list"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "status": {"type": "string",
                                       "enum": ["pending", "in_progress", "done"]},
                        },
                        "required": ["text"],
                    },
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the working directory. Returns stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

MAX_OUTPUT = 8_000  # chars returned to the model per tool call
SEARCH_MAX_LINES = 40  # matching lines returned by search
SEARCH_PER_FILE = 5    # matching lines per file
SEARCH_SKIP_DIRS = {".git", "build", ".gradle", ".idea", ".cxx", ".kotlin", "node_modules",
                    "__pycache__", ".venv", "venv", ".agent-cache", "captures", "out", "dist"}


# ---- Implementations --------------------------------------------------------

class ToolError(Exception):
    pass


class Tools:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.workdir.resolve()
        self.repo_map = RepoMap(self.root)
        self._map_built = False
        self.todos: list[dict] = []  # session task list: {text, status}

    _DIFF_MAX_LINES = 60  # lines of a diff shown before a write is confirmed

    def ensure_map(self) -> RepoMap:
        """Build (or refresh from cache) the repo map; cheap after the first time."""
        self.repo_map.build()
        self._map_built = True
        return self.repo_map

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ToolError(f"Path escapes working directory: {path}")
        return p

    def _confirm(self, action: str) -> bool:
        if self.cfg.auto_approve:
            return True
        return Confirm.ask(f"[yellow]Allow:[/yellow] {action}", default=True)

    def _show_diff(self, rel: str, old_text: str, new_text: str) -> None:
        """Print a colored, line-capped unified diff before a write is confirmed,
        so the user sees what will change instead of approving blind."""
        if self.cfg.auto_approve:
            return  # no human to show it to
        if old_text == new_text:
            console.print(f"[dim]{rel}: no change[/dim]")
            return
        label = "new file" if old_text == "" else rel
        diff = difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(),
            fromfile=label, tofile=label, lineterm="", n=2,
        )
        adds = dels = shown = 0
        body = []
        for line in diff:
            if line.startswith(("---", "+++")):
                continue
            if line.startswith("+"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1
            if shown < self._DIFF_MAX_LINES:
                if line.startswith("@@"):
                    body.append(f"[cyan]{line}[/cyan]")
                elif line.startswith("+"):
                    body.append(f"[green]{line}[/green]")
                elif line.startswith("-"):
                    body.append(f"[red]{line}[/red]")
                else:
                    body.append(f"[dim]{line}[/dim]")
                shown += 1
        console.print(f"[yellow]~ {rel}[/yellow] [dim](+{adds} -{dels})[/dim]")
        console.print("\n".join(body))
        if shown >= self._DIFF_MAX_LINES:
            console.print(f"[dim]... diff truncated ({self._DIFF_MAX_LINES} lines shown)[/dim]")

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ToolError(f"Unknown tool: {name}")
        out = fn(**args)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... [truncated {len(out) - MAX_OUTPUT} chars]"
        return out

    def t_list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        entries = sorted(e for e in p.iterdir() if e.name not in skip)
        return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries) or "(empty)"

    def t_read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        lines = self._resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        max_lines = 400
        start = max(int(start_line or 1), 1)
        end = min(int(end_line or start + max_lines - 1), total, start + max_lines - 1)
        out, size = [], 0
        for i in range(start, end + 1):
            row = f"{i:>5} {lines[i - 1]}"
            if size + len(row) > MAX_OUTPUT - 300 and out:
                end = i - 1  # stop at a line boundary, keep room for the hint
                break
            out.append(row)
            size += len(row) + 1
        body = "\n".join(out)
        if start > 1 or end < total:
            body = f"[{path}: lines {start}-{end} of {total}]\n" + body
            if end < total:
                body += f"\n[... {total - end} more lines; use start_line={end + 1} to continue]"
        return body or "(empty file)"

    def t_repo_map(self, query: str = "", path: str = "") -> str:
        return self.ensure_map().render(query=query, budget=self.cfg.map_chars, focus=path)

    def t_outline(self, path: str) -> str:
        rel = self._resolve(path).relative_to(self.root).as_posix()
        return self.ensure_map().outline(rel)

    def t_write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        self._show_diff(path, old, content)
        if not self._confirm(f"write {path} ({len(content)} chars)"):
            return "User denied the write."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {path}"

    def t_edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._resolve(path)
        text = p.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count != 1:
            raise ToolError(f"old_str found {count} times in {path}; it must be unique.")
        self._show_diff(path, text, text.replace(old_str, new_str))
        if not self._confirm(f"edit {path}"):
            return "User denied the edit."
        p.write_text(text.replace(old_str, new_str), encoding="utf-8")
        return f"Edited {path}"

    def t_apply_patch(self, edits: list) -> str:
        if not isinstance(edits, list) or not edits:
            raise ToolError("apply_patch needs a non-empty 'edits' list.")
        # Phase 1: validate every edit against the current (in-memory) file state.
        # Later edits to the same file see earlier ones, so a sequence of edits to
        # one file is checked as it will actually apply.
        pending: dict[Path, str] = {}   # resolved path -> new full text
        originals: dict[Path, str] = {}  # resolved path -> on-disk text (for the diff summary)
        plan = []  # (rel_path, is_new) per edit, for the confirm prompt / summary
        for i, e in enumerate(edits):
            if not isinstance(e, dict):
                raise ToolError(f"edit #{i + 1} is not an object.")
            path, new_str = e.get("path"), e.get("new_str")
            old_str = e.get("old_str", "")
            if not path or new_str is None:
                raise ToolError(f"edit #{i + 1} needs 'path' and 'new_str'.")
            p = self._resolve(path)
            if p not in pending:
                if p.exists():
                    originals[p] = pending[p] = p.read_text(encoding="utf-8")
                else:
                    originals[p] = None
                    pending[p] = None  # not yet created
            is_new = old_str == ""
            if is_new:
                if pending[p] is not None:
                    raise ToolError(f"edit #{i + 1}: {path} already exists; "
                                    "pass the text to replace in old_str instead of \"\".")
                pending[p] = new_str
            else:
                if pending[p] is None:
                    raise ToolError(f"edit #{i + 1}: {path} does not exist.")
                count = pending[p].count(old_str)
                if count != 1:
                    raise ToolError(f"edit #{i + 1}: old_str found {count} times in {path}; "
                                    "it must match exactly once (include surrounding lines "
                                    "to make it unique).")
                pending[p] = pending[p].replace(old_str, new_str)
            plan.append((p.relative_to(self.root).as_posix(), is_new))

        files = {rel for rel, _ in plan}
        summary = ", ".join(sorted(files))
        for p, text in pending.items():
            self._show_diff(p.relative_to(self.root).as_posix(),
                            originals[p] or "", text)
        if not self._confirm(f"apply {len(edits)} edit(s) to {len(files)} file(s): {summary}"):
            return "User denied the patch."
        # Phase 2: all edits validated — write.
        for p, text in pending.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        created = sorted(rel for rel, is_new in plan if is_new)
        changed = sorted(f for f in files if f not in created)
        parts = [f"Applied {len(edits)} edit(s)."]
        if changed:
            parts.append("Changed: " + ", ".join(changed))
        if created:
            parts.append("Created: " + ", ".join(created))
        return " ".join(parts)

    def t_dependency_graph(self, path: str) -> str:
        rel = self._resolve(path).relative_to(self.root).as_posix()
        return self.ensure_map().deps(rel)

    def t_search(self, pattern: str, path: str = ".", regex=False) -> str:
        if isinstance(regex, str):  # small models sometimes send "true"
            regex = regex.strip().lower() in ("true", "1", "yes")
        target = self._resolve(path)
        rel = target.relative_to(self.root).as_posix() or "."
        mode = "-E" if regex else "-F"  # plain text by default: "foo(" must not be a regex error
        in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.root,
                                capture_output=True, text=True).returncode == 0
        if in_git:  # respects .gitignore (build/ etc.), includes untracked files
            cmd = ["git", "grep", "-n", "-I", "--untracked", mode, "-e", pattern, "--", rel]
        else:
            cmd = ["grep", "-rnI", mode] + [f"--exclude-dir={d}" for d in SEARCH_SKIP_DIRS] \
                  + ["-e", pattern, rel]
        try:
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "Search timed out. Narrow it with path (e.g. a module folder)."
        if r.returncode > 1:  # 1 = no matches; 2 (grep) / 128 (git grep) = error
            hint = " Set regex=false to search for plain text." if regex else ""
            return f"Search error: {r.stderr.strip()[:300]}.{hint}"

        hits = []
        for line in r.stdout.splitlines():
            if line.startswith("./"):
                line = line[2:]
            file = line.split(":", 1)[0]
            if not any(part in SEARCH_SKIP_DIRS for part in file.split("/")):
                hits.append(line)
        if not hits:
            where = "the project" if rel == "." else rel
            return (f"No matches for {pattern!r} in {where}. Try a shorter name, another spelling, "
                    "or repo_map(query).")

        by_file = {}
        for line in hits:
            by_file.setdefault(line.split(":", 1)[0], []).append(line)
        out, shown = [], 0
        for file, lines in by_file.items():
            if shown >= SEARCH_MAX_LINES:
                break
            out.append(file)
            for line in lines[:SEARCH_PER_FILE]:
                _, num, text = (line.split(":", 2) + ["", ""])[:3]
                out.append(f"  {num}: {text.strip()[:160]}")
                shown += 1
            if len(lines) > SEARCH_PER_FILE:
                out.append(f"  ... {len(lines) - SEARCH_PER_FILE} more in this file")
        header = f"{len(hits)} matches in {len(by_file)} files"
        if shown < len(hits):
            header += f" (showing {shown}; narrow with path or a more specific pattern)"
        return header + "\n" + "\n".join(out)

    _TODO_STATUSES = {"pending", "in_progress", "done"}

    def _render_todos(self) -> str:
        if not self.todos:
            return "(todo list is empty)"
        marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
        done = sum(1 for t in self.todos if t["status"] == "done")
        lines = [f"Tasks ({done}/{len(self.todos)} done):"]
        lines += [f"  {marks[t['status']]} {t['text']}" for t in self.todos]
        return "\n".join(lines)

    def t_todo(self, action: str, items: list = None) -> str:
        if action == "list":
            return self._render_todos()
        if action != "set":
            raise ToolError("todo action must be 'set' or 'list'.")
        if not isinstance(items, list):
            raise ToolError("todo action='set' needs an 'items' list.")
        new = []
        for i, it in enumerate(items):
            if not isinstance(it, dict) or not it.get("text"):
                raise ToolError(f"todo item #{i + 1} needs a 'text' field.")
            status = it.get("status", "pending")
            if status not in self._TODO_STATUSES:
                raise ToolError(f"todo item #{i + 1} has invalid status {status!r}; "
                                "use pending, in_progress or done.")
            new.append({"text": str(it["text"]), "status": status})
        self.todos = new
        return self._render_todos()

    def t_run_command(self, command: str) -> str:
        if not self._confirm(f"run `{command}`"):
            return "User denied the command."
        try:
            r = subprocess.run(
                command, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "Command timed out after 120s."
        return f"exit code: {r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
