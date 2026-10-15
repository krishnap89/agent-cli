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
        lines = self._resolve(path).read_text(errors="replace").splitlines()
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
        if not self._confirm(f"write {path} ({len(content)} chars)"):
            return "User denied the write."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {path}"

    def t_edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._resolve(path)
        text = p.read_text()
        count = text.count(old_str)
        if count != 1:
            raise ToolError(f"old_str found {count} times in {path}; it must be unique.")
        if not self._confirm(f"edit {path}"):
            return "User denied the edit."
        p.write_text(text.replace(old_str, new_str))
        return f"Edited {path}"

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
