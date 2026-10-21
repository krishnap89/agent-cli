"""Tab-completion for REPL commands using prompt_toolkit."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, List, Optional

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from .coder import Coder

ALL_COMMANDS = [
    "/add", "/ask", "/autotest", "/check", "/clear", "/code", "/compile",
    "/diff", "/doc", "/drop", "/exit", "/help", "/lint", "/ls", "/map",
    "/mode", "/model", "/modules", "/notes", "/quit", "/read", "/reset", "/review",
    "/run", "/skill", "/skills", "/test", "/test-gen", "/tokens", "/undo",
]

FILE_COMMANDS = {"/add", "/read"}
DROP_COMMAND = "/drop"
TARGET_COMMANDS = {"/doc", "/test-gen"}


class CoderCompleter(Completer):
    def __init__(self, coder: Coder):
        self.coder = coder

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text:
            return

        parts = text.split(None, 1)
        cmd = parts[0] if parts else ""
        has_space = len(text) > len(cmd) and text[len(cmd)] == " "

        if cmd.startswith("/") and not has_space:
            yield from self._complete_command(cmd)
            return

        if not cmd.startswith("/"):
            return

        arg_text = parts[1] if len(parts) > 1 else ""
        tokens = arg_text.split()
        current = tokens[-1] if tokens else ""
        if not arg_text or arg_text.endswith(" "):
            current = ""

        if cmd in FILE_COMMANDS:
            yield from self._complete_files(current)
        elif cmd == DROP_COMMAND:
            yield from self._complete_chat_files(current)
        elif cmd in TARGET_COMMANDS:
            first_arg = tokens[0] if tokens else ""
            if len(tokens) <= 1 and not arg_text.endswith(" "):
                yield from self._complete_targets(current)
            elif cmd == "/test-gen" and len(tokens) <= 2:
                yield from self._complete_methods(first_arg, current)

    def _complete_command(self, prefix: str) -> Iterable[Completion]:
        for cmd in ALL_COMMANDS:
            if cmd.startswith(prefix):
                yield Completion(cmd, start_position=-len(prefix))

    def _complete_files(self, prefix: str) -> Iterable[Completion]:
        try:
            files = self.coder.all_files()
        except Exception:
            return
        yield from _match(files, prefix)

    def _complete_chat_files(self, prefix: str) -> Iterable[Completion]:
        files = list(self.coder.editable) + list(self.coder.read_only)
        yield from _match(files, prefix)

    def _complete_targets(self, prefix: str) -> Iterable[Completion]:
        yield from self._complete_files(prefix)
        yield from self._complete_class_names(prefix)

    def _complete_class_names(self, prefix: str) -> Iterable[Completion]:
        rm = self.coder.repo_map
        if rm is None or not rm.files:
            return
        seen = set()
        for info in rm.files.values():
            for d in info.get("defs", []):
                if d[2] == "class" and d[3] not in seen:
                    seen.add(d[3])
                    name = d[3]
                    if name.lower().startswith(prefix.lower()):
                        yield Completion(name, start_position=-len(prefix))

    def _complete_methods(self, target: str, prefix: str) -> Iterable[Completion]:
        rm = self.coder.repo_map
        if rm is None or not rm.files:
            return
        target_file = self._resolve_target_file(target)
        if not target_file or target_file not in rm.files:
            return
        info = rm.files[target_file]
        seen = set()
        for d in info.get("defs", []):
            if d[2] == "def" and d[3] not in seen:
                seen.add(d[3])
                name = d[3]
                if name.lower().startswith(prefix.lower()):
                    yield Completion(name, start_position=-len(prefix))

    def _resolve_target_file(self, target: str) -> str | None:
        if "/" in target or target.endswith((".kt", ".kts", ".swift", ".py")):
            return target
        rm = self.coder.repo_map
        if rm is None:
            return None
        for rel, info in rm.files.items():
            for d in info.get("defs", []):
                if d[2] == "class" and d[3] == target:
                    return rel
        return None


def _match(candidates: List[str], prefix: str) -> Iterable[Completion]:
    lower = prefix.lower()
    for c in candidates:
        if c.lower().startswith(lower):
            yield Completion(c, start_position=-len(prefix))


# The agent REPL (main.py) has a small, fixed set of commands. (command, hint)
# pairs, kept in the order they're shown in the menu.
AGENT_COMMANDS = [
    ("/chat", "chat only, no tools"),
    ("/agent", "use tools"),
    ("/review", "code review [target]"),
    ("/map", "repo map [query]"),
    ("/notes", "show project notes"),
    ("/skills", "list skills"),
    ("/skill", "pin a skill [name|off]"),
    ("/model", "switch model [name]"),
    ("/grammar", "fix grammar & spelling"),
    ("/rephrase", "rewrite more clearly"),
    ("/tone", "rewrite in a tone <style> <text>"),
    ("/email", "notes → email"),
    ("/slack", "notes → Slack message"),
    ("/clear", "reset history"),
    ("/exit", "quit"),
]


# Characters that make up an @file token: the path plus the leading '@'.
_AT_TOKEN_RE = re.compile(r"@[\w./\\-]*$")


class AgentCompleter(Completer):
    """Completes two things in the agent REPL:

    - a slash command when the whole line is a bare '/cmd' (with a hint beside
      each);
    - a repo file path when the word under the cursor starts with '@', anywhere
      in the message (e.g. "explain @agent_cli/main.py").

    Once a slash command and a space are typed, command completion stays out of
    the way so the argument text is free-form; '@' completion still works there.
    """

    #: read by input_reader to decide whether to pop the menu up without Tab
    complete_while_typing = True

    def __init__(self, files_fn=None):
        # files_fn() -> list[str] of repo-relative paths. Called lazily and the
        # result cached, so we don't shell out to git on every keystroke.
        self._files_fn = files_fn
        self._files = None  # type: Optional[List[str]]

    def _files_list(self) -> List[str]:
        if self._files is None:
            try:
                self._files = list(self._files_fn()) if self._files_fn else []
            except Exception:
                self._files = []
        return self._files

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor

        # @file completion: look at the token ending at the cursor.
        at = _AT_TOKEN_RE.search(text)
        if at:
            prefix = at.group(0)[1:]  # drop the leading '@'
            lower = prefix.lower()
            for path in self._files_list():
                if lower in path.lower():
                    # Replace the '@' + partial path; keep the '@' in the result.
                    yield Completion("@" + path, start_position=-len(at.group(0)))
            return

        # Slash-command completion: only when the whole line is a bare '/cmd'.
        if not text.startswith("/") or " " in text or "\n" in text:
            return
        for cmd, hint in AGENT_COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=hint)
