"""coder: a lightweight Aider-style coding assistant for local OpenAI-compatible LLMs.

Workflow (same idea as Aider):
- You /add the files to work on; their full contents are sent with each request.
- A repo map summarises the rest of the codebase so the model knows what exists.
- The model edits by writing SEARCH/REPLACE blocks in plain text (no tool calling).
- Edits are applied immediately; failed blocks are sent back for the model to fix.
- /undo reverts the last set of edits, /run runs a command and can share its output.
"""
import fnmatch
import glob
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax

from .agent import THINK_RE, show
from .android import (
    detect_android, compile_tasks, run_gradle, parse_kotlin_errors,
    format_kotlin_errors, format_detected_modules, is_offline_dependency_error,
    is_build_config_file, map_file_to_module, module_display_name, run_ktlint,
)
from .checks import builtin_lint, custom_lint, ext_lint_cmd, run_tests, truncate_output
from .config import Config
from .notes import find_notes, gather_init_info, load_notes, notes_exist, INIT_PROMPT
from .skills import load_skills, select_skills, format_skill_context, list_skills_display, Skill
from .review import (
    collect_material, split_material_by_file, build_review_messages,
    build_merge_messages, needs_split, REVIEW_SYSTEM,
)
from .editblock import EditError, HEAD_RE, UPDATED_RE, apply_edit, parse_edit_blocks, unified_diff
from .input_reader import read_message
from .llm import LLMError, chat, list_models
from .repomap import RepoMap
from .testgen import (
    resolve_target, detect_test_setup, classify_class, check_required_libs,
    has_suspend_funs, has_flow_or_viewmodelscope, has_constructor_deps,
    collect_context, find_test_examples, find_test_helpers, test_file_path,
    build_plan_prompt, build_write_prompt, compile_test_task, run_test_task,
    find_junit_xml, parse_junit_xml, format_report, parse_bug_classifications,
    _package_from_file, KIND_RULES,
)

console = Console()

# ---- Prompts -------------------------------------------------------------------------

CODE_SYSTEM = """Act as an expert software developer. The user asks for changes to their code.
If a request is ambiguous, ask a question instead of guessing.

Once you understand the request:
1. Briefly explain the change.
2. Make the change with SEARCH/REPLACE blocks, as described below.

# SEARCH/REPLACE block format
1. The file path alone on a line, exactly as shown in the chat.
2. An opening fence with the language, e.g. ```python
3. The line: <<<<<<< SEARCH
4. The existing lines to find. They must match the file EXACTLY, character for
   character, including indentation, comments and blank lines.
5. The line: =======
6. The new lines that replace them.
7. The line: >>>>>>> REPLACE
8. A closing fence: ```

Rules:
- Keep blocks small: the lines that change plus 1-3 lines around them, enough to
  make the match unique. Never copy a whole long file into SEARCH.
- Only the first match is replaced. Use several blocks for several changes.
- Only edit files that were added to the chat. If you need to see or edit another
  file, name its path, ask the user to add it, and stop.
- To create a new file, use an empty SEARCH section.
- To delete code, use an empty REPLACE section.

# Example
User: In greet.py, make hello() take a name.

Assistant: I'll add a `name` parameter and use it in the message.

greet.py
```python
<<<<<<< SEARCH
def hello():
    print("Hello!")
=======
def hello(name):
    print(f"Hello, {name}!")
>>>>>>> REPLACE
```
"""

ASK_SYSTEM = """Act as an expert software developer. Answer the user's questions about
their code clearly and concisely. Use the files and the repo map you are given.
Do NOT write SEARCH/REPLACE blocks and do not claim to have changed files. If you
need to see a file that is not in the chat, name its path and ask the user to add it.
"""

LANGS = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
         ".jsx": "jsx", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
         ".php": "php", ".cs": "csharp", ".kt": "kotlin", ".c": "c", ".h": "c",
         ".cpp": "cpp", ".swift": "swift", ".sh": "bash", ".md": "markdown",
         ".json": "json", ".yml": "yaml", ".yaml": "yaml", ".html": "html", ".css": "css",
         ".sql": "sql", ".toml": "toml"}

HELP = """Commands:
  /add <files|globs>   add files the model can edit
  /read <files>        add read-only reference files
  /drop [files]        remove files (no args = all)
  /ls                  list files in the chat
  /ask <question>      one question without edits (or /mode ask)
  /code <request>      one request with edits (or /mode code)
  /mode ask|code       switch mode
  /diff                show the last edits
  /undo                undo the last edits
  /run <command>       run a command; optionally share output with the model
  /lint [files]        run lint on files (default: all editable files)
  /test                run AGENT_TEST_CMD and show results
  /autotest on|off     toggle auto-test after edits
  /compile [all]       run Gradle compile check on modules of files in chat
  /modules             list detected modules and file-to-module mapping
  /review [target]     review changes (uncommitted, staged, branch, files)
  /test-gen <class>    generate unit tests for a Kotlin class
  /doc <target>        add KDoc to Kotlin declarations (class, file, folder)
  /skills              list available skills
  /skill <name>        pin a skill (stays active until /skill off)
  /skill off           unpin the current skill
  /notes               show project notes file
  /map [query]         show the repo map
  /tokens              estimate context usage
  /clear               clear chat history (keep files)
  /reset               drop all files and clear history
  /help, /exit
Multi-line: paste directly, or wrap in \"\"\" lines."""


class Coder:
    def __init__(self, cfg: Config, mode: str = "code"):
        self.cfg = cfg
        self.root = cfg.workdir.resolve()
        self.mode = mode
        self.editable: Dict[str, None] = {}   # ordered sets of repo-relative paths
        self.read_only: Dict[str, None] = {}
        self.history: List[dict] = []
        self.undo_stack: List[Dict[str, Optional[str]]] = []
        self.last_diffs: List[str] = []
        self.repo_map = RepoMap(self.root) if cfg.use_map else None
        self._all_files: Optional[List[str]] = None
        self.android_info = detect_android(self.root, cfg.android)
        if self.android_info is not None and not cfg.compile_enabled:
            cfg.compile_enabled = "on"
        self.skills = load_skills(self.root)
        self._pinned_skill: Optional[Skill] = None

    # ---- files -------------------------------------------------------------------

    def _rel(self, path: str) -> str:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"{path} is outside {self.root}")
        return p.relative_to(self.root).as_posix()

    def all_files(self) -> List[str]:
        if self._all_files is None:
            try:
                out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                     cwd=self.root, capture_output=True, text=True, timeout=30)
                files = out.stdout.splitlines() if out.returncode == 0 else []
            except (OSError, subprocess.SubprocessError):
                files = []
            if not files and self.repo_map:
                files = self.repo_map.list_files()
            self._all_files = files
        return self._all_files

    def add(self, patterns: List[str], read_only: bool = False) -> None:
        target = self.read_only if read_only else self.editable
        for pat in patterns:
            matches = sorted(glob.glob(str(self.root / pat), recursive=True))
            matches = [m for m in matches if Path(m).is_file()]
            if not matches:
                if not read_only and not any(c in pat for c in "*?["):
                    if Confirm.ask(f"{pat} doesn't exist. Create it?", default=False):
                        p = self.root / pat
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.touch()
                        matches = [str(p)]
                if not matches:
                    similar = [f for f in self.all_files() if Path(pat).name.lower() in f.lower()][:5]
                    hint = f" Did you mean: {', '.join(similar)}" if similar else ""
                    console.print(f"[red]No file matches {pat}.{hint}[/red]")
                    continue
            for m in matches:
                try:
                    rel = self._rel(m)
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")
                    continue
                self.editable.pop(rel, None)
                self.read_only.pop(rel, None)
                target[rel] = None
                kind = "read-only" if read_only else "editable"
                console.print(f"[dim]Added {rel} ({kind})[/dim]")

    def drop(self, patterns: List[str]) -> None:
        if not patterns:
            self.editable.clear()
            self.read_only.clear()
            console.print("[dim]Dropped all files.[/dim]")
            return
        for pat in patterns:
            gone = [f for f in list(self.editable) + list(self.read_only)
                    if f == pat or fnmatch.fnmatch(f, pat) or f.endswith("/" + pat)]
            for f in gone:
                self.editable.pop(f, None)
                self.read_only.pop(f, None)
                console.print(f"[dim]Dropped {f}[/dim]")
            if not gone:
                console.print(f"[yellow]{pat} is not in the chat.[/yellow]")

    def list_files(self) -> None:
        if not self.editable and not self.read_only:
            console.print("[dim]No files in the chat. Use /add <file>.[/dim]")
        for f in self.editable:
            console.print(f"  {f}")
        for f in self.read_only:
            console.print(f"  {f} [dim](read-only)[/dim]")

    def _file_block(self, rel: str) -> str:
        try:
            text = (self.root / rel).read_text(errors="replace")
        except OSError as e:
            return f"{rel}\n(could not read: {e})\n"
        fence = "```"
        while fence in text:
            fence += "`"
        lang = LANGS.get(Path(rel).suffix.lower(), "")
        return f"{rel}\n{fence}{lang}\n{text}{'' if text.endswith(chr(10)) else chr(10)}{fence}\n"

    # ---- prompt assembly ------------------------------------------------------------

    def _get_active_skills(self, query: str) -> List[Skill]:
        """Return skills to inject: pinned skill, or auto-selected from message."""
        if self._pinned_skill is not None:
            return [self._pinned_skill]
        if self.skills:
            return select_skills(self.skills, query)
        return []

    def _context_messages(self, query: str) -> List[dict]:
        msgs = []
        if self.cfg.notes_enabled:
            _, notes_text = load_notes(self.root, self.cfg.notes_file,
                                       self.cfg.notes_chars, warn_once=False)
            if notes_text:
                msgs += [
                    {"role": "user", "content": "Project notes from the developer. Follow them:\n\n" + notes_text},
                    {"role": "assistant", "content": "Ok, I'll follow these project notes."},
                ]
        active_skills = self._get_active_skills(query)
        for skill in active_skills:
            skill_text = format_skill_context(skill, self.cfg.skill_chars)
            msgs += [
                {"role": "user", "content": "Follow these skill instructions:\n\n" + skill_text},
                {"role": "assistant", "content": f"Ok, I'll follow the {skill.name} skill instructions."},
            ]
            if self._pinned_skill is None:
                console.print(f"[dim]Skill: {skill.name}[/dim]")
        if self.repo_map is not None:
            self.repo_map.build()
            in_chat = set(self.editable) | set(self.read_only)
            q = query + " " + " ".join(Path(f).stem for f in in_chat)
            rmap = self.repo_map.render(query=q, budget=self.cfg.map_chars, exclude=in_chat)
            msgs += [
                {"role": "user", "content": "Here is a map of other files in this repository. "
                 "It is a summary only; ask me to add a file if you need to see or edit it.\n\n" + rmap},
                {"role": "assistant", "content": "Ok, I'll use the map to find relevant code."},
            ]
        if self.read_only:
            body = "\n".join(self._file_block(f) for f in self.read_only)
            msgs += [
                {"role": "user", "content": "Here are READ-ONLY files for reference. Do not edit them.\n\n" + body},
                {"role": "assistant", "content": "Ok, I won't edit those."},
            ]
        if self.editable:
            body = "\n".join(self._file_block(f) for f in self.editable)
            msgs += [
                {"role": "user", "content": "These files are in the chat. Their contents below are "
                 "current; trust them over anything earlier in the conversation.\n\n" + body},
                {"role": "assistant", "content": "Ok, any edits will be based on these exact contents."},
            ]
        else:
            msgs += [
                {"role": "user", "content": "No files have been added to the chat yet."},
                {"role": "assistant", "content": "Ok. If I need files, I'll ask you to add them."},
            ]
        return msgs

    def build_messages(self, query: str) -> List[dict]:
        system = CODE_SYSTEM if self.mode == "code" else ASK_SYSTEM
        context = self._context_messages(query)
        fixed = len(system) + sum(len(m["content"]) for m in context)
        # Drop the oldest history until everything fits the budget.
        history = list(self.history)
        while history and fixed + sum(len(m["content"]) for m in history) > self.cfg.context_chars:
            history.pop(0)
        if len(history) < len(self.history) and not getattr(self, "_warned_trim", False):
            console.print("[dim](older chat history left out to fit the context budget)[/dim]")
            self._warned_trim = True
        if history and history[0]["role"] == "assistant":
            history.pop(0)
        if fixed > self.cfg.context_chars:
            console.print(f"[yellow]Files in the chat are about {fixed // 4} tokens, over the "
                          f"budget of ~{self.cfg.context_chars // 4}. /drop some files or use "
                          "/read for reference-only ones.[/yellow]")
        return [{"role": "system", "content": system}] + context + history

    def token_report(self) -> None:
        msgs = self.build_messages("")
        parts = [("system prompt", len(msgs[0]["content"]))]
        if self.cfg.notes_enabled:
            _, notes_text = load_notes(self.root, self.cfg.notes_file, self.cfg.notes_chars, warn_once=False)
            if notes_text:
                parts.append(("project notes", len(notes_text)))
        active_skills = self._get_active_skills("")
        for skill in active_skills:
            skill_text = format_skill_context(skill, self.cfg.skill_chars)
            parts.append((f"skill: {skill.name}", len(skill_text)))
        if self.repo_map is not None:
            parts.append(("repo map", len(msgs[1]["content"])))
        for f in list(self.editable) + list(self.read_only):
            parts.append((f, len(self._file_block(f))))
        parts.append(("chat history", sum(len(m["content"]) for m in self.history)))
        total = sum(n for _, n in parts)
        for name, n in parts:
            console.print(f"  ~{n // 4:>6} tokens  {name}")
        console.print(f"  ~{total // 4:>6} tokens  TOTAL (budget ~{self.cfg.context_chars // 4})")

    # ---- talking to the model ---------------------------------------------------------

    def _call_model(self, messages: List[dict]) -> Optional[str]:
        try:
            with console.status("[dim]waiting for model...[/dim]") as status:
                def progress(phase, chars):
                    status.update(f"[dim]{phase}... ({chars} chars)[/dim]")

                def retry(err, attempt, wait):
                    console.print(f"[yellow]{err.splitlines()[0][:150]}[/yellow]")
                    console.print(f"[dim]Retrying in {wait:.0f}s (attempt {attempt + 1})...[/dim]")

                reply = chat(self.cfg, messages, on_progress=progress, on_retry=retry)
        except LLMError as e:
            console.print(f"[red]{e}[/red]")
            return None
        text = THINK_RE.sub("", reply.content).strip()
        if reply.finish_reason == "length":
            console.print("[yellow](reply was cut off: raise AGENT_MAX_TOKENS)[/yellow]")
        return text

    def send(self, user_text: str, mode: Optional[str] = None, _resend: bool = False) -> None:
        old_mode = self.mode
        if mode:
            self.mode = mode
        try:
            self._send(user_text, _resend)
        finally:
            self.mode = old_mode

    def _send(self, user_text: str, resend: bool) -> None:
        self.history.append({"role": "user", "content": user_text})
        changed: Dict[str, Optional[str]] = {}
        for attempt in range(self.cfg.max_reflections + 1):
            text = self._call_model(self.build_messages(user_text))
            if text is None:
                self.history.pop()  # failed request: forget it
                return
            if not text:
                console.print("[yellow]The model returned an empty reply.[/yellow]")
                self.history.pop()
                return
            self.history.append({"role": "assistant", "content": text})
            self._display(text)
            if self.mode != "code":
                break
            blocks, problems = parse_edit_blocks(text, default_path=self._single_file())
            if not blocks and not problems:
                break
            errors = self._apply(blocks, changed) + problems
            if not errors:
                break
            if attempt == self.cfg.max_reflections:
                console.print("[red]Some edits still failed; giving up on them.[/red]")
                break
            console.print(f"[yellow]{len(errors)} edit(s) failed, asking the model to fix them...[/yellow]")
            feedback = "\n\n".join(errors)
            feedback += ("\n\nFix the failed SEARCH/REPLACE blocks above. Only resend the blocks "
                         "that failed; the others were already applied. Copy the SEARCH lines "
                         "exactly from the current file contents.")
            self.history.append({"role": "user", "content": feedback})

        if changed:
            self._run_checks_loop(changed, user_text)
            self.undo_stack.append(changed)
            self.last_diffs = []
            for rel, old in changed.items():
                new = (self.root / rel).read_text(errors="replace") if (self.root / rel).exists() else ""
                d = unified_diff(rel, old or "", new)
                if d:
                    self.last_diffs.append(d)
            if self.repo_map is not None:
                self._all_files = None

        if not resend:
            self._offer_mentioned_files(text or "", user_text, applied=bool(changed))

    def _single_file(self) -> Optional[str]:
        return next(iter(self.editable)) if len(self.editable) == 1 else None

    def _apply(self, blocks, changed: Dict[str, Optional[str]]) -> List[str]:
        errors = []
        for b in blocks:
            try:
                rel = self._rel(b.path.strip())
            except ValueError as e:
                errors.append(f"Edit for {b.path} rejected: {e}")
                continue
            path = self.root / rel
            if rel not in self.editable:
                if path.exists():
                    if self.cfg.auto_approve or Confirm.ask(
                            f"The model wants to edit {rel}, which isn't in the chat. Allow?", default=True):
                        self.add([rel])
                    else:
                        errors.append(f"Edit for {rel} rejected: the user did not add it to the chat.")
                        continue
                elif not b.search.strip():
                    if not (self.cfg.auto_approve or Confirm.ask(f"Create new file {rel}?", default=True)):
                        errors.append(f"Creating {rel} was rejected by the user.")
                        continue
                else:
                    errors.append(f"Edit for {rel} failed: that file does not exist.")
                    continue
            old = path.read_text(errors="replace") if path.exists() else None
            try:
                new = apply_edit(old or "", b.search, b.replace)
            except EditError as e:
                errors.append(f"SEARCH/REPLACE block for {rel} failed.\n{e}\n\nThe failed block was:\n"
                              f"<<<<<<< SEARCH\n{b.search}=======\n{b.replace}>>>>>>> REPLACE")
                continue
            changed.setdefault(rel, old)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new)
            if rel not in self.editable:
                self.editable[rel] = None
        return errors

    def _display(self, text: str) -> None:
        """Show prose as markdown and edit blocks as compact syntax-highlighted code."""
        lines, out, i = text.splitlines(), [], 0
        while i < len(lines):
            if HEAD_RE.match(lines[i]):
                if out:
                    self._show_prose(out)
                    out = []
                block = []
                while i < len(lines):
                    block.append(lines[i])
                    if UPDATED_RE.match(lines[i]):
                        break
                    i += 1
                console.print(Syntax("\n".join(block), "diff", theme="ansi_dark", word_wrap=True))
            else:
                out.append(lines[i])
            i += 1
        if out:
            self._show_prose(out)

    def _show_prose(self, lines: List[str]) -> None:
        text = "\n".join(lines)
        text = re.sub(r"(?m)^\s*(```|~~~)[\w+-]*\s*$\n?", "", text)  # fences around blocks
        if text.strip():
            show(text.strip())

    def _offer_mentioned_files(self, reply: str, user_text: str, applied: bool) -> None:
        """If the model names repo files that aren't in the chat, offer to add them."""
        files = set(self.all_files())
        in_chat = set(self.editable) | set(self.read_only)
        by_name: Dict[str, List[str]] = {}
        for f in files:
            by_name.setdefault(Path(f).name, []).append(f)
        found = []
        for tok in re.findall(r"[\w./-]+\.[A-Za-z0-9]+", reply):
            tok = tok.strip("./")
            cand = tok if tok in files else (by_name.get(tok, [None])[0] if len(by_name.get(tok, [])) == 1 else None)
            if cand and cand not in in_chat and cand not in found:
                found.append(cand)
        if not found:
            return
        console.print(f"[cyan]The model mentioned: {', '.join(found[:6])}[/cyan]")
        if Confirm.ask("Add these files to the chat?", default=True):
            self.add(found[:6])
            if not applied and Confirm.ask("Re-send your request with these files?", default=True):
                self.history = self.history[:-2] if len(self.history) >= 2 else []
                self.send(user_text, _resend=True)

    # ---- checks (lint + test) --------------------------------------------------------------

    def _run_checks_loop(self, changed: Dict[str, Optional[str]], user_text: str) -> None:
        """After edits are applied, run lint and optionally tests, feeding errors back."""
        if self.mode != "code" or not self.cfg.lint_enabled:
            console.print(f"[green]Applied edits to {', '.join(changed)}[/green] [dim](/undo to revert)[/dim]")
            return

        changed_files = list(changed.keys())
        for fix_attempt in range(self.cfg.max_fix_attempts + 1):
            # --- lint (built-in + custom + per-ext) ---
            lint_errors = self.run_lint(changed_files)

            # --- ktlint (before compile) ---
            if not lint_errors and self.android_info is not None:
                ktlint_errs = run_ktlint(self.root, changed_files, self.cfg.ktlint_mode)
                if ktlint_errs:
                    lint_errors.extend(ktlint_errs)

            if lint_errors:
                err_text = "\n".join(lint_errors)
                if fix_attempt < self.cfg.max_fix_attempts:
                    console.print(f"[yellow]Lint errors:[/yellow]\n{err_text}")
                    console.print(f"[dim]Asking the model to fix it (attempt {fix_attempt + 1} "
                                  f"of {self.cfg.max_fix_attempts})...[/dim]")
                    if not self._fix_with_model(err_text, "lint", changed, user_text):
                        return
                    continue
                else:
                    console.print(f"[red]Lint errors remain after {self.cfg.max_fix_attempts} "
                                  f"fix attempts:[/red]\n{err_text}")
                    console.print(f"[green]Applied edits to {', '.join(changed)}[/green] "
                                  f"[dim](/undo to revert)[/dim]")
                    return

            console.print("[dim]Lint: passed[/dim]")

            # --- compile check (Gradle) ---
            compile_err = self._run_compile_check(changed_files, set(changed.keys()))
            if compile_err:
                if fix_attempt < self.cfg.max_fix_attempts:
                    console.print(f"[dim]Asking the model to fix it (attempt {fix_attempt + 1} "
                                  f"of {self.cfg.max_fix_attempts})...[/dim]")
                    if not self._fix_with_model(compile_err, "the compile check", changed, user_text):
                        return
                    continue
                else:
                    console.print(f"[red]Compile errors remain after {self.cfg.max_fix_attempts} "
                                  f"fix attempts.[/red]")
                    console.print(f"[green]Applied edits to {', '.join(changed)}[/green] "
                                  f"[dim](/undo to revert)[/dim]")
                    return

            # --- tests ---
            if not self.cfg.auto_test or not self.cfg.test_cmd:
                console.print(f"[green]Applied edits to {', '.join(changed)}[/green] [dim](/undo to revert)[/dim]")
                return

            passed, output = run_tests(self.root, self.cfg.test_cmd, self.cfg.test_timeout)
            if passed:
                console.print(f"[dim]Tests: {self.cfg.test_cmd} ... passed[/dim]")
                console.print(f"[green]Applied edits to {', '.join(changed)}[/green] [dim](/undo to revert)[/dim]")
                return

            truncated = truncate_output(output, self.cfg.check_output_chars)
            if fix_attempt < self.cfg.max_fix_attempts:
                console.print(f"[yellow]Tests failed:[/yellow]\n{truncated[-2000:]}")
                console.print(f"[dim]Asking the model to fix it (attempt {fix_attempt + 1} "
                              f"of {self.cfg.max_fix_attempts})...[/dim]")
                if not self._fix_with_model(truncated, "the tests", changed, user_text):
                    return
                continue
            else:
                console.print(f"[red]Tests still failing after {self.cfg.max_fix_attempts} "
                              f"fix attempts.[/red]")
                console.print(f"[green]Applied edits to {', '.join(changed)}[/green] "
                              f"[dim](/undo to revert)[/dim]")
                return

        console.print(f"[green]Applied edits to {', '.join(changed)}[/green] [dim](/undo to revert)[/dim]")

    def _run_compile_check(
        self,
        changed_files: List[str],
        changed_set: set,
    ) -> Optional[str]:
        """Run Gradle compile check if applicable. Returns error text for model or None."""
        if self.cfg.compile_enabled != "on" or self.android_info is None:
            return None

        modules = self.android_info["modules"]
        tasks, is_build_config = compile_tasks(changed_files, modules, self.cfg.android_variant)
        if not tasks:
            return None

        task_desc = " ".join(tasks)
        console.print(f"[dim]Compile: ./gradlew {task_desc}[/dim]")

        passed, output = run_gradle(
            self.root, tasks, self.cfg.gradle_args, self.cfg.compile_timeout,
        )

        if passed:
            console.print("[dim]Compile: passed[/dim]")
            return None

        if is_offline_dependency_error(output):
            console.print(
                "[yellow]Compile failed due to missing dependencies in offline mode.\n"
                "Run one online build in Android Studio or remove --offline from "
                "AGENT_GRADLE_ARGS.[/yellow]"
            )
            return None

        errors, all_pre_existing = parse_kotlin_errors(
            output, self.root, changed_set,
        )

        if all_pre_existing and errors:
            files_str = ", ".join(sorted(set(e["file"] for e in errors if e["file"])))
            console.print(
                f"[yellow]Compile errors in files you didn't change: {files_str}[/yellow]"
            )
            return None

        if errors:
            err_text = format_kotlin_errors(errors, self.root)
            console.print(f"[yellow]Compile errors:[/yellow]\n{err_text}")
            return err_text

        # No e: lines found; send last 60 lines of output
        lines = output.splitlines()
        tail = "\n".join(lines[-60:])
        console.print(f"[yellow]Compile failed:[/yellow]\n{tail[-2000:]}")
        return tail

    def _fix_with_model(self, error_text: str, what: str,
                        changed: Dict[str, Optional[str]], user_text: str) -> bool:
        """Send check errors to model for fixing. Returns True if model replied with edits."""
        truncated = truncate_output(error_text, self.cfg.check_output_chars)
        feedback = (f"Your edits were applied, but {what} failed:\n\n{truncated}\n\n"
                    "Fix the problem with SEARCH/REPLACE blocks. The files in the chat "
                    "show their current contents, including your previous edits.")
        self.history.append({"role": "user", "content": feedback})
        text = self._call_model(self.build_messages(user_text))
        if text is None:
            return False
        self.history.append({"role": "assistant", "content": text})
        self._display(text)
        blocks, problems = parse_edit_blocks(text, default_path=self._single_file())
        if blocks or problems:
            errors = self._apply(blocks, changed) + problems
            if errors:
                console.print(f"[yellow]{len(errors)} edit(s) failed during fix attempt.[/yellow]")
        return True

    def run_lint(self, files: Optional[List[str]] = None) -> List[str]:
        """Run lint checks on the given files. Returns list of error strings.

        Lookup order per file: AGENT_LINT_CMD_<EXT>, then AGENT_LINT_CMD, then built-in.
        """
        if files is None:
            files = list(self.editable.keys())

        by_ext = {}  # type: Dict[str, List[str]]
        builtin_files = []  # type: List[str]
        global_lint_files = []  # type: List[str]
        for f in files:
            ext = Path(f).suffix.lower()
            per_ext = ext_lint_cmd(ext)
            if per_ext:
                by_ext.setdefault(ext, []).append(f)
            else:
                builtin_files.append(f)
                global_lint_files.append(f)

        errors = builtin_lint(self.root, builtin_files)

        for ext, ext_files in by_ext.items():
            cmd = ext_lint_cmd(ext)
            if cmd:
                err = custom_lint(self.root, ext_files, cmd)
                if err:
                    errors.append(err)

        if self.cfg.lint_cmd and global_lint_files:
            custom_err = custom_lint(self.root, global_lint_files, self.cfg.lint_cmd)
            if custom_err:
                errors.append(custom_err)

        return errors

    def run_test(self) -> Optional[str]:
        """Run test command. Returns error output or None on success."""
        if not self.cfg.test_cmd:
            return None
        passed, output = run_tests(self.root, self.cfg.test_cmd, self.cfg.test_timeout)
        return None if passed else output

    # ---- review --------------------------------------------------------------------------

    def review(self, target: str = "") -> None:
        """Run a code review. Always uses ask mode; restores mode afterwards."""
        try:
            material, files, desc = collect_material(self.root, target)
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]")
            return

        budget = self.cfg.context_chars
        split = needs_split(material, budget)
        mat_size = len(material)
        console.print(f"[dim]Reviewing {desc}: {mat_size:,} chars"
                       f"{', split per file' if split else ''}[/dim]")

        if split:
            review_text = self._review_split(material, desc)
        else:
            review_text = self._review_single(material, desc)

        if review_text is None:
            return

        show(review_text)

        self.history.append({"role": "user", "content": f"Review of {desc}"})
        self.history.append({"role": "assistant", "content": review_text})

        not_in_chat = [f for f in files
                       if f not in self.editable and f not in self.read_only]
        if not_in_chat:
            console.print(f"\n[cyan]Reviewed files not in chat: {', '.join(not_in_chat[:10])}[/cyan]")
            if Confirm.ask("Add them so you can ask for fixes?", default=True):
                self.add(not_in_chat[:10])

    def _review_single(self, material: str, desc: str) -> Optional[str]:
        """Send one review request."""
        messages = build_review_messages(material, desc)
        return self._call_model(messages)

    def _review_split(self, material: str, desc: str) -> Optional[str]:
        """Review per file, then merge."""
        chunks = split_material_by_file(material)
        if not chunks:
            return self._review_single(material, desc)

        findings = []  # type: List[str]
        for i, (fname, chunk) in enumerate(chunks):
            console.print(f"[dim]  [{i + 1}/{len(chunks)}] {fname}[/dim]")
            messages = build_review_messages(chunk, fname)
            text = self._call_model(messages)
            if text is None:
                return None
            findings.append(f"## {fname}\n{text}")

        if len(findings) == 1:
            return findings[0]

        console.print("[dim]  Merging findings...[/dim]")
        merge_msgs = build_merge_messages(findings)
        return self._call_model(merge_msgs)

    # ---- test-gen --------------------------------------------------------------------------

    def test_gen(self, target: str, method: str = "", auto_accept: bool = False) -> None:
        """Generate unit tests for a Kotlin class."""
        if self.android_info is None:
            console.print("[yellow]No Android/Gradle project detected.[/yellow]")
            return
        modules = self.android_info["modules"]

        # Build repo map if needed
        repo_map_files = {}  # type: Dict[str, Any]
        if self.repo_map is not None:
            self.repo_map.build()
            repo_map_files = self.repo_map.files

        # Step 1: Resolve
        try:
            rel, class_name, method, mod = resolve_target(
                self.root, target, method, repo_map_files, modules,
            )
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]")
            return

        mod_info = modules.get(mod, {})
        mod_path = mod_info.get("path", ".")
        console.print(f"[dim]Target: {class_name} in {rel} (module {mod})[/dim]")

        # Step 3: Classify
        kind = classify_class(self.root / rel, class_name)
        if kind == "dao":
            console.print("[yellow]DAO classes need instrumented or Robolectric tests; "
                          "out of scope for /test-gen.[/yellow]")
            return
        if kind == "ui":
            console.print("[yellow]UI components (Activity, Fragment, Composable) need "
                          "UI/instrumented tests; out of scope for JVM unit tests.[/yellow]")
            return
        console.print(f"[dim]Classification: {kind}[/dim]")

        # Step 2: Detect test setup
        setup = detect_test_setup(self.root, mod, modules)
        console.print(f"[dim]Libraries: {', '.join(sorted(setup['libs'])) or '(none)'}[/dim]")

        needs_suspend = has_suspend_funs(self.root / rel) or has_flow_or_viewmodelscope(self.root / rel)
        needs_mock = has_constructor_deps(self.root / rel)
        missing = check_required_libs(setup["libs"], needs_suspend, needs_mock,
                                       setup["dsl"], setup.get("catalog"))
        if missing:
            for msg in missing:
                console.print(f"[red]{msg}[/red]")
            if not self.cfg.testgen_allow_gradle_edit:
                console.print("[dim]Set AGENT_TESTGEN_ALLOW_GRADLE_EDIT=1 to let the model add them.[/dim]")
                return

        # Determine test path
        t_path = test_file_path(self.root, mod_path, rel, class_name)
        existing_test = t_path if (self.root / t_path).is_file() else None
        console.print(f"[dim]Test file: {t_path}[/dim]")

        # Style examples and helpers
        examples = find_test_examples(
            self.root, mod_path, class_name, kind,
            max_examples=self.cfg.testgen_examples,
        )
        helpers = find_test_helpers(self.root, mod_path)

        # Dispatcher setup for ViewModel rule
        dispatcher_setup = "Dispatchers.setMain(StandardTestDispatcher()) in @Before / resetMain in @After"
        for h in helpers:
            if h["kind"] == "dispatcher-rule":
                dispatcher_setup = f"@get:Rule val {h['name'].lower()} = {h['name']}()"
                break
        if kind == "viewmodel" and kind in KIND_RULES:
            KIND_RULES["viewmodel"] = KIND_RULES["viewmodel"].replace("{dispatcher_setup}", dispatcher_setup)

        # Notes
        notes_text = ""
        if self.cfg.notes_enabled:
            from .notes import load_notes
            _, notes_text = load_notes(self.root, self.cfg.notes_file,
                                       self.cfg.notes_chars, warn_once=False)

        # Step 4: Collect context
        ctx = collect_context(
            self.root, rel, class_name, method, kind, setup,
            examples, helpers, notes_text, existing_test,
            repo_map_files, self.cfg.context_chars,
        )

        # Step 5: Plan
        plan_prompt = build_plan_prompt(class_name, method, existing_test is not None)
        plan_messages = [
            {"role": "system", "content": ASK_SYSTEM},
            {"role": "user", "content": ctx + "\n\n" + plan_prompt},
        ]
        console.print("[dim]Generating test plan...[/dim]")
        plan_text = self._call_model(plan_messages)
        if plan_text is None:
            return

        show(plan_text)

        if self.cfg.testgen_plan and not auto_accept:
            console.print("\n[cyan][a]ccept, [c]ancel, or type changes[/cyan]")
            try:
                choice = read_message("[dim]plan › [/dim]").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if choice.lower() in ("c", "cancel"):
                console.print("[dim]Cancelled.[/dim]")
                return
            if choice.lower() not in ("a", "accept", ""):
                # Revision loop
                revision_msg = [
                    {"role": "system", "content": ASK_SYSTEM},
                    {"role": "user", "content": ctx + "\n\n" + plan_prompt},
                    {"role": "assistant", "content": plan_text},
                    {"role": "user", "content": f"Revise the plan: {choice}"},
                ]
                plan_text = self._call_model(revision_msg)
                if plan_text is None:
                    return
                show(plan_text)

        # Step 6: Write tests
        changed = {}  # type: Dict[str, Optional[str]]
        write_prompt = build_write_prompt(setup["libs"], helpers, t_path, plan_text)
        write_system = CODE_SYSTEM

        # Ensure the test file's parent directory exists
        test_full = self.root / t_path
        test_full.parent.mkdir(parents=True, exist_ok=True)

        # Add test path as editable
        if t_path not in self.editable:
            self.editable[t_path] = None

        console.print("[dim]Writing tests...[/dim]")
        self.history.append({"role": "user", "content": f"Generate tests for {class_name}"})

        # Build messages with context + write prompt
        write_messages = [
            {"role": "system", "content": write_system},
            {"role": "user", "content": ctx + "\n\n" + write_prompt},
        ]
        text = self._call_model(write_messages)
        if text is None:
            self.history.pop()
            return

        self.history.append({"role": "assistant", "content": text})
        self._display(text)

        blocks, problems = parse_edit_blocks(text, default_path=t_path)
        if blocks or problems:
            # Restrict edits to the test file only
            filtered_blocks = []
            for b in blocks:
                try:
                    b_rel = self._rel(b.path.strip())
                except ValueError:
                    b_rel = b.path.strip()
                if b_rel != t_path:
                    feedback = f"You may only edit {t_path}. Block for {b_rel} rejected."
                    self.history.append({"role": "user", "content": feedback})
                    continue
                filtered_blocks.append(b)
            errors = self._apply(filtered_blocks, changed) + problems
            if errors:
                console.print(f"[yellow]{len(errors)} edit(s) failed.[/yellow]")

        if not changed:
            console.print("[yellow]No test file was created.[/yellow]")
            self.history.pop()
            self.history.pop()
            return

        # Step 7: Compile
        fix_rounds = 0
        compile_task = compile_test_task(mod, modules, self.cfg.android_variant)
        for fix_round in range(self.cfg.testgen_max_fix + 1):
            console.print(f"[dim]Compile: ./gradlew {compile_task}[/dim]")
            passed, output = run_gradle(
                self.root, [compile_task], self.cfg.gradle_args, self.cfg.compile_timeout,
            )
            if passed:
                console.print("[dim]Compile: passed[/dim]")
                break

            errors, all_pre = parse_kotlin_errors(output, self.root, set(changed.keys()))
            if all_pre and errors:
                files_str = ", ".join(sorted(set(e["file"] for e in errors if e["file"])))
                console.print(f"[yellow]Compile errors in other files: {files_str}[/yellow]")
                break

            if errors:
                # Only send errors in the test file
                test_errors = [e for e in errors if e["file"] == t_path]
                other_errors = [e for e in errors if e["file"] != t_path]
                if other_errors and not test_errors:
                    console.print("[yellow]Compile errors only in files outside the test; stopping.[/yellow]")
                    break
                err_text = format_kotlin_errors(test_errors or errors, self.root)
            else:
                lines = output.splitlines()
                err_text = "\n".join(lines[-60:])

            if fix_round < self.cfg.testgen_max_fix:
                fix_rounds += 1
                console.print(f"[dim]Compile error, fix round {fix_rounds}...[/dim]")
                if not self._fix_with_model(err_text, "compilation", changed, ""):
                    break
            else:
                console.print(f"[red]Compile errors remain after {self.cfg.testgen_max_fix} fix rounds.[/red]")
                break
        else:
            pass  # compile passed on first try

        # Step 8: Run tests
        pkg = _package_from_file(self.root / rel)
        test_fqcn = f"{pkg}.{class_name}Test" if pkg else f"{class_name}Test"
        test_task = run_test_task(mod, modules, test_fqcn, self.cfg.android_variant)
        test_args = f'--tests "{test_fqcn}"'
        console.print(f"[dim]Running: ./gradlew {test_task} {test_args}[/dim]")

        test_passed, test_output = run_gradle(
            self.root,
            [test_task, "--tests", test_fqcn],
            self.cfg.gradle_args,
            self.cfg.test_timeout,
        )

        # Parse results
        xml_path = find_junit_xml(self.root, mod_path, self.cfg.android_variant, test_fqcn)
        test_results = []  # type: List[Dict]
        if xml_path:
            test_results = parse_junit_xml(xml_path)

        total_tests = len(test_results)
        passed_count = sum(1 for r in test_results if r["passed"])
        ignored_count = 0
        suspected_bugs = []  # type: List[str]

        # Handle test failures
        if not test_passed and test_results:
            failures = [r for r in test_results if not r["passed"]]
            if failures:
                failure_text = "\n".join(
                    f"FAILED: {f['name']}\n{f['failure_message']}\n{f['failure_trace'][:500]}"
                    for f in failures
                )
                feedback = (
                    f"These tests failed:\n\n{failure_text}\n\n"
                    "Start your reply with one line per failing test:\n"
                    "<test name>: TEST_BUG or <test name>: CODE_BUG - <reason>\n"
                    "TEST_BUG: fix the test.\n"
                    "CODE_BUG: the production code is wrong. Do NOT change the assertion; "
                    'add // SUSPECTED BUG: <reason> and @Ignore("Suspected bug: <reason>").'
                )
                for remaining_fix in range(max(self.cfg.testgen_max_fix - fix_rounds, 1)):
                    console.print(f"[dim]Test failures, asking model to fix...[/dim]")
                    if not self._fix_with_model(feedback, "tests", changed, ""):
                        break
                    # Re-run tests
                    test_passed, test_output = run_gradle(
                        self.root,
                        [test_task, "--tests", test_fqcn],
                        self.cfg.gradle_args,
                        self.cfg.test_timeout,
                    )
                    if test_passed:
                        if xml_path and xml_path.is_file():
                            test_results = parse_junit_xml(xml_path)
                        break

            # Count ignored (CODE_BUG)
            if (self.root / t_path).is_file():
                test_text = (self.root / t_path).read_text(errors="replace")
                bug_matches = re.findall(
                    r'// SUSPECTED BUG:\s*(.+)', test_text
                )
                ignored_count = len(bug_matches)
                suspected_bugs = bug_matches
                total_tests = len(test_results)
                passed_count = total_tests - ignored_count - sum(
                    1 for r in test_results if not r["passed"]
                )

        # Save undo state
        self.undo_stack.append(changed)
        self.last_diffs = []
        for c_rel, old in changed.items():
            new = (self.root / c_rel).read_text(errors="replace") if (self.root / c_rel).exists() else ""
            d = unified_diff(c_rel, old or "", new)
            if d:
                self.last_diffs.append(d)

        # Step 9: Report
        report = format_report(t_path, total_tests, passed_count, ignored_count,
                                suspected_bugs, fix_rounds)
        console.print(f"\n[green]{report}[/green]")

        self.history.append({"role": "user", "content": f"Test generation report for {class_name}"})
        self.history.append({"role": "assistant", "content": report})

    # ---- doc (KDoc generation) -----------------------------------------------------------

    def doc_gen(self, target: str, update_mode: bool = False) -> None:
        """Generate KDoc for Kotlin declarations."""
        from .docgen import (
            scan_declarations, filter_declarations, detect_kdoc_style,
            build_style_prompt, build_doc_messages, find_outdated_kdoc,
            verify_code_unchanged, verify_kdoc_balanced, resolve_doc_target,
            format_doc_report,
        )

        repo_map_files = {}  # type: Dict[str, Any]
        if self.repo_map is not None:
            self.repo_map.build()
            repo_map_files = self.repo_map.files

        # Resolve target to file(s)
        try:
            kt_files = resolve_doc_target(self.root, target, repo_map_files)
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]")
            return

        if len(kt_files) > 10:
            if not (self.cfg.auto_approve or Confirm.ask(
                    f"{len(kt_files)} Kotlin files found. Continue?", default=True)):
                return

        # Detect style from existing KDoc
        module_path = ""
        if kt_files:
            parts = kt_files[0].split("/")
            if "src" in parts:
                module_path = "/".join(parts[:parts.index("src")])
        style = detect_kdoc_style(self.root, module_path)
        console.print(f"[dim]Style: {'tags' if style['uses_tags'] else 'prose'}, "
                       f"{style['summary_style']}, {style['max_width']} cols[/dim]")

        notes_text = ""
        if self.cfg.notes_enabled:
            _, notes_text = load_notes(self.root, self.cfg.notes_file,
                                       self.cfg.notes_chars, warn_once=False)

        total_documented = 0
        total_skipped_existing = 0
        total_skipped_override = 0
        total_updated = 0
        files_changed = 0
        all_changed = {}  # type: Dict[str, Optional[str]]
        code_verified = True

        for file_idx, rel in enumerate(kt_files):
            path = self.root / rel
            try:
                source = path.read_text(errors="replace")
            except OSError as e:
                console.print(f"[yellow]Cannot read {rel}: {e}[/yellow]")
                continue

            decls = scan_declarations(source)
            all_decls = list(decls)

            # Count skips
            for d in all_decls:
                if d.is_override:
                    total_skipped_override += 1
                elif d.has_kdoc and not update_mode:
                    total_skipped_existing += 1

            # Filter
            to_doc = filter_declarations(
                decls, rel,
                visibility_filter=self.cfg.doc_visibility,
                include_properties=self.cfg.doc_properties,
                update_mode=update_mode,
            )

            # In update mode, also find outdated ones
            if update_mode:
                outdated = find_outdated_kdoc(source, all_decls, style["uses_tags"])
                outdated_names = {d.name for d in outdated}
                to_doc = [d for d in to_doc if not d.has_kdoc or d.name in outdated_names]

            if not to_doc:
                continue

            console.print(f"[dim][{file_idx + 1}/{len(kt_files)}] {rel}: "
                           f"{len(to_doc)} declaration{'s' if len(to_doc) != 1 else ''}[/dim]")

            # Batch processing
            batch_size = self.cfg.doc_per_step
            for batch_start in range(0, len(to_doc), batch_size):
                batch = to_doc[batch_start:batch_start + batch_size]
                batch_update = update_mode and all(d.has_kdoc for d in batch)

                # Save original for guardrail
                current_source = path.read_text(errors="replace")
                if rel not in all_changed:
                    all_changed[rel] = source  # original before any changes

                messages = build_doc_messages(
                    rel, current_source, batch, style,
                    notes_text=notes_text, update_mode=batch_update,
                )
                text = self._call_model(messages)
                if text is None:
                    continue

                # Parse and apply SEARCH/REPLACE blocks
                blocks, problems = parse_edit_blocks(text, default_path=rel)
                if not blocks:
                    continue

                # Apply blocks
                batch_changed = {}  # type: Dict[str, Optional[str]]
                errors = self._apply(blocks, batch_changed)
                if errors:
                    for fix_round in range(self.cfg.doc_max_fix):
                        feedback = "\n\n".join(errors)
                        feedback += ("\n\nFix the failed SEARCH/REPLACE blocks above. Only resend "
                                     "the blocks that failed. Copy the SEARCH lines exactly.")
                        retry_msgs = messages + [
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": feedback},
                        ]
                        text = self._call_model(retry_msgs)
                        if text is None:
                            break
                        blocks, _ = parse_edit_blocks(text, default_path=rel)
                        if not blocks:
                            break
                        errors = self._apply(blocks, batch_changed)
                        if not errors:
                            break

                if not batch_changed:
                    continue

                # Guardrail: verify only comments changed
                new_source = path.read_text(errors="replace")
                diff_line = verify_code_unchanged(current_source, new_source)
                if diff_line is not None:
                    console.print(f"[red]Code change detected at line {diff_line} in {rel}. "
                                   f"Reverting batch.[/red]")
                    path.write_text(current_source)
                    code_verified = False
                    continue

                # Verify KDoc balanced
                unmatched = verify_kdoc_balanced(new_source)
                if unmatched is not None:
                    console.print(f"[red]Unmatched /** at line {unmatched} in {rel}. "
                                   f"Reverting batch.[/red]")
                    path.write_text(current_source)
                    code_verified = False
                    continue

                if batch_update:
                    total_updated += len(batch)
                else:
                    total_documented += len(batch)

            if rel in all_changed:
                new_content = path.read_text(errors="replace")
                if new_content != all_changed[rel]:
                    files_changed += 1

        # One undo snapshot for the whole command
        if all_changed:
            self.undo_stack.append(all_changed)
            self.last_diffs = []
            for rel, old in all_changed.items():
                new = (self.root / rel).read_text(errors="replace") if (self.root / rel).exists() else ""
                d = unified_diff(rel, old or "", new)
                if d:
                    self.last_diffs.append(d)
            if self.repo_map is not None:
                self._all_files = None

        report = format_doc_report(
            total_documented, files_changed, total_skipped_existing,
            total_skipped_override, total_updated, code_verified,
        )
        console.print(f"\n[green]{report}[/green]")

        self.history.append({"role": "user", "content": f"/doc {target}"})
        self.history.append({"role": "assistant", "content": report})

    # ---- commands -------------------------------------------------------------------------

    def undo(self) -> None:
        if not self.undo_stack:
            console.print("[dim]Nothing to undo.[/dim]")
            return
        for rel, old in self.undo_stack.pop().items():
            path = self.root / rel
            if old is None:
                path.unlink(missing_ok=True)
                self.editable.pop(rel, None)
                console.print(f"[dim]Removed new file {rel}[/dim]")
            else:
                path.write_text(old)
                console.print(f"[dim]Restored {rel}[/dim]")
        self.history.append({"role": "user", "content": "I undid your last edits; the files are back "
                             "to how they were before. Don't redo them unless I ask."})
        self.history.append({"role": "assistant", "content": "Ok."})
        self.last_diffs = []

    def show_diff(self) -> None:
        if not self.last_diffs:
            console.print("[dim]No edits in the last request.[/dim]")
        for d in self.last_diffs:
            console.print(Syntax(d, "diff", theme="ansi_dark"))

    def run(self, command: str) -> None:
        failed = True
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=600)
            output = (r.stdout + r.stderr).strip()
            status = f"exit code {r.returncode}"
            failed = r.returncode != 0
        except subprocess.TimeoutExpired:
            output, status = "(timed out after 600s)", "timeout"
        console.print(output[-5000:] if output else "(no output)", markup=False, highlight=False)
        console.print(f"[dim]{status}[/dim]")
        if output and Confirm.ask("Add the output to the chat?", default=failed):
            text = output if len(output) < 8000 else "...(truncated)...\n" + output[-8000:]
            self.history.append({"role": "user", "content": f"I ran `{command}` ({status}):\n```\n{text}\n```"})
            self.history.append({"role": "assistant", "content": "Ok, I've seen the output."})
            console.print("[dim]Added. Now ask, e.g. 'fix the failing test'.[/dim]")


# ---- CLI ---------------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    files: Optional[List[str]] = typer.Argument(None, help="Files to add to the chat."),
    read: Optional[List[str]] = typer.Option(None, "--read", "-r", help="Read-only files."),
    message: Optional[str] = typer.Option(None, "--message", help="Send one message and exit."),
    ask: bool = typer.Option(False, "--ask", help="Start in ask mode (no edits)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmations."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-u"),
    workdir: Path = typer.Option(Path.cwd(), "--dir", "-d"),
    init_notes: bool = typer.Option(False, "--init-notes", help="Generate AGENT.md project notes."),
    review: Optional[str] = typer.Option(None, "--review", help="Review changes and exit. Targets: staged, <branch>, <file>."),
    test_gen: Optional[str] = typer.Option(None, "--test-gen", help="Generate tests for a Kotlin class and exit."),
    doc: Optional[str] = typer.Option(None, "--doc", help="Add KDoc to a target (class, file, folder) and exit."),
    doc_update: bool = typer.Option(False, "--doc-update", help="Update outdated KDoc instead of adding new."),
):
    cfg = Config(workdir=workdir.resolve(), auto_approve=yes)
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url
    if init_notes:
        _do_init_notes(cfg)
        return

    coder = Coder(cfg, mode="ask" if ask else "code")
    if files:
        coder.add(files)
    if read:
        coder.add(read, read_only=True)

    if review is not None:
        coder.review(review)
        return

    if test_gen is not None:
        parts = test_gen.split()
        tg_target = parts[0] if parts else ""
        tg_method = parts[1] if len(parts) > 1 else ""
        coder.test_gen(tg_target, tg_method, auto_accept=yes)
        return

    if doc is not None:
        coder.doc_gen(doc, update_mode=doc_update)
        return

    if message:
        coder.send(message)
        return

    console.print(f"[bold]coder[/bold] · {cfg.model} @ {cfg.base_url} · {cfg.workdir}")
    if coder.android_info is not None:
        mods = format_detected_modules(coder.android_info["modules"])
        console.print(f"[dim]Android project detected (modules: {mods})[/dim]")
    if cfg.notes_enabled:
        notes_path, notes_text = load_notes(cfg.workdir, cfg.notes_file, cfg.notes_chars)
        if notes_path:
            console.print(f"[dim]Project notes: {notes_path.name} ({len(notes_text):,} chars)[/dim]")
    if coder.skills:
        console.print(f"[dim]Skills: {len(coder.skills)} loaded[/dim]")
    if coder.repo_map is not None:
        with console.status("[dim]indexing repo map...[/dim]"):
            total, _ = coder.repo_map.build()
        console.print(f"[dim]Repo map: {total} files. Type /help for commands.[/dim]")
    coder.list_files()
    console.print()

    while True:
        n = len(coder.editable) + len(coder.read_only)
        color = "green" if coder.mode == "code" else "blue"
        prompt = f"[bold {color}]{coder.mode}[/bold {color}][dim] ({n} file{'s' if n != 1 else ''})[/dim] › "
        try:
            text = read_message(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        args = arg.split()
        try:
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                console.print(HELP, markup=False, highlight=False)
            elif cmd == "/add":
                coder.add(args)
            elif cmd == "/read":
                coder.add(args, read_only=True)
            elif cmd == "/drop":
                coder.drop(args)
            elif cmd == "/ls":
                coder.list_files()
            elif cmd == "/mode":
                if arg in ("ask", "code"):
                    coder.mode = arg
                    console.print(f"[dim]Mode: {arg}[/dim]")
                else:
                    console.print("[yellow]Use /mode ask or /mode code[/yellow]")
            elif cmd == "/ask":
                coder.send(arg, mode="ask") if arg else setattr(coder, "mode", "ask")
            elif cmd == "/code":
                coder.send(arg, mode="code") if arg else setattr(coder, "mode", "code")
            elif cmd == "/undo":
                coder.undo()
            elif cmd == "/diff":
                coder.show_diff()
            elif cmd == "/lint":
                lint_files = args if args else None
                errors = coder.run_lint(lint_files)
                if errors:
                    for e in errors:
                        console.print(f"[yellow]{e}[/yellow]")
                    if Confirm.ask("Ask the model to fix them?", default=True):
                        feedback = "\n\n".join(errors)
                        coder.send(f"Fix these lint errors:\n\n{feedback}", mode="code")
                else:
                    console.print("[green]Lint: passed[/green]")
            elif cmd == "/test":
                if not coder.cfg.test_cmd:
                    console.print("[yellow]No test command set. Set AGENT_TEST_CMD.[/yellow]")
                else:
                    console.print(f"[dim]Running: {coder.cfg.test_cmd}[/dim]")
                    err = coder.run_test()
                    if err:
                        truncated = truncate_output(err, coder.cfg.check_output_chars)
                        console.print(truncated[-3000:], markup=False, highlight=False)
                        console.print("[red]Tests failed.[/red]")
                        if Confirm.ask("Ask the model to fix them?", default=True):
                            coder.send(f"Fix these test failures:\n\n{truncated}", mode="code")
                    else:
                        console.print("[green]Tests passed.[/green]")
            elif cmd == "/autotest":
                if arg in ("on", "1"):
                    coder.cfg.auto_test = True
                    console.print("[dim]Auto-test: on[/dim]")
                elif arg in ("off", "0"):
                    coder.cfg.auto_test = False
                    console.print("[dim]Auto-test: off[/dim]")
                else:
                    state = "on" if coder.cfg.auto_test else "off"
                    console.print(f"[dim]Auto-test: {state}. Usage: /autotest on|off[/dim]")
            elif cmd == "/compile":
                if coder.android_info is None:
                    console.print("[yellow]No Android/Gradle project detected.[/yellow]")
                elif coder.cfg.compile_enabled != "on":
                    console.print("[yellow]Compile check is off (AGENT_COMPILE=off).[/yellow]")
                else:
                    modules = coder.android_info["modules"]
                    if arg == "all":
                        all_mod_files = []
                        for mod_name, mod_info in modules.items():
                            mod_path = mod_info["path"]
                            for f in list(coder.editable) + list(coder.read_only):
                                all_mod_files.append(f)
                        compile_files = all_mod_files or list(coder.editable)
                    else:
                        compile_files = list(coder.editable)
                    if not compile_files:
                        console.print("[dim]No files in the chat to compile.[/dim]")
                    else:
                        err = coder._run_compile_check(compile_files, set(compile_files))
                        if err:
                            if Confirm.ask("Ask the model to fix them?", default=True):
                                coder.send(f"Fix these compile errors:\n\n{err}", mode="code")
                        elif err is None:
                            pass  # already printed pass/skip message
            elif cmd == "/modules":
                if coder.android_info is None:
                    console.print("[dim]No Android/Gradle project detected.[/dim]")
                else:
                    modules = coder.android_info["modules"]
                    console.print("[dim]Detected modules:[/dim]")
                    for mod_name in sorted(modules):
                        info = modules[mod_name]
                        kind = "Android" if info["android"] else "Kotlin/JVM"
                        console.print(f"  {module_display_name(mod_name)} ({info['path']}) [{kind}]")
                    in_chat = list(coder.editable) + list(coder.read_only)
                    if in_chat:
                        console.print("[dim]File → module mapping:[/dim]")
                        for f in in_chat:
                            mod = map_file_to_module(f, modules)
                            mod_disp = module_display_name(mod) if mod else "(none)"
                            console.print(f"  {f} → {mod_disp}")
            elif cmd == "/test-gen":
                parts = arg.split()
                tg_target = parts[0] if parts else ""
                tg_method = parts[1] if len(parts) > 1 else ""
                if not tg_target:
                    console.print("[yellow]Usage: /test-gen <class|file> [method][/yellow]")
                else:
                    coder.test_gen(tg_target, tg_method)
            elif cmd == "/doc":
                if not arg:
                    console.print("[yellow]Usage: /doc <class|file|folder> [--update][/yellow]")
                else:
                    doc_update = "--update" in arg
                    doc_target = arg.replace("--update", "").strip()
                    if doc_target:
                        coder.doc_gen(doc_target, update_mode=doc_update)
                    else:
                        console.print("[yellow]Usage: /doc <class|file|folder> [--update][/yellow]")
            elif cmd == "/review":
                coder.review(arg)
            elif cmd == "/skills":
                console.print(list_skills_display(coder.skills))
            elif cmd == "/skill":
                if not arg or arg == "off":
                    coder._pinned_skill = None
                    console.print("[dim]Skill unpinned.[/dim]")
                else:
                    match = [s for s in coder.skills if s.name == arg]
                    if match:
                        coder._pinned_skill = match[0]
                        console.print(f"[dim]Skill pinned: {match[0].name}[/dim]")
                    else:
                        console.print(f"[yellow]No skill named '{arg}'. Use /skills to list.[/yellow]")
            elif cmd == "/notes":
                if cfg.notes_enabled:
                    notes_path, notes_text = load_notes(cfg.workdir, cfg.notes_file, cfg.notes_chars, warn_once=False)
                    if notes_path:
                        console.print(f"[dim]Notes file: {notes_path}[/dim]")
                        console.print(notes_text, markup=False, highlight=False)
                    else:
                        console.print("[dim]No project notes file found. Create one with:\n"
                                      "  coder --init-notes\n"
                                      "Or create AGENT.md in the project root.[/dim]")
                else:
                    console.print("[dim]Project notes disabled (AGENT_NOTES=0).[/dim]")
            elif cmd == "/run":
                coder.run(arg) if arg else console.print("[yellow]Usage: /run <command>[/yellow]")
            elif cmd == "/map":
                if coder.repo_map is None:
                    console.print("[dim]Repo map is off (AGENT_MAP=0).[/dim]")
                else:
                    coder.repo_map.build()
                    console.print(coder.repo_map.render(query=arg, budget=cfg.map_chars),
                                  markup=False, highlight=False)
            elif cmd == "/tokens":
                coder.token_report()
            elif cmd == "/clear":
                coder.history.clear()
                console.print("[dim]History cleared.[/dim]")
            elif cmd == "/reset":
                coder.history.clear()
                coder.drop([])
            elif cmd == "/check":
                console.print(", ".join(list_models(cfg)))
            elif text.startswith("/"):
                console.print(f"[yellow]Unknown command {cmd}. Type /help.[/yellow]")
            else:
                coder.send(text)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        console.print()


def _do_init_notes(cfg: Config) -> None:
    """Generate AGENT.md by asking the model about the project."""
    if notes_exist(cfg.workdir, cfg.notes_file):
        console.print("[yellow]A project notes file already exists. "
                      "Edit it directly or delete it first.[/yellow]")
        return

    from .repomap import RepoMap
    rm = RepoMap(cfg.workdir)
    with console.status("[dim]indexing repo...[/dim]"):
        rm.build()
    map_text = rm.render(budget=cfg.map_chars)
    info = gather_init_info(cfg.workdir, map_text)
    prompt = INIT_PROMPT.format(info=info)

    console.print("[dim]Asking the model to draft project notes...[/dim]")
    from .llm import chat as llm_chat, LLMError
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Write concise, accurate project documentation."},
        {"role": "user", "content": prompt},
    ]
    try:
        with console.status("[dim]waiting for model...[/dim]"):
            reply = llm_chat(cfg, messages)
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        return

    draft = reply.content.strip()
    console.print("\n" + draft + "\n")

    if not Confirm.ask("Save as AGENT.md?", default=True):
        console.print("[dim]Not saved.[/dim]")
        return

    out = cfg.workdir / "AGENT.md"
    out.write_text(draft + "\n")
    console.print(f"[green]Saved {out}[/green]")
    console.print("[dim]Review and edit it — the model drafted it from your code, "
                  "so check that it's accurate.[/dim]")


if __name__ == "__main__":
    app()
