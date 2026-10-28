import json
import re
import time

from rich.console import Console
from rich.markdown import Markdown

from .config import (ACTION_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, REPO_MAP_SECTION,
                     Config, SYSTEM_PROMPT)
from .llm import LLMError, chat
from .notes import load_notes
from .tools import TOOL_SCHEMAS, JIRA_TOOL_SCHEMAS, ToolError, Tools
from .jira import is_configured as is_jira_configured

console = Console()

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
# Also strips an unclosed <think> (model ran out of tokens while thinking)
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
# A line that is nothing but a lead-in to the action ("So let me read that
# file.", "I'll now call the tool.") carries no information on its own, so when
# we pick the gist we skip past these to the last substantive line.
_THINK_LEADIN_RE = re.compile(
    r"^(?:so|now|ok(?:ay)?|alright|well|let me|let's|i'?ll|i will|i'?m going to|"
    r"i should|i need to|i want to|first|next|then)\b",
    re.IGNORECASE,
)


def thinking_gist(reasoning: str, content: str) -> str:
    """One dim line summarizing what the model was thinking, or '' if none.

    Reasoning can arrive split out in `reasoning` (server field) or inline as
    <think>...</think> in `content`. We show just enough to convey the agent's
    train of thought without dumping the whole scratchpad into the terminal."""
    text = reasoning.strip()
    if not text:
        inline = THINK_RE.search(content)
        text = inline.group(1).strip() if inline else ""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    # The conclusion is usually near the end, but models often close with a
    # content-free lead-in ("So I'll read that file now."). Walk back from the
    # end to the last line that says something, keeping the lead-in only if
    # every line is one.
    gist = lines[-1]
    for ln in reversed(lines):
        if not _THINK_LEADIN_RE.match(ln):
            gist = ln
            break
    gist = " ".join(gist.split())  # collapse whitespace
    if len(gist) > 140:
        gist = gist[:137].rstrip() + "..."
    return gist


CODE_RE = re.compile(r"(```[\s\S]*?(?:```|$)|`[^`\n]*`)")


def show(text: str) -> None:
    """Print as markdown. rich silently drops HTML-like text (e.g. <div>),
    so escape '<' everywhere except inside code, where it already shows fine."""
    parts = CODE_RE.split(text)
    safe = "".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))
    console.print(Markdown(safe))


# Newer Qwen models (Qwen3-Coder, Qwen3.5+) are trained on an XML tool-call format:
#   <tool_call><function=search><parameter=pattern>Auth</parameter></function></tool_call>
XML_FUNC_RE = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


def _xml_value(value: str):
    """Parameters arrive as text; turn numbers/booleans/JSON back into values."""
    v = value.strip()
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def render_tool_docs(cfg=None) -> str:
    """Turn the JSON schemas into prompt text the model can read.

    The read-only Jira tools are appended only when Jira is configured, so the
    model isn't told about tools it can't use."""
    schemas = list(TOOL_SCHEMAS)
    if cfg is not None and is_jira_configured(cfg):
        schemas += JIRA_TOOL_SCHEMAS
    docs = []
    for t in schemas:
        props = t["input_schema"].get("properties", {})
        required = set(t["input_schema"].get("required", []))
        params = ", ".join(
            f'{k}: {v["type"]}{"" if k in required else " (optional)"}'
            for k, v in props.items()
        )
        docs.append(f'- {t["name"]}({params}): {t["description"]}')
    return "\n".join(docs)


def _match_brace(s: str, start: int) -> int:
    """Index of the '}' matching the '{' at `start`, or -1 (string-aware)."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _bare_json_calls(text: str) -> list[str]:
    """Find bare JSON tool calls a model emitted without the <tool_call> wrapper.

    Models often wrap the call in prose ("I'll map the repo." before, "STOP"
    after) and emit a bare `{"name": ...}` object on its own line. We take
    balanced-brace objects that begin a line and carry a "name" key, so a JSON
    example quoted mid-sentence (not at a line start) is left alone.
    """
    out = []  # type: list[str]
    for m in re.finditer(r"(?m)^\s*(?:```(?:json)?\s*)?\{", text):
        start = text.index("{", m.start())
        end = _match_brace(text, start)
        if end == -1:
            continue
        obj = text[start:end + 1]
        if '"name"' in obj:
            out.append(obj)
    return out


def parse_tool_calls(text: str) -> tuple[list[dict], list[str]]:
    """Extract tool calls from model text. Returns (calls, parse_errors)."""
    calls, errors = [], []
    blocks = TOOL_CALL_RE.findall(text)
    if not blocks:
        blocks = _bare_json_calls(text)
    for raw in blocks:
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        func = XML_FUNC_RE.search(raw)
        if func and not raw.startswith("{"):
            args = {k: _xml_value(v) for k, v in XML_PARAM_RE.findall(func.group(2))}
            calls.append({"name": func.group(1), "arguments": args})
            continue
        try:
            obj = json.loads(raw)
            name = obj["name"]
            args = obj.get("arguments", {})
            if isinstance(args, str):  # some models double-encode
                args = json.loads(args)
            calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append(f"Could not parse tool call ({e}): {raw[:200]}")
    return calls, errors


# --- ACTION-line tool protocol (cfg.tool_format == "action") -----------------
# For models that return empty content for the XML <tool_call> form but will
# happily write plain prose (e.g. Nemotron). Each tool call is a line:
#   ACTION <tool> key=value key=value ...
# edit_file instead carries OLD/NEW heredoc blocks after the line, and the
# list-valued tools (apply_patch, todo) carry a single arg=<inline JSON>.
ACTION_LINE_RE = re.compile(r"^\s*ACTION\s+([\w.-]+)\s*(.*)$")
# key=value pairs on an ACTION line: a value runs up to the next " key=" or EOL,
# so values may contain spaces without quoting.
_KV_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
# Numeric args that should become ints, read from the schemas so the parser
# stays in step with the tools.
_INT_ARGS = {"start_line", "end_line"}


def _action_scalar(value: str):
    """Turn an ACTION value string into an int/bool/number where it clearly is
    one, else leave it as the raw (stripped) string."""
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    return v


def parse_action_calls(text: str) -> tuple[list[dict], list[str]]:
    """Parse ACTION-line tool calls. Returns the same (calls, errors) shape as
    parse_tool_calls, so the rest of the loop is unchanged."""
    calls, errors = [], []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = ACTION_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name, rest = m.group(1), m.group(2).strip()
        i += 1

        # edit_file: path on the ACTION line, then <<<OLD..OLD and <<<NEW..NEW.
        if name == "edit_file":
            path = (dict(_KV_RE.findall(rest)).get("path") or "").strip()
            old, new, ok = [], [], True
            for tag, bucket in (("OLD", old), ("NEW", new)):
                while i < len(lines) and not lines[i].strip().startswith(f"<<<{tag}"):
                    if ACTION_LINE_RE.match(lines[i]):  # ran into the next call
                        break
                    i += 1
                if i >= len(lines) or not lines[i].strip().startswith(f"<<<{tag}"):
                    errors.append(f"edit_file for {path or '?'} is missing its <<<{tag} block.")
                    ok = False
                    break
                i += 1  # past the <<<TAG line
                while i < len(lines) and lines[i].strip() != tag:
                    bucket.append(lines[i])
                    i += 1
                i += 1  # past the closing TAG line
            if ok:
                calls.append({"name": "edit_file", "arguments": {
                    "path": path, "old_str": "\n".join(old), "new_str": "\n".join(new)}})
            continue

        # List-valued tools: a single arg=<inline JSON>.
        if name in ("apply_patch", "todo"):
            raw = rest[rest.find("=") + 1:].strip() if "=" in rest else ""
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"Could not parse JSON arg of {name} ({e}): {raw[:200]}")
                continue
            key = "edits" if name == "apply_patch" else "action"
            # todo's JSON carries the whole argument object; apply_patch's is the
            # edits list. Accept either an object (used as-is) or a bare value.
            args = value if isinstance(value, dict) else {key: value}
            calls.append({"name": name, "arguments": args})
            continue

        # Simple tools: key=value pairs.
        args = {}
        for k, v in _KV_RE.findall(rest):
            args[k] = int(v) if k in _INT_ARGS and re.fullmatch(r"-?\d+", v.strip()) \
                else _action_scalar(v)
        calls.append({"name": name, "arguments": args})
    return calls, errors


def strip_action_blocks(text: str) -> str:
    """Remove ACTION lines and their OLD/NEW heredoc bodies, leaving the prose
    the model wrote for the user."""
    out, lines, i = [], text.splitlines(), 0
    in_heredoc = None  # "OLD" or "NEW" while inside a block
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if in_heredoc:
            if stripped == in_heredoc:
                in_heredoc = None
            i += 1
            continue
        if ACTION_LINE_RE.match(line):
            i += 1
            continue
        if stripped.startswith("<<<OLD"):
            in_heredoc = "OLD"
            i += 1
            continue
        if stripped.startswith("<<<NEW"):
            in_heredoc = "NEW"
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip()


_CREATE_FILE_RE = re.compile(
    r"\b(creat\w*|generat\w*|writ\w*|mak\w*|add|save)\b.{0,40}?"
    r"\b(file|readme|\.\w{1,6}\b|\w+\.(md|py|txt|json|ya?ml|html|css|js|ts|sh|toml|cfg|ini))",
    re.IGNORECASE | re.DOTALL,
)


def _asks_to_create_file(prompt: str) -> bool:
    """Heuristic: did the user ask to create/generate/write a file?"""
    return bool(_CREATE_FILE_RE.search(prompt))


def _looks_like_file_body(text: str) -> bool:
    """Heuristic: the reply is (or contains) file content rather than a short answer.

    A fenced code block, or a reply that opens with a document/code marker.
    """
    if "```" in text:
        return True
    head = text.lstrip()[:40]
    return head.startswith(("# ", "<!DOCTYPE", "<html", "<?xml", "{", "import ", "from ", "#!"))


class Agent:
    def __init__(self, cfg: Config, chat_only: bool = False):
        self.cfg = cfg
        self.tools = Tools(cfg)
        self.set_mode(chat_only)

    def set_mode(self, chat_only: bool) -> None:
        """Switch between agent mode (tools) and chat-only mode. Clears history."""
        self.chat_only = chat_only
        if chat_only:
            system = CHAT_SYSTEM_PROMPT
        else:
            template = (ACTION_SYSTEM_PROMPT if self.cfg.tool_format == "action"
                        else SYSTEM_PROMPT)
            system = template.format(
                workdir=self.cfg.workdir,
                tool_docs=render_tool_docs(self.cfg),
                repo_map_section=self._repo_map_section(),
            )
            if self.cfg.notes_enabled:
                _, notes_text = load_notes(self.cfg.workdir, self.cfg.notes_file,
                                           self.cfg.notes_chars, warn_once=True)
                if notes_text:
                    system += "\n# Project notes\n" + notes_text + "\n"
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def _repo_map_section(self) -> str:
        """Built once per session so the system prompt stays the same (good for caching)."""
        if not self.cfg.use_map:
            return ""
        if not hasattr(self, "_map_text"):
            with console.status("[dim]indexing codebase for repo map...[/dim]"):
                rm = self.tools.ensure_map()
                self._map_text = rm.render(budget=self.cfg.map_chars)
                console.print(f"[dim]Repo map: {len(rm.files)} files indexed.[/dim]")
        return REPO_MAP_SECTION.format(repo_map=self._map_text)

    @property
    def mode_name(self) -> str:
        return "chat" if self.chat_only else "agent"

    def reset(self) -> None:
        self.messages = self.messages[:1]

    def ask(self, prompt: str) -> None:
        self.messages.append({"role": "user", "content": prompt})

        nudged = False
        file_nudged = False
        started = time.monotonic()
        tool_count = 0
        for _ in range(self.cfg.max_turns):
            try:
                with console.status("[dim]waiting for model...[/dim]") as status:
                    def progress(phase: str, chars: int) -> None:
                        status.update(f"[dim]{phase}... ({chars} chars)[/dim]")

                    def retry(err: str, attempt: int, wait: float) -> None:
                        console.print(f"[yellow]{err.splitlines()[0][:150]}[/yellow]")
                        console.print(f"[dim]Retrying in {wait:.0f}s (attempt {attempt + 1})...[/dim]")

                    # stop the model from inventing tool results
                    stop = None if self.chat_only else ["<tool_response>"]
                    reply = chat(self.cfg, self.messages, stop=stop,
                                 on_progress=progress, on_retry=retry)
            except LLMError as e:
                console.print(f"[red]{e}[/red]")
                self.messages.pop() if self.messages[-1]["role"] == "user" else None
                return
            text = THINK_RE.sub("", reply.content).strip()
            if self.cfg.show_thinking:
                gist = thinking_gist(reply.reasoning, reply.content)
                if gist:
                    console.print(f"[dim]\N{THOUGHT BALLOON} {gist}[/dim]")
            if self.chat_only:  # no tools: show everything as-is
                calls, errors, visible = [], [], text
            elif self.cfg.tool_format == "action":
                calls, errors = parse_action_calls(text)
                # Hide the ACTION lines (and any OLD/NEW blocks) from the visible
                # reply, leaving only the prose the model wrote for the user.
                visible = strip_action_blocks(text)
            else:
                calls, errors = parse_tool_calls(text)
                visible = TOOL_CALL_RE.sub("", text).strip()
                if reply.tool_calls:
                    # The server's own tool parser took the call out of the text.
                    # Use it, and write it back as text so the history stays in
                    # the format the system prompt teaches.
                    native, native_errors = self._from_native(reply.tool_calls)
                    calls += native
                    errors += native_errors
                    text = (text + "\n" + "\n".join(
                        "<tool_call>\n" + json.dumps(c) + "\n</tool_call>" for c in native)).strip()

            if not text and not errors:
                # Empty reply: explain why instead of returning silently.
                if reply.finish_reason == "length":
                    console.print(
                        f"[yellow]Model ran out of tokens (max_tokens={self.cfg.max_tokens}) "
                        "before answering, probably while thinking. "
                        "Raise AGENT_MAX_TOKENS.[/yellow]"
                    )
                    return
                thought_only = bool(reply.reasoning.strip())
                if not nudged:
                    nudged = True
                    console.print("[dim](empty reply, asking the model again)[/dim]")
                    self.messages.append({"role": "assistant", "content": "(no reply)"})
                    if self.chat_only:
                        nudge = "You sent an empty reply. Please answer the question."
                    elif thought_only:
                        nudge = ("You only thought about what to do. Now do it: write the tool "
                                 "call in the <tool_call> format, or give the final answer.")
                    else:
                        nudge = "You sent an empty reply. Either make a tool call or answer the question."
                    self.messages.append({"role": "user", "content": nudge})
                    continue
                console.print("[yellow]Model returned an empty reply twice.[/yellow]")
                if thought_only and not self.cfg.keep_special_tokens:
                    console.print("[yellow]It only produced thinking text: the server is probably "
                                  "stripping the tool-call markup. Unset AGENT_KEEP_SPECIAL_TOKENS "
                                  "(or set it to 1).[/yellow]")
                elif thought_only:
                    console.print("[yellow]It only produced thinking text. Try AGENT_THINKING=0, "
                                  "or AGENT_STREAM=0 AGENT_DEBUG=1 to see the raw response.[/yellow]")
                else:
                    console.print("[yellow]Try rephrasing, or run with AGENT_DEBUG=1 to see the "
                                  "raw response.[/yellow]")
                return

            self.messages.append({"role": "assistant", "content": text})
            if visible:
                show(visible)

            if not calls and not errors:
                if reply.finish_reason == "length":
                    console.print("[yellow](reply was cut off: raise AGENT_MAX_TOKENS)[/yellow]")
                # The model sometimes answers a "create a file" request by printing
                # the file content in a code block instead of calling write_file.
                # Nudge it once to actually write it, then let it answer normally.
                if (not self.chat_only and not file_nudged
                        and _asks_to_create_file(prompt) and _looks_like_file_body(text)):
                    file_nudged = True
                    console.print("[dim](model printed file content instead of writing it; "
                                  "asking it to call write_file)[/dim]")
                    self.messages.append({
                        "role": "user",
                        "content": ("You printed the file content but did not create the file. "
                                    "A code block does nothing on its own. Emit a single "
                                    "write_file tool call now, with the full content in the "
                                    "tool call's `content` field, in the <tool_call> format. "
                                    "Do not print the content again outside the tool call."),
                    })
                    continue
                self._print_footer(started, tool_count)
                return  # plain answer = done

            results = [self._run_tool(c) for c in calls]
            results += [f"<tool_response>\nError: {e}\n</tool_response>" for e in errors]
            tool_count += len(calls)
            self.messages.append({"role": "user", "content": "\n".join(results)})

        console.print("[red]Stopped: hit max_turns.[/red]")
        self._print_footer(started, tool_count)

    @staticmethod
    def _print_footer(started: float, tool_count: int) -> None:
        """Close a turn with how long it took and how much work it did."""
        elapsed = time.monotonic() - started
        took = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
        if tool_count:
            console.print(f"[dim]done in {took}, {tool_count} tool call"
                          f"{'' if tool_count == 1 else 's'}[/dim]")
        else:
            console.print(f"[dim]done in {took}[/dim]")

    @staticmethod
    def _from_native(tool_calls: list) -> tuple[list[dict], list[str]]:
        """Convert native tool calls ({name, arguments as JSON string or dict})."""
        calls, errors = [], []
        for tc in tool_calls:
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as e:
                    errors.append(f"Could not parse arguments of {tc.get('name')} ({e}): {args[:200]}")
                    continue
            if not tc.get("name"):
                errors.append("A tool call had no tool name.")
                continue
            calls.append({"name": tc["name"], "arguments": args})
        return calls, errors

    # Color tool calls by what they do, so the eye parses the stream faster:
    # reads/search are safe (cyan), writes mutate files (yellow), commands run
    # arbitrary shell (magenta).
    _TOOL_COLORS = {
        "write_file": "yellow", "edit_file": "yellow", "apply_patch": "yellow",
        "run_command": "magenta", "todo": "blue",
    }

    def _run_tool(self, call: dict) -> str:
        name, args = call["name"], call["arguments"]
        color = self._TOOL_COLORS.get(name, "cyan")
        console.print(f"[{color}]> {name}[/{color}] [dim]{json.dumps(args)[:120]}[/dim]")
        try:
            output = self.tools.run(name, args)
        except (ToolError, OSError, TypeError) as e:
            output = f"Error: {e}"
            console.print(f"[red]{output}[/red]")
        else:
            console.print(f"  [dim]{self._summarize(name, output)}[/dim]")
        return f"<tool_response>\n[{name}]\n{output}\n</tool_response>"

    @staticmethod
    def _summarize(name: str, output: str) -> str:
        """One-line gist of a tool result, so the user can follow the agent's
        work instead of watching silent tool calls scroll past."""
        text = output.strip()
        if not text:
            return "(no output)"
        first = text.splitlines()[0].strip()
        lines = text.count("\n") + 1
        if name in ("search", "repo_map", "dependency_graph", "outline"):
            # these already lead with a count/header line
            return first[:160]
        if name == "read_file":
            return f"{lines} lines"
        if name == "list_dir":
            return f"{lines} entr{'y' if lines == 1 else 'ies'}"
        if name in ("write_file", "edit_file", "apply_patch", "todo", "run_command"):
            return first[:160]
        # default: first line, plus a hint if there's more
        return first[:160] + (f" (+{lines - 1} more lines)" if lines > 1 else "")
