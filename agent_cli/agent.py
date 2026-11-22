import json
import re
import time

from rich.console import Console
from rich.markdown import Markdown

from .checks import relative_paths
from .config import (ACTION_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, REPO_MAP_SECTION,
                     Config, SYSTEM_PROMPT)
from .llm import ACTION_BLOCK, CUT_OFF_REASONS, XML_CALL, XML_LITERAL, LLMError, chat
from .mathtext import plain_math
from .notes import load_notes
from .repomap import RepoMap
from .term import safe_markup, safe_text
from .tools import TOOL_SCHEMAS, JIRA_TOOL_SCHEMAS, Tools
from .jira import is_configured as is_jira_configured

console = Console(emoji=False)  # :codes: in model text are not emoji

_CALL_RE = re.compile(XML_CALL)
# What to hide from the user: whole calls, and a broken one to the end of the reply
_CALL_MARKUP_RE = re.compile(rf"{XML_CALL}|<tool_call>.*\Z", re.DOTALL)
_CALL_OR_THINK_RE = re.compile(rf"({XML_LITERAL})|<think>.*?(?:</think>|\Z)", re.DOTALL)
# Also strips an unclosed <think> (model ran out of tokens while thinking)
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
# Either an action block (kept) or thinking (dropped), whichever starts first
_BLOCK_OR_THINK_RE = re.compile(rf"({ACTION_BLOCK})|<think>.*?(?:</think>|\Z)", re.DOTALL)
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
    gist = " ".join(plain_math(gist).split())  # collapse whitespace
    if len(gist) > 140:
        gist = gist[:137].rstrip() + "..."
    return gist


CODE_RE = re.compile(r"(```[\s\S]*?(?:```|$)|`[^`\n]*`)")


def show(text: str, math: bool = True) -> None:
    """Print as markdown. rich silently drops HTML-like text (e.g. <div>),
    so escape '<' everywhere except inside code, where it already shows fine.
    math=False is for text that has already been through plain_math."""
    if math:
        text = plain_math(text)
    parts = CODE_RE.split(safe_text(text))
    safe = "".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))
    console.print(Markdown(safe))


# Newer Qwen models (Qwen3-Coder, Qwen3.5+) are trained on an XML tool-call format:
#   <tool_call><function=search><parameter=pattern>Auth</parameter></function></tool_call>
XML_FUNC_RE = re.compile(r"<function=([\w.-]+)>(.*)</function>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)
# Arguments where every space and newline counts
_VERBATIM_ARGS = {"content", "old_str", "new_str", "command"}


def _xml_value(name: str, key: str, value: str):
    """Parameters arrive as text. String arguments stay text; for the others turn
    numbers/booleans/JSON back into values."""
    if key in _VERBATIM_ARGS:
        # only the newline the format puts after and before the tags
        return value.removeprefix("\n").removesuffix("\n")
    v = value.strip()
    if _PARAMS.get(name, {}).get(key) == "string":
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def strip_thinking(text: str, tool_format: str) -> str:
    keep = _BLOCK_OR_THINK_RE if tool_format == "action" else _CALL_OR_THINK_RE
    return keep.sub(lambda m: m.group(1) or "", text).strip()


def response_tag(cfg: Config) -> str:
    """Tag that wraps results sent back to the model."""
    return "action_response" if cfg.tool_format == "action" else "tool_response"


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


def _bare_json_calls(text: str) -> list[tuple[int, int]]:
    """Find bare JSON tool calls a model emitted without the <tool_call> wrapper.

    Models often wrap the call in prose ("I'll map the repo." before, "STOP"
    after) and emit a bare `{"name": ...}` object on its own line. We take
    balanced-brace objects that begin a line and carry a "name" key, so a JSON
    example quoted mid-sentence (not at a line start) is left alone. Returns
    their (start, end) spans; what is inside one is never a call of its own.
    """
    out, done = [], 0
    for m in re.finditer(r"(?m)^\s*(?:```(?:json)?\s*)?\{", text):
        start = text.index("{", m.start())
        if start < done:
            continue
        end = _match_brace(text, start)
        if end == -1:
            if re.match(r'\{\s*"(?:name|arguments)"', text[start:]):
                break  # a call cut short: the lines below are its content
            continue
        if '"name"' in text[start:end + 1]:
            out.append((start, end + 1))
            done = end + 1
    return out


_ESCAPE_HINT = ("If a value contains tags such as tool_call, parameter or tool_response, "
                "send the call as JSON and write each < in it as \\u003c.")


def _function_call(body: str) -> dict:
    """Read a <function=...> call that XML_CALL matched."""
    func = XML_FUNC_RE.search(body)
    name, args = func.group(1), {}
    for key, value in XML_PARAM_RE.findall(func.group(2)):
        if key in args:
            raise ValueError(f"{name} has parameter {key} twice")
        args[key] = _xml_value(name, key, value)
    return {"name": name, "arguments": args}


def _json_call(body: str) -> dict:
    obj = json.loads(body.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    name = obj["name"]
    args = obj.get("arguments", {})
    if isinstance(args, str):  # some models double-encode
        args = json.loads(args)
    return {"name": name, "arguments": args}


def parse_tool_calls(text: str, cut_off: bool = False) -> tuple[list[dict], list[str]]:
    """Extract tool calls from model text. Returns (calls, parse_errors).

    A call's values can be file content that looks like calls, so each call is
    read whole. At the first one that can't be, reading stops. A <function=...>
    call has no quoting, so any doubt about the reply also cancels those before
    it: their end may have been file content. JSON calls before them still run.

    cut_off: the reply ended early."""
    calls, errors = [], []
    first = text.find("<tool_call>")
    # Bare JSON only matters without tags, or when it starts before the first one.
    bare = _bare_json_calls(text) if first == -1 or "{" in text[:first] else []
    if first == -1 or any(start < first < end for start, end in bare):
        if re.search(r"(?m)^\s*<function=[\w.-]+>\s*<parameter=", text):
            return [], ["Nothing was run: a <function=...> call must be inside <tool_call> tags."]
        for start, end in bare:
            try:
                calls.append(_json_call(text[start:end]))
            except (ValueError, KeyError, TypeError, RecursionError) as e:
                errors.append(f"Could not parse tool call ({e}): {text[start:start + 200]}")
        return calls, errors

    problem = ""
    unquoted_from = None  # index in calls of the first <function=...> call
    pos = 0
    while not problem:
        start = text.find("<tool_call>", pos)
        between = text[pos:] if start == -1 else text[pos:start]
        if re.search(r"</(?:tool_call|function|parameter)>", between):
            problem = "found a closing tag outside a tool call"
            break
        if start == -1:
            if cut_off and unquoted_from is not None and between.strip():
                problem = "the reply was cut off after a function call"
            break
        match = _CALL_RE.match(text, start)
        pos = match.end() if match else len(text)
        body = text[start + len("<tool_call>"):pos]
        if match:
            body = body[:-len("</tool_call>")]
        try:
            if body.lstrip().startswith("<function="):
                if unquoted_from is None:
                    unquoted_from = len(calls)
                if not match:
                    raise ValueError("the function call is not closed, or a value in it has "
                                     "call tags. If the reply was cut off, send it again, shorter")
                calls.append(_function_call(body))
            else:
                # Not closed is fine for JSON at the end: half of it never parses.
                calls.append(_json_call(body))
        except (ValueError, KeyError, TypeError, RecursionError) as e:
            problem = f"{e}: {body.strip()[:200]}"
        if not match:
            break
    if problem:
        if "</think>" in text:
            calls.clear()  # thinking may have leaked: any of them may be a draft
        elif unquoted_from is not None:
            del calls[unquoted_from:]
        which = "Nothing from that call on" if calls else "No call in the reply"
        errors.append(f"Could not parse tool call ({problem}). {which} was run. {_ESCAPE_HINT}")
    return calls, errors


# --- ACTION-line tool protocol (cfg.tool_format == "action") -----------------
# For models that return empty content for the XML <tool_call> form but will
# happily write plain prose (e.g. Nemotron). Each tool call is a line:
#   ACTION <tool> key=value key=value ...
# write_file carries the file between <<<CONTENT and CONTENT>>> lines, a
# multi-line run_command sits between <<<COMMAND and COMMAND>>> lines, and
# edit_file instead carries OLD/NEW heredoc blocks after the line, and the
# list-valued tools (apply_patch, todo) carry a single arg=<inline JSON>.
ACTION_LINE_RE = re.compile(r"^\s*ACTION\s+([\w.-]+)\s*(.*)$")
# Each action's arguments and their types, read from the schemas so the parser
# stays in step with the tools.
_PARAMS = {t["name"]: {k: v.get("type") for k, v in t["input_schema"].get("properties", {}).items()}
           for t in TOOL_SCHEMAS + JIRA_TOOL_SCHEMAS}
# A word= that starts the text or follows a space.
_KEY_RE = re.compile(r"(?<!\S)(\w+)=")


def _split_args(name: str, rest: str) -> dict:
    """Split an ACTION line's key=value text at that action's own argument names.

    Any other word= is part of the value (make DESTDIR=/tmp, status=Done in JQL),
    and so is a repeated name. A one-line run_command is always the whole rest
    of the line. Raises ValueError if the text doesn't start with an argument."""
    params = _PARAMS.get(name)
    if params is None or not rest:
        return {}  # nothing to split, or an unknown action that will be refused
    first = _KEY_RE.match(rest)
    if not first or first.group(1) not in params:
        names = ", ".join(f"{k}=" for k in params)
        raise ValueError(f"its arguments must start with one of {names}")
    marks, seen = [], set()
    for m in ([first] if name == "run_command" else _KEY_RE.finditer(rest)):
        if m.group(1) in params and m.group(1) not in seen:
            seen.add(m.group(1))
            marks.append(m)
    args = {}
    for n, m in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(rest)
        key, value = m.group(1), rest[m.end():end].strip()
        args[key] = value if params[key] == "string" else _action_scalar(value)
    return args


CONTENT_OPEN, CONTENT_CLOSE = "<<<CONTENT", "CONTENT>>>"
OLD_OPEN, OLD_CLOSE = "<<<OLD", "OLD"
NEW_OPEN, NEW_CLOSE = "<<<NEW", "NEW"
COMMAND_OPEN, COMMAND_CLOSE = "<<<COMMAND", "COMMAND>>>"
_OPENERS = (CONTENT_OPEN, OLD_OPEN, NEW_OPEN, COMMAND_OPEN)


def _read_block(lines: list[str], i: int, opener: str, closer: str) -> tuple[list[str], int]:
    """Read one block that starts at lines[i], blank lines aside.

    Returns (body, index after the block). Raises ValueError if it isn't closed."""
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines) or lines[i].strip() != opener:
        raise ValueError(f"expected a {opener} line, with no other text before it")
    start = i = i + 1
    # Markers count only at the start of a line, so a file can show them indented.
    while i < len(lines) and lines[i].rstrip() != closer:
        if lines[i].rstrip() == opener:
            raise ValueError(f"found a second {opener} inside the block; "
                             "close each block, and indent markers that are file content")
        i += 1
    if i == len(lines):
        raise ValueError(f"no closing {closer} line at the start of a line. "
                         "If the reply was cut off, send the change in smaller parts")
    return lines[start:i], i + 1


def _check_end(lines: list[str], i: int, opener: str, closer: str) -> None:
    """Raise ValueError if the block that ended before lines[i] may have ended elsewhere."""
    for later in lines[i:]:
        if later.rstrip() == opener:
            return
        if later.rstrip() == closer:
            raise ValueError(f"found a second {closer}, so the end of the block is unclear; "
                             "indent markers that are file content")


def _read_write(rest: str, lines: list[str], i: int) -> tuple[dict, int]:
    path = rest[len("path="):].strip() if rest.startswith("path=") else ""
    if not path or re.search(r"\scontent=|<<<CONTENT", path):
        raise ValueError("the ACTION line must hold only path=<file>; "
                         "the content goes in a block below it")
    body, i = _read_block(lines, i, CONTENT_OPEN, CONTENT_CLOSE)
    _check_end(lines, i, CONTENT_OPEN, CONTENT_CLOSE)
    return {"path": path, "content": "\n".join(body) + "\n" if body else ""}, i


def _read_edit(rest: str, lines: list[str], i: int) -> tuple[dict, int]:
    path = _split_args("edit_file", rest).get("path", "")
    if not path:
        raise ValueError("the ACTION line needs path=<file>")
    old, i = _read_block(lines, i, OLD_OPEN, OLD_CLOSE)
    new, i = _read_block(lines, i, NEW_OPEN, NEW_CLOSE)
    _check_end(lines, i, NEW_OPEN, NEW_CLOSE)
    return {"path": path, "old_str": "\n".join(old), "new_str": "\n".join(new)}, i


def _needs_more_lines(command: str) -> str:
    """Why a one-line shell command needs more lines, or "" if it is whole.

    Running only the first line can destroy data (`cat > f <<EOF` empties f) and
    the lines below would be read as actions. When unsure it says incomplete:
    that costs a retry with a block, the other way round costs a file."""
    closers = []  # what is still open, innermost last
    top_level = []  # the command without quoted and bracketed text, for the keyword check
    escaped = -1  # index of the last backslash-escaped character
    i = 0
    while i < len(command):
        c, two = command[i], command[i:i + 2]
        top, step = closers[-1] if closers else "", 1
        outside = not closers
        if top == "'":
            if c == "'":
                closers.pop()
        elif c == "\\":
            if i + 1 == len(command):
                return "it ends in a backslash"
            step, outside, escaped = 2, False, i + 1
        elif top == "$'":
            if c == "'":
                closers.pop()
        elif top == '"' and c == '"':
            closers.pop()
        elif top == "`" and c == "`":
            closers.pop()
        elif command.startswith("$((", i):
            closers.append("))")
            step = 3
        elif two in ("$(", "${"):
            closers.append(")" if two == "$(" else "}")
            step = 2
        elif c == "`":
            closers.append("`")
        elif top == '"':
            pass
        elif top == "}" and c == "}":
            closers.pop()
        elif top == "))" and two == "))":
            closers.pop()
            step = 2
        elif two == "$'":
            closers.append("$'")
            step = 2
        elif c in "'\"":
            closers.append(c)
        elif top == "}":
            pass
        elif top == "))":
            if c == "(":
                closers.append(")")
        elif c == "#" and (i == 0 or (escaped != i - 1 and command[i - 1] in " \t;&|()")):
            break  # a comment
        elif command.startswith("<<<", i):
            step = 3  # a here-string is one line
        elif two == "<<":
            return "it opens a heredoc"
        elif two == "((":
            closers.append("))")
            step = 2
        elif c == "(":
            closers.append("()")  # a subshell or f(), unlike the ")" of $(...)
        elif c == ")" and top in (")", "()"):
            closers.pop()
            outside = top == "()" and not closers
        seen = outside and (not closers or closers == ["()"])
        top_level.append(command[i:i + step] if seen else "Q")
        i += step
    if closers:
        return "it leaves a quote or bracket open"

    words = re.findall(r"&&|\|[|&]?|[;&()]|[^\s;&|()]+", "".join(top_level))
    depth, starts = 0, True  # starts: this word begins a statement
    for word in words:
        if starts and word in ("if", "for", "while", "until", "case", "select", "{"):
            depth += 1
        elif starts and word in ("fi", "done", "esac", "}"):
            depth -= 1
        starts = (word in (";", "&", "&&", "|", "||", "|&", "(", ")")
                  or (starts and word in ("if", "while", "until", "do", "then", "else", "elif",
                                          "{", "}", "!", "time", "fi", "done", "esac"))
                  or (depth > 0 and word == "in"))
    if depth > 0:
        return "it opens a block that it doesn't close"
    if words and words[-1] in ("&&", "|", "||", "|&"):
        return f"it ends in {words[-1]}"
    return ""


def _read_command(rest: str, lines: list[str], i: int) -> tuple[dict, int]:
    if rest in ("", "command="):
        body, i = _read_block(lines, i, COMMAND_OPEN, COMMAND_CLOSE)
        _check_end(lines, i, COMMAND_OPEN, COMMAND_CLOSE)
        if not "".join(body).strip():
            raise ValueError("the command block is empty")
        return {"command": "\n".join(body)}, i
    args = _split_args("run_command", rest)
    why = _needs_more_lines(args["command"])
    following = next((line.strip() for line in lines[i:] if line.strip()), "")
    if args["command"].startswith(COMMAND_OPEN) or following == COMMAND_OPEN:
        why = "a block goes under a bare ACTION run_command line"
    if why:
        raise ValueError(f"{why}. For a command of several lines write ACTION run_command "
                         f"alone on its line, then {COMMAND_OPEN}, the command and "
                         f"{COMMAND_CLOSE}, each starting a line")
    return args, i


# Actions with blocks. Their readers raise ValueError whenever a block's end is in
# doubt: a wrong guess saves partial code, or runs the file's own text as actions.
_BLOCK_READERS = {"write_file": _read_write, "edit_file": _read_edit,
                  "run_command": _read_command}


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


def parse_action_calls(text: str, cut_off: bool = False) -> tuple[list[dict], list[str]]:
    """Parse ACTION-line tool calls. Returns the same (calls, errors) shape as
    parse_tool_calls, so the rest of the loop is unchanged.

    cut_off: the reply ended early, so its last action is not returned: it may be
    half of what was meant ("rm -rf build" of "rm -rf build/tmp")."""
    calls, errors = [], []
    last_is_call = False  # False when the reply's last action is already an error
    # Not splitlines(): it also breaks on \\f and U+2028, which can be file content.
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        marker = lines[i].rstrip()
        if marker in _OPENERS:
            last_is_call = False
            errors.append(f"{marker} is out of place: blocks go right after their ACTION line. "
                          "Nothing from that line on was run.")
            break
        m = ACTION_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name, rest = m.group(1), m.group(2).strip()
        i += 1
        last_is_call = False

        if name in _BLOCK_READERS:
            try:
                args, i = _BLOCK_READERS[name](rest, lines, i)
            except ValueError as e:
                errors.append(f"{name} was not run: {e}. Nothing after it was run either.")
                break  # what follows may be file content: never run it
            calls.append({"name": name, "arguments": args})
            last_is_call = True
            continue

        if name not in _PARAMS:
            errors.append(f"Unknown action: {name}. Nothing after it was run.")
            break  # it may have had a body, which must not run as actions

        # List-valued tools: a single arg=<inline JSON>.
        if name in ("apply_patch", "todo"):
            raw = rest[rest.find("=") + 1:].strip() if "=" in rest else ""
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"Could not parse JSON arg of {name} ({e}): {raw[:200]}. "
                              "It must be one line of JSON. Nothing after it was run.")
                break  # the lines below may be the rest of the JSON
            key = "edits" if name == "apply_patch" else "action"
            # todo's JSON carries the whole argument object; apply_patch's is the
            # edits list. Accept either an object (used as-is) or a bare value.
            args = value if isinstance(value, dict) else {key: value}
            calls.append({"name": name, "arguments": args})
            last_is_call = True
            continue

        # Simple tools: key=value pairs.
        try:
            args = _split_args(name, rest)
        except ValueError as e:
            errors.append(f"{name} was not run: {e}.")
            continue
        calls.append({"name": name, "arguments": args})
        last_is_call = True
    if cut_off and last_is_call:
        errors.append(f"{calls.pop()['name']} was not run: the reply was cut off, so this last "
                      "action may be incomplete. Send it again, in a shorter reply.")
    return calls, errors


def strip_action_blocks(text: str) -> str:
    """Remove ACTION lines and their CONTENT and OLD/NEW bodies, leaving the
    prose the model wrote for the user."""
    out, lines, i = [], text.replace("\r\n", "\n").split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip() in _OPENERS:
            break
        i += 1
        action = ACTION_LINE_RE.match(line)
        if not action:
            out.append(line)
        elif action.group(1) in _BLOCK_READERS:
            try:
                _, i = _BLOCK_READERS[action.group(1)](action.group(2).strip(), lines, i)
            except ValueError:
                break  # the rest was not run and may be file content
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
        self._map_text = ""  # stays empty until /map: startup never indexes
        self.set_mode(chat_only)

    def set_mode(self, chat_only: bool) -> None:
        """Switch between agent mode (tools) and chat-only mode. Clears history."""
        self.chat_only = chat_only
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]

    def _system_prompt(self) -> str:
        if self.chat_only:
            system = CHAT_SYSTEM_PROMPT
        else:
            template = (ACTION_SYSTEM_PROMPT if self.cfg.tool_format == "action"
                        else SYSTEM_PROMPT)
            system = template.format(
                tool_docs=render_tool_docs(self.cfg),
                repo_map_section=self._repo_map_section(),
            )
            if self.cfg.notes_enabled:
                _, notes_text = load_notes(self.cfg.workdir, self.cfg.notes_file,
                                           self.cfg.notes_chars, warn_once=True)
                if notes_text:
                    system += "\n# Project notes\n" + notes_text + "\n"
        return system

    def _repo_map_section(self) -> str:
        """Changes only on /map, so the system prompt stays the same (good for caching)."""
        return REPO_MAP_SECTION.format(repo_map=self._map_text) if self._map_text else ""

    def load_map(self) -> RepoMap:
        """/map: index the repo and put its map in the system prompt, keeping history."""
        rm = self.tools.ensure_map()
        if self.cfg.use_map and rm.files:
            # never the /map query: the prompt serves the whole session
            self._map_text = rm.render(budget=self.cfg.map_chars)
            self.messages[0]["content"] = self._system_prompt()
        return rm

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
                        console.print(f"[yellow]{safe_markup(err.splitlines()[0][:150])}[/yellow]")
                        console.print(f"[dim]Retrying in {wait:.0f}s (attempt {attempt + 1})...[/dim]")

                    # stop the model from inventing tool results
                    stop = None if self.chat_only else [f"<{response_tag(self.cfg)}>"]
                    reply = chat(self.cfg, self.messages, stop=stop,
                                 on_progress=progress, on_retry=retry,
                                 tool_format="" if self.chat_only else self.cfg.tool_format)
            except LLMError as e:
                console.print(f"[red]{safe_markup(str(e))}[/red]")
                self.messages.pop() if self.messages[-1]["role"] == "user" else None
                return
            text = strip_thinking(reply.content, self.cfg.tool_format)
            if self.cfg.show_thinking:
                gist = thinking_gist(reply.reasoning, reply.content)
                if gist:
                    console.print(f"[dim]\N{THOUGHT BALLOON} {safe_markup(gist)}[/dim]")
            cut_off = reply.finish_reason in CUT_OFF_REASONS
            if self.chat_only:  # no tools: show everything as-is
                calls, errors, visible = [], [], text
            elif self.cfg.tool_format == "action":
                calls, errors = parse_action_calls(text, cut_off)
                # Hide the ACTION lines (and their blocks) from the visible
                # reply, leaving only the prose the model wrote for the user.
                visible = strip_action_blocks(text)
            else:
                calls, errors = parse_tool_calls(text, cut_off)
                visible = _CALL_MARKUP_RE.sub("", text).strip()
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
                    elif self.cfg.tool_format == "action":
                        nudge = (("You only thought about what to do. Now do it: "
                                  if thought_only else "You sent an empty reply. ")
                                 + "Write an ACTION line, or give the final answer.")
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
            if cut_off:
                why = ("the connection closed early" if reply.finish_reason == "incomplete"
                       else "raise AGENT_MAX_TOKENS")
                console.print(f"[yellow](reply was cut off: {why})[/yellow]")

            if not calls and not errors:
                # The model sometimes answers a "create a file" request by printing
                # the file content in a code block instead of calling write_file.
                # Nudge it once to actually write it, then let it answer normally.
                if (not self.chat_only and not file_nudged
                        and _asks_to_create_file(prompt) and _looks_like_file_body(text)):
                    file_nudged = True
                    console.print("[dim](model printed file content instead of writing it; "
                                  "asking it to write the file)[/dim]")
                    if self.cfg.tool_format == "action":
                        how = ("Write it now in exactly this form, with no code fence:\n"
                               f"ACTION write_file path=<path>\n{CONTENT_OPEN}\n"
                               f"<full content>\n{CONTENT_CLOSE}\n"
                               "Do not print the content again outside the block.")
                    else:
                        how = ("Emit a single write_file tool call now, with the full content "
                               "in the tool call's `content` field, in the <tool_call> format. "
                               "Do not print the content again outside the tool call.")
                    self.messages.append({
                        "role": "user",
                        "content": ("You printed the file content but did not create the file. "
                                    "A code block does nothing on its own. " + how),
                    })
                    continue
                self._print_footer(started, tool_count)
                return  # plain answer = done

            results = [self._run_tool(c) for c in calls]
            results += [self._wrap_result(f"Error: {e}") for e in errors]
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
        name, args = str(call["name"]), call["arguments"]
        color = self._TOOL_COLORS.get(name, "cyan")
        console.print(f"[{color}]> {safe_markup(name)}[/{color}] [dim]{safe_markup(json.dumps(args)[:120])}[/dim]")
        try:
            output = self.tools.run(name, args)
        except Exception as e:  # goes back to the model; Ctrl-C still gets through
            message = str(e) or type(e).__name__
            if self.cfg.tool_format == "action":
                # Python 3.10+ puts the class in argument errors: "Tools.t_search() ..."
                message = message.replace("Tools.t_", "")
            output = relative_paths(f"Error: {message}", self.tools.root)
            console.print(f"[red]{safe_markup(output)}[/red]")
        else:
            console.print(f"  [dim]{safe_markup(self._summarize(name, output))}[/dim]")
        return self._wrap_result(f"[{name}]\n{output}")

    def _wrap_result(self, body: str) -> str:
        tag = response_tag(self.cfg)
        return f"<{tag}>\n{body}\n</{tag}>"

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
