import json
import re

from rich.console import Console
from rich.markdown import Markdown

from .config import CHAT_SYSTEM_PROMPT, REPO_MAP_SECTION, Config, SYSTEM_PROMPT
from .llm import LLMError, chat
from .notes import load_notes
from .tools import TOOL_SCHEMAS, ToolError, Tools

console = Console()

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
# Also strips an unclosed <think> (model ran out of tokens while thinking)
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


CODE_RE = re.compile(r"(```[\s\S]*?(?:```|$)|`[^`\n]*`)")


def show(text: str) -> None:
    """Print as markdown. rich silently drops HTML-like text (e.g. <div>),
    so escape '<' everywhere except inside code, where it already shows fine."""
    parts = CODE_RE.split(text)
    safe = "".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))
    console.print(Markdown(safe))


def render_tool_docs() -> str:
    """Turn the JSON schemas into prompt text the model can read."""
    docs = []
    for t in TOOL_SCHEMAS:
        props = t["input_schema"].get("properties", {})
        required = set(t["input_schema"].get("required", []))
        params = ", ".join(
            f'{k}: {v["type"]}{"" if k in required else " (optional)"}'
            for k, v in props.items()
        )
        docs.append(f'- {t["name"]}({params}): {t["description"]}')
    return "\n".join(docs)


def parse_tool_calls(text: str) -> tuple[list[dict], list[str]]:
    """Extract tool calls from model text. Returns (calls, parse_errors)."""
    calls, errors = [], []
    for raw in TOOL_CALL_RE.findall(text):
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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
            system = SYSTEM_PROMPT.format(
                workdir=self.cfg.workdir,
                tool_docs=render_tool_docs(),
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
            if self.chat_only:  # no tools: show everything as-is
                calls, errors, visible = [], [], text
            else:
                calls, errors = parse_tool_calls(text)
                visible = TOOL_CALL_RE.sub("", text).strip()

            if not text:
                # Empty reply: explain why instead of returning silently.
                if reply.finish_reason == "length":
                    console.print(
                        f"[yellow]Model ran out of tokens (max_tokens={self.cfg.max_tokens}) "
                        "before answering, probably while thinking. "
                        "Raise AGENT_MAX_TOKENS.[/yellow]"
                    )
                    return
                if not nudged:
                    nudged = True
                    console.print("[dim](empty reply, asking the model again)[/dim]")
                    self.messages.append({"role": "assistant", "content": "(no reply)"})
                    self.messages.append({"role": "user", "content":
                        "You sent an empty reply. Please answer the question."
                        if self.chat_only else
                        "You sent an empty reply. Either make a tool call or answer the question."})
                    continue
                console.print("[yellow]Model returned an empty reply twice. "
                              "Try rephrasing, or run with AGENT_DEBUG=1 to see the raw response.[/yellow]")
                return

            self.messages.append({"role": "assistant", "content": text})
            if visible:
                show(visible)

            if not calls and not errors:
                if reply.finish_reason == "length":
                    console.print("[yellow](reply was cut off: raise AGENT_MAX_TOKENS)[/yellow]")
                return  # plain answer = done

            results = [self._run_tool(c) for c in calls]
            results += [f"<tool_response>\nError: {e}\n</tool_response>" for e in errors]
            self.messages.append({"role": "user", "content": "\n".join(results)})

        console.print("[red]Stopped: hit max_turns.[/red]")

    def _run_tool(self, call: dict) -> str:
        name, args = call["name"], call["arguments"]
        console.print(f"[cyan]> {name}[/cyan] [dim]{json.dumps(args)[:120]}[/dim]")
        try:
            output = self.tools.run(name, args)
        except (ToolError, OSError, TypeError) as e:
            output = f"Error: {e}"
            console.print(f"[red]{output}[/red]")
        return f"<tool_response>\n[{name}]\n{output}\n</tool_response>"
