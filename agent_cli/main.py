from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .agent import Agent
from .config import Config
from .input_reader import read_message
from .llm import list_models

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="One-shot task. Omit for interactive mode."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve writes and commands."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-u", help="OpenAI-compatible endpoint"),
    workdir: Path = typer.Option(Path.cwd(), "--dir", "-d"),
    check: bool = typer.Option(False, "--check", help="Test the server and list models."),
    chat_only: bool = typer.Option(False, "--chat", "-c", help="Chat only: no tools, no file access."),
    show_map: Optional[str] = typer.Option(None, "--map", help='Print the repo map and exit. Use --map "" or --map "query".'),
):
    cfg = Config(workdir=workdir.resolve(), auto_approve=yes)
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url
    if check:
        try:
            models = list_models(cfg)
            console.print(f"[green]Server OK[/green] at {cfg.base_url}")
            for m in models:
                console.print(f"  - {m}")
        except Exception as e:
            console.print(f"[red]Cannot reach {cfg.base_url}: {e}[/red]")
        return

    if show_map is not None:
        from .repomap import RepoMap
        rm = RepoMap(cfg.workdir)
        with console.status("[dim]indexing...[/dim]"):
            total, parsed = rm.build()
        console.print(f"[dim]{total} files indexed ({parsed} parsed, rest from cache)[/dim]\n")
        console.print(rm.render(query=show_map, budget=cfg.map_chars), markup=False, highlight=False)
        return

    agent = Agent(cfg, chat_only=chat_only)

    if prompt:
        agent.ask(prompt)
        return

    console.print(f"[bold]agent[/bold] · {cfg.model} @ {cfg.base_url} · {cfg.workdir}")
    console.print(f"[dim]Mode: {agent.mode_name}. "
                  "/chat = chat only, /agent = use tools, /map [query] = repo map, /clear = reset, /exit = quit.\n"
                  '""" starts/ends a multi-line message; pastes are kept together.[/dim]\n')
    while True:
        try:
            color = "blue" if agent.chat_only else "green"
            text = read_message(f"[bold {color}]{agent.mode_name} › [/bold {color}]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text in ("/chat", "/agent"):
            agent.set_mode(text == "/chat")
            console.print(f"[dim]Switched to {agent.mode_name} mode (history cleared).[/dim]")
            continue
        if text.startswith("/map"):
            query = text[4:].strip()
            console.print(agent.tools.ensure_map().render(query=query, budget=cfg.map_chars),
                          markup=False, highlight=False)
            continue
        if text == "/clear":
            agent.reset()
            console.print("[dim]History cleared.[/dim]")
            continue
        try:
            agent.ask(text)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        console.print()


if __name__ == "__main__":
    app()
