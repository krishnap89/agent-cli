"""The --version option shared by every command."""
import typer

from . import __version__


def _show(value: bool) -> None:
    if value:
        typer.echo(f"agent-cli {__version__}")
        raise typer.Exit()


# Eager, so it exits before any config, server or project work.
VERSION_OPTION = typer.Option(False, "--version", callback=_show, is_eager=True,
                              help="Show the version and exit.")
