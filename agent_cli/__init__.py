"""agent-cli package.

Load a .env file into the environment before anything else, so the AGENT_*
defaults in config.py (evaluated at import time) can be overridden by a file
instead of exporting vars every session. Precedence ends up:

    CLI flags  >  real env vars  >  .env file  >  built-in defaults

The real environment wins over .env (override=False), so an exported var or a
one-off `AGENT_MODEL=foo agent ...` still beats the file.

Search order (first file found wins, nearest to the user's project first):
  1. ./.env in the current working directory
  2. ~/.config/agent-cli/.env   (persistent, repo-independent)

python-dotenv is optional: if it isn't installed we silently skip, so the CLI
still runs on a bare install.
"""
import os
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (Path.cwd() / ".env", Path.home() / ".config" / "agent-cli" / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)
            break


_load_env()
