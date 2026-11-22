"""Edit-style rules appended to the coder and micro Editor system prompts.

Adapted from the ideas in the MIT-licensed "ponytail" skill
(https://github.com/DietrichGebert/ponytail): make the smallest change that
fully solves the request, and say what was skipped.

AGENT_STYLE=full (default) | lite | off   (off exists to measure the baseline)
"""
import os
from typing import Optional

FULL_RULES = """# Change style: smallest complete change
- Make the smallest change that fully solves the request; nothing it did not ask for.
- Reuse existing helpers before writing new ones; never duplicate a function.
- No speculative options, abstractions, config or "future-proofing".
- Don't reformat, rename or reorder code you aren't otherwise changing.
- Prefer deleting code to adding it when both solve the problem.
- Add or update a test when you add branching logic, if the project has tests.
- If you knowingly take a shortcut, mark it: `<comment> shortcut: <limit>, <when to upgrade>`.
- End your reply with one line: `Risk: <what could break, or "low">`."""

LITE_RULES = """# Change style
- Make the smallest change that fully solves the request; reuse existing helpers.
- Don't touch code the request doesn't need.
- End your reply with one line: `Risk: <what could break, or "low">`."""


def style_name(value: Optional[str] = None) -> str:
    """Normalised AGENT_STYLE: full, lite or off. Unknown values mean full."""
    v = (os.getenv("AGENT_STYLE", "full") if value is None else value).strip().lower()
    return v if v in ("full", "lite", "off") else "full"


def style_rules(value: Optional[str] = None) -> str:
    """The rules text for a style ('' for off)."""
    name = style_name(value)
    return {"full": FULL_RULES, "lite": LITE_RULES}.get(name, "")


def with_style(system: str, value: Optional[str] = None) -> str:
    """`system` with the style rules appended (unchanged when the style is off)."""
    rules = style_rules(value)
    return f"{system.rstrip()}\n\n{rules}\n" if rules else system
