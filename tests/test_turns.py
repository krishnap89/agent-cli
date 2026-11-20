from datetime import datetime

from rich.console import Console

from agent_cli.turns import BAR, prompt_header

WHEN = datetime(2026, 11, 20, 14, 32)


def _render(renderable) -> str:
    console = Console(record=True, width=60, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_header_has_number_mode_time_and_text():
    out = _render(prompt_header(7, "why is it slow?", "agent", "green", when=WHEN))
    assert "#7 · agent · 14:32" in out
    assert "━" in out
    assert f"{BAR}why is it slow?" in out


def test_multiline_text_gets_bar_on_every_line():
    out = _render(prompt_header(1, "first\nsecond", "chat", "blue", when=WHEN))
    assert f"{BAR}first" in out and f"{BAR}second" in out


def test_markup_in_text_is_shown_literally():
    out = _render(prompt_header(2, "fix [bold]this[/bold]", "agent", "green", when=WHEN))
    assert "fix [bold]this[/bold]" in out


def test_rule_only_when_text_is_none():
    out = _render(prompt_header(3, None, "agent", "green", when=WHEN))
    assert "#3 · agent · 14:32" in out
    assert BAR not in out
