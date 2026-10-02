"""Skills: loadable instruction sets from .agent/skills/<name>/SKILL.md.

Each SKILL.md has YAML-like front matter (parsed by hand, no pyyaml):
  ---
  name: skill-name
  description: one-line summary
  triggers:
    - keyword1
    - keyword2
  ---
  Markdown instructions follow.

Skills are scored against user messages by counting trigger/description word
matches.  The top 1–2 (score > 0) are injected as context.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

console = Console()

SKILLS_DIR = ".agent/skills"


@dataclass
class Skill:
    name: str
    description: str
    triggers: List[str]
    body: str
    path: Path

    def truncated(self, budget: int) -> str:
        if len(self.body) <= budget:
            return self.body
        return self.body[:budget] + "\n...(skill truncated)..."


def _parse_front_matter(text: str) -> Tuple[Dict[str, object], str]:
    """Parse YAML-like front matter by hand (no pyyaml dependency).

    Returns (metadata_dict, body_after_front_matter).
    Supports scalar values and simple list values (  - item).
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fm_block = text[4:end]  # after first "---\n"
    body = text[end + 4:]   # after closing "---\n"
    if body.startswith("\n"):
        body = body[1:]

    meta: Dict[str, object] = {}
    current_key = ""
    current_list: Optional[List[str]] = None

    for line in fm_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # list item?
        if stripped.startswith("- ") and current_key and current_list is not None:
            current_list.append(stripped[2:].strip())
            continue
        # new key: value
        if ":" in stripped:
            if current_key and current_list is not None:
                meta[current_key] = current_list
            colon = stripped.index(":")
            key = stripped[:colon].strip()
            val = stripped[colon + 1:].strip()
            if val:
                meta[key] = val
                current_key = ""
                current_list = None
            else:
                current_key = key
                current_list = []

    if current_key and current_list is not None:
        meta[current_key] = current_list

    return meta, body


def load_skill(skill_dir: Path) -> Optional[Skill]:
    """Load a single skill from a directory containing SKILL.md."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return None
    try:
        text = md.read_text(errors="replace")
    except OSError:
        return None
    meta, body = _parse_front_matter(text)
    name = str(meta.get("name", skill_dir.name))
    description = str(meta.get("description", ""))
    raw_triggers = meta.get("triggers", [])
    triggers = list(raw_triggers) if isinstance(raw_triggers, list) else []
    if not body.strip():
        return None
    return Skill(name=name, description=description, triggers=triggers,
                 body=body.strip(), path=md)


def load_skills(workdir: Path) -> List[Skill]:
    """Load all skills from <workdir>/.agent/skills/*/SKILL.md."""
    skills_root = workdir / SKILLS_DIR
    if not skills_root.is_dir():
        return []
    loaded = []
    for entry in sorted(skills_root.iterdir()):
        if entry.is_dir():
            skill = load_skill(entry)
            if skill is not None:
                loaded.append(skill)
    return loaded


def _tokenise(text: str) -> List[str]:
    """Split text into lowercase words for matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def score_skill(skill: Skill, message: str) -> int:
    """Score a skill against a user message.

    Counts how many trigger words and description words appear in the message.
    """
    msg_words = set(_tokenise(message))
    if not msg_words:
        return 0
    score = 0
    for trigger in skill.triggers:
        for word in _tokenise(trigger):
            if word in msg_words:
                score += 1
    for word in _tokenise(skill.description):
        if word in msg_words:
            score += 1
    return score


def select_skills(skills: List[Skill], message: str,
                  max_skills: int = 2) -> List[Skill]:
    """Pick the top 1–2 skills that match the message (score > 0)."""
    scored = [(score_skill(s, message), s) for s in skills]
    scored = [(sc, s) for sc, s in scored if sc > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_skills]]


def format_skill_context(skill: Skill, budget: int) -> str:
    """Format a skill's instructions for injection into the prompt."""
    header = f"# Skill: {skill.name}"
    if skill.description:
        header += f"\n{skill.description}\n"
    return header + "\n" + skill.truncated(budget)


def list_skills_display(skills: List[Skill]) -> str:
    """Format the skill list for /skills command."""
    if not skills:
        return "No skills loaded. Create .agent/skills/<name>/SKILL.md to add one."
    lines = []
    for s in skills:
        triggers = ", ".join(s.triggers[:5]) if s.triggers else "(none)"
        lines.append(f"  {s.name:20s} {s.description or '(no description)'}")
        lines.append(f"  {'':20s} triggers: {triggers}")
    return "\n".join(lines)
