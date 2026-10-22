"""Pick a model per micro-agent role from a role->model map, no LLM call.

`cfg.auto_model` turns this on. When off, `model_for` always returns the single
`cfg.model` so the loop behaves exactly as before.

Each micro-agent role does a different job, so a specialised model can be
assigned to each via the `AGENT_MODEL_MAP` JSON knob (parsed into
`cfg.model_map`). A typical setup points Editor/Fixer at a code model, Planner
at a reasoning model, Locator at a small fast model, and Reviewer at a review
model. Any role the map does not name falls back to the map's "default" key, or
to `cfg.model` when that is absent too.

The roles are: Locator, Planner, Editor, Fixer, Reviewer, Summarizer.
"""
import json
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..config import Config

# All micro-agent roles, so callers (and docs) have the canonical set.
ROLES = ("Locator", "Planner", "Editor", "Fixer", "Reviewer", "Summarizer")

# Fallback key: a model under this name in the map applies to every role the map
# does not name explicitly.
DEFAULT_KEY = "default"


def parse_map(raw: str) -> Dict[str, str]:
    """Parse the AGENT_MODEL_MAP JSON into a {role: model} dict.

    Tolerant by design: a blank, malformed, or non-object value yields an empty
    map so auto-selection silently falls back to cfg.model rather than breaking
    a run. Non-string values are dropped. Role keys are matched case-insensitively
    against the canonical names, so {"editor": "..."} works too.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(obj, dict):
        return {}
    canon = {r.lower(): r for r in ROLES}
    canon[DEFAULT_KEY] = DEFAULT_KEY
    out: Dict[str, str] = {}
    for key, value in obj.items():
        if not isinstance(value, str) or not value.strip():
            continue
        role = canon.get(str(key).strip().lower())
        if role:
            out[role] = value.strip()
    return out


def model_for(cfg: "Config", role: str) -> str:
    """The model name this role should run on.

    With auto_model off, always cfg.model. With it on, the role's mapped model,
    then the map's "default", then cfg.model.
    """
    if not cfg.auto_model:
        return cfg.model
    mapping = parse_map(cfg.model_map)
    return mapping.get(role) or mapping.get(DEFAULT_KEY) or cfg.model
