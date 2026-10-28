"""Read-only Jira Cloud client.

Talks to the Jira Cloud REST API v3 over stdlib urllib (no SDK), matching the
rest of this codebase. Authentication is HTTP Basic with an account email and
an API token (create one at id.atlassian.com → Security → API tokens), the
standard scheme for *.atlassian.net sites.

Only reads: fetch one issue, or search issues with JQL. Nothing here mutates
Jira, so the tools built on it need no confirmation prompt.

Configured through three env vars (see Config): AGENT_JIRA_URL, AGENT_JIRA_EMAIL,
AGENT_JIRA_TOKEN. When any is missing, is_configured() is False and the Jira
tools are not advertised to the model.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .config import Config


class JiraError(Exception):
    pass


def is_configured(cfg: Config) -> bool:
    return bool(cfg.jira_url and cfg.jira_email and cfg.jira_token)


def _auth_header(cfg: Config) -> str:
    raw = f"{cfg.jira_email}:{cfg.jira_token}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _get(cfg: Config, path: str, params: Optional[dict] = None) -> Any:
    """GET <site>/rest/api/3/<path> and return the parsed JSON body."""
    url = cfg.jira_url.rstrip("/") + "/rest/api/3/" + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": _auth_header(cfg),
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.jira_timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300].strip()
        if e.code == 401:
            raise JiraError("Jira auth failed (401): check AGENT_JIRA_EMAIL and "
                            "AGENT_JIRA_TOKEN.") from e
        if e.code == 404:
            raise JiraError(f"Not found (404) at {url}: {body}") from e
        raise JiraError(f"HTTP {e.code} from Jira: {body}") from e
    except urllib.error.URLError as e:
        raise JiraError(f"Cannot reach Jira at {cfg.jira_url}: {e.reason}") from e


# ---- ADF (Atlassian Document Format) -> plain text --------------------------

def adf_to_text(node: Any) -> str:
    """Flatten a Jira rich-text (ADF) field to readable plain text.

    Descriptions and comments come back as a nested document, not a string.
    We keep paragraphs, list items and line breaks; everything else is walked
    for its text. Good enough for the model to read; not a faithful renderer.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    out: list[str] = []

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        t = n.get("type")
        if t == "text":
            out.append(n.get("text", ""))
            return
        if t == "hardBreak":
            out.append("\n")
            return
        for child in n.get("content", []) or []:
            walk(child)
        if t in ("paragraph", "heading", "listItem", "blockquote", "codeBlock"):
            out.append("\n")
        if t == "listItem":
            # Prefix the line we just emitted; cheap bullet.
            pass

    walk(node)
    text = "".join(out)
    # Collapse the runs of blank lines the block separators can create.
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    for ln in lines:
        if ln == "" and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


# ---- public operations ------------------------------------------------------

def get_issue_fields(cfg: Config, key: str) -> dict:
    """Fetch one issue and return a flat dict of the fields callers need.

    Keys: key, summary, status, issuetype, priority, description (plain text),
    comments (list of plain-text strings), labels (list). Used by code that
    composes the issue into a prompt, where the rendered blob from get_issue()
    would be harder to work with.
    """
    key = key.strip().upper()
    data = _get(cfg, f"issue/{urllib.parse.quote(key)}", {
        "fields": "summary,status,issuetype,priority,assignee,reporter,labels,"
                  "description,comment",
    })
    f = data.get("fields", {})

    def name(obj: Any) -> str:
        return obj.get("displayName") or obj.get("name") or "" if isinstance(obj, dict) else ""

    comments = (f.get("comment") or {}).get("comments") or []
    return {
        "key": data.get("key", key),
        "summary": f.get("summary", ""),
        "status": name(f.get("status")),
        "issuetype": name(f.get("issuetype")),
        "priority": name(f.get("priority")),
        "labels": f.get("labels") or [],
        "description": adf_to_text(f.get("description")),
        "comments": [adf_to_text(c.get("body")) for c in comments],
    }


def get_issue(cfg: Config, key: str) -> str:
    """Return a human/model-readable summary of one issue, e.g. PROJ-123."""
    key = key.strip().upper()
    data = _get(cfg, f"issue/{urllib.parse.quote(key)}", {
        "fields": "summary,status,issuetype,priority,assignee,reporter,labels,"
                  "description,comment",
    })
    f = data.get("fields", {})

    def name(obj: Any) -> str:
        return obj.get("displayName") or obj.get("name") or "" if isinstance(obj, dict) else ""

    lines = [
        f"{data.get('key', key)}: {f.get('summary', '')}",
        f"Type: {name(f.get('issuetype'))}  "
        f"Status: {name(f.get('status'))}  "
        f"Priority: {name(f.get('priority'))}",
        f"Assignee: {name(f.get('assignee')) or 'Unassigned'}  "
        f"Reporter: {name(f.get('reporter'))}",
    ]
    labels = f.get("labels") or []
    if labels:
        lines.append("Labels: " + ", ".join(labels))

    desc = adf_to_text(f.get("description"))
    lines.append("\n--- Description ---\n" + (desc or "(none)"))

    comments = (f.get("comment") or {}).get("comments") or []
    if comments:
        lines.append(f"\n--- Comments ({len(comments)}) ---")
        for c in comments:
            author = name(c.get("author"))
            body = adf_to_text(c.get("body"))
            lines.append(f"[{author}] {body}")

    return "\n".join(lines)


def search_issues(cfg: Config, jql: str, max_results: int = 20) -> str:
    """Run a JQL query and return a compact list of matching issues."""
    max_results = max(1, min(int(max_results), 50))
    data = _get(cfg, "search", {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,status,issuetype,assignee",
    })
    issues = data.get("issues") or []
    total = data.get("total", len(issues))
    if not issues:
        return f"No issues match: {jql}"

    def name(obj: Any) -> str:
        return obj.get("displayName") or obj.get("name") or "" if isinstance(obj, dict) else ""

    lines = [f"{len(issues)} of {total} issue(s) for: {jql}"]
    for it in issues:
        f = it.get("fields", {})
        lines.append(
            f"{it.get('key')}  [{name(f.get('status'))}]  "
            f"{name(f.get('issuetype'))}  "
            f"{f.get('summary', '')}  "
            f"(@{name(f.get('assignee')) or 'unassigned'})"
        )
    return "\n".join(lines)
