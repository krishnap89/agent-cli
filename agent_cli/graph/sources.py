"""Which files the code graph reads, and per-file fact extraction for the
non-Kotlin ones (Gradle build scripts, Android resource XML, manifests).

Every extractor returns a JSON-serializable dict with a "lang" key:
  kotlin    see kotlin_parser.parse_kotlin
  gradle    {"deps": [[configuration, project_path, line]], "namespace": str,
             "includes": [[":path", line]]}   (includes: settings files only)
  res       {"rtype": "layout", "rname": "activity_main" (file resources),
             "ids": [[name, line]], "values": [[type, name, line]],
             "classes": [[class_name, line]]}
  manifest  {"package": str, "classes": [[class_name, line]]}
"""
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from ..repomap import MAX_FILE_BYTES, SKIP_DIRS
from .kotlin_parser import parse_kotlin

GRADLE_NAMES = frozenset({"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"})
# Resource folders whose files are resources themselves (R.layout.main -> layout/main.xml).
FILE_RES_TYPES = frozenset({
    "layout", "menu", "navigation", "drawable", "mipmap", "xml", "anim", "animator",
    "color", "font", "transition", "raw",
})
VALUE_TAGS = {
    "string": "string", "color": "color", "dimen": "dimen", "bool": "bool",
    "integer": "integer", "style": "style", "plurals": "plurals",
    "string-array": "array", "integer-array": "array", "array": "array",
    "declare-styleable": "styleable", "attr": "attr", "item": None,
}


def classify(rel: str) -> Optional[str]:
    """The extractor for a repo-relative path, or None to skip the file."""
    p = Path(rel)
    name = p.name
    if any(part in SKIP_DIRS for part in p.parts[:-1]):
        return None
    if name in GRADLE_NAMES:
        return "gradle"
    suffix = p.suffix.lower()
    if suffix == ".kt":
        return "kotlin"
    if suffix == ".xml":
        if name == "AndroidManifest.xml":
            return "manifest"
        parts = p.parts
        if len(parts) >= 3 and parts[-3] == "res":
            return "res"
    return None


def list_sources(root: Path) -> List[str]:
    """Repo-relative paths of graph inputs (git ls-files, else a walk)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return sorted(f for f in out.stdout.splitlines() if classify(f))
    except (OSError, subprocess.SubprocessError):
        pass
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if classify(rel):
                found.append(rel)
    return sorted(found)


def extract(root: Path, rel: str) -> Optional[dict]:
    """Facts for one file, or None if it cannot be read or is too big."""
    lang = classify(rel)
    path = root / rel
    try:
        if lang is None or path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if lang == "kotlin":
        facts = parse_kotlin(text)
    elif lang == "gradle":
        facts = parse_gradle(text, settings=rel.rsplit("/", 1)[-1].startswith("settings."))
    elif lang == "manifest":
        facts = parse_manifest(text)
    else:
        facts = parse_res_xml(text, rel)
    facts["lang"] = lang
    return facts


# ---- Gradle ------------------------------------------------------------------

_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_PROJECT_DEP_RE = re.compile(
    r"""(\w+)\s*\(?\s*project\s*\(\s*(?:path\s*[:=]\s*)?["']([^"']+)["']""")
_ACCESSOR_DEP_RE = re.compile(r"""(\w+)\s*\(?\s*projects((?:\.\w+)+)""")
_INCLUDE_RE = re.compile(r"""\binclude\b\s*\(?([^\n)]*)""")
_QUOTED_RE = re.compile(r"""["']([^"']+)["']""")
_NAMESPACE_RE = re.compile(r"""\b(?:namespace|applicationId)\s*=?\s*["']([\w.]+)["']""")


def _blank_comments(text: str) -> str:
    """Remove comments but keep line numbers (and offsets) intact."""
    return _COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _line_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def parse_gradle(text: str, settings: bool = False) -> dict:
    text = _blank_comments(text)
    deps = []
    for m in _PROJECT_DEP_RE.finditer(text):
        deps.append([m.group(1), m.group(2), _line_at(text, m.start())])
    for m in _ACCESSOR_DEP_RE.finditer(text):
        # typesafe accessor: projects.core.data -> ":core:data" (matched
        # loosely later, since featureLogin may mean :feature-login)
        deps.append([m.group(1), "~" + m.group(2).replace(".", ":"), _line_at(text, m.start())])
    includes = []
    if settings:
        for m in _INCLUDE_RE.finditer(text):
            line = _line_at(text, m.start())
            for q in _QUOTED_RE.findall(m.group(1)):
                includes.append([q if q.startswith(":") else ":" + q, line])
    ns = _NAMESPACE_RE.search(text)
    return {"deps": deps, "includes": includes, "namespace": ns.group(1) if ns else ""}


# ---- Android XML -------------------------------------------------------------

_XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_ID_DEF_RE = re.compile(r"""@\+id/([\w.]+)""")
_TAG_RE = re.compile(r"<([A-Za-z_][\w.\-]*)([^>]*)>", re.S)
_NAME_ATTR_RE = re.compile(r"""\bname\s*=\s*["']([^"']+)["']""")
_CLASS_ATTR_RE = re.compile(
    r"""\b(?:tools:context|android:name|class|app:layoutManager|android:targetClass)\s*=\s*["']([\w.$]+)["']""")


def _blank_xml_comments(text: str) -> str:
    return _XML_COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _class_like(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return bool(last) and last[0].isupper() and ("." in name)


def parse_res_xml(text: str, rel: str) -> dict:
    text = _blank_xml_comments(text)
    folder = rel.split("/")[-2]
    rtype = folder.split("-", 1)[0]
    facts = {"rtype": rtype, "rname": "", "ids": [], "values": [], "classes": []}  # type: dict
    if rtype in FILE_RES_TYPES:
        facts["rname"] = rel.rsplit("/", 1)[-1].split(".", 1)[0]
    seen_ids = set()
    for m in _ID_DEF_RE.finditer(text):
        if m.group(1) not in seen_ids:
            seen_ids.add(m.group(1))
            facts["ids"].append([m.group(1), _line_at(text, m.start())])
    for m in _TAG_RE.finditer(text):
        tag, attrs = m.group(1), m.group(2)
        line = _line_at(text, m.start())
        if rtype == "values" and tag in VALUE_TAGS and VALUE_TAGS[tag]:
            nm = _NAME_ATTR_RE.search(attrs)
            if nm:
                facts["values"].append([VALUE_TAGS[tag], nm.group(1).replace(".", "_"), line])
        if _class_like(tag) and not tag.startswith(("android.", "androidx.", "com.google.android.")):
            facts["classes"].append([tag, line])
        for cm in _CLASS_ATTR_RE.finditer(attrs):
            val = cm.group(1)
            if val.startswith(".") or _class_like(val):
                facts["classes"].append([val, line])
    return facts


_MANIFEST_PKG_RE = re.compile(r"""<manifest\b[^>]*\bpackage\s*=\s*["']([\w.]+)["']""", re.S)


def parse_manifest(text: str) -> dict:
    text = _blank_xml_comments(text)
    pkg = _MANIFEST_PKG_RE.search(text)
    classes = []
    for m in _TAG_RE.finditer(text):
        tag, attrs = m.group(1), m.group(2)
        if tag not in ("activity", "service", "receiver", "provider", "application",
                       "activity-alias"):
            continue
        for nm in re.finditer(r"""android:name\s*=\s*["']([\w.$]+)["']""", attrs):
            classes.append([nm.group(1), _line_at(text, m.start())])
    return {"package": pkg.group(1) if pkg else "", "classes": classes}
