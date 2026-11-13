"""Fact extraction for iOS / Swift project files (everything except .swift).

  spm        Package.swift  {"targets": [[name, path, kind, [deps], line]]}
             kind: target | executable | test | macro; paths are relative to
             the package directory
  xcodeproj  project.pbxproj {"targets": [[module, kind, [files], [folders], [deps], line]]}
             files/folders are repo-relative; deps are target / package-product names
  ib         .storyboard / .xib  {"rtype": "storyboard"|"nib", "rname": stem,
             "ids": [[storyboardIdentifier, line]], "classes": [[Module.Class|Class, line]],
             "actions": [[selector_base_name, line]]}
  strings    .strings  {"keys": [[key, line]]}
  xcstrings  String Catalog  {"keys": [[key, line]]}
  xcasset    Contents.json of an .imageset/.colorset/...  {"rtype": "image"|"color"|..., "rname": name}
  plist      Info.plist  {"classes": [[Class, line]]}
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

ASSET_SETS = {".imageset": "image", ".colorset": "color", ".symbolset": "image",
              ".dataset": "data"}


def _line_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _module_name(target: str) -> str:
    """Xcode's default PRODUCT_MODULE_NAME for a target name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", target)
    return "_" + name if name[:1].isdigit() else name


# ---- Package.swift ------------------------------------------------------------

_SPM_TARGET_RE = re.compile(r"\.(target|executableTarget|testTarget|macro|plugin)\s*\(")
_SPM_NAME_RE = re.compile(r"""\bname\s*:\s*"([^"]+)\"""")
_SPM_PATH_RE = re.compile(r"""\bpath\s*:\s*"([^"]+)\"""")
_SPM_DEPS_RE = re.compile(r"\bdependencies\s*:\s*\[")


def _balanced(text: str, open_at: int, opener: str = "(", closer: str = ")") -> int:
    """Index just past the closer matching text[open_at] (strings skipped)."""
    depth, i, n = 0, open_at, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = text.find('"', i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def parse_package_swift(text: str) -> dict:
    text = re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    targets = []
    for m in _SPM_TARGET_RE.finditer(text):
        open_at = m.end() - 1
        body = text[open_at:_balanced(text, open_at)]
        nm = _SPM_NAME_RE.search(body)
        if not nm:
            continue
        kind = {"executableTarget": "executable", "testTarget": "test"}.get(m.group(1), m.group(1))
        pm = _SPM_PATH_RE.search(body)
        path = pm.group(1) if pm else ("Tests/" if kind == "test" else "Sources/") + nm.group(1)
        deps = []
        dm = _SPM_DEPS_RE.search(body)
        if dm:
            lst = body[dm.end() - 1:_balanced(body, dm.end() - 1, "[", "]")]
            # "Name", .target(name: "Name"), .product(name: "P", package: "pkg"), .byName(name: "N")
            for q in re.finditer(r'''(?:\.(?:target|product|byName)\s*\(\s*name\s*:\s*)?"([^"]+)"''', lst):
                prefix = lst[max(0, q.start() - 12):q.start()]
                if "package:" in prefix:
                    continue
                deps.append(q.group(1))
        targets.append([nm.group(1), path.rstrip("/"), kind, deps, _line_at(text, m.start())])
    return {"targets": targets}


# ---- project.pbxproj (OpenStep plist) --------------------------------------------

class _PlistParser:
    _TOKEN = re.compile(r'\s+|/\*.*?\*/|//[^\n]*|"(?:[^"\\]|\\.)*"|[{}()=;,]|[^\s{}()=;,"]+', re.S)

    def __init__(self, text: str):
        self.toks = []  # type: List[Tuple[str, int]]
        for m in self._TOKEN.finditer(text):
            s = m.group(0)
            if s[0].isspace() or s.startswith("/*") or s.startswith("//"):
                continue
            self.toks.append((s, m.start()))
        self.i = 0

    def value(self) -> Any:
        if self.i >= len(self.toks):
            return None
        s, _pos = self.toks[self.i]
        self.i += 1
        if s == "{":
            d = {}
            while self.i < len(self.toks) and self.toks[self.i][0] != "}":
                key = self.value()
                if self.i < len(self.toks) and self.toks[self.i][0] == "=":
                    self.i += 1
                d[key] = self.value()
                if self.i < len(self.toks) and self.toks[self.i][0] == ";":
                    self.i += 1
            self.i += 1
            return d
        if s == "(":
            items = []
            while self.i < len(self.toks) and self.toks[self.i][0] != ")":
                items.append(self.value())
                if self.i < len(self.toks) and self.toks[self.i][0] == ",":
                    self.i += 1
            self.i += 1
            return items
        if s.startswith('"'):
            return bytes(s[1:-1], "utf-8").decode("unicode_escape", errors="replace")
        return s


def parse_openstep_plist(text: str) -> Any:
    """Parse an old-style (OpenStep) plist such as project.pbxproj."""
    return _PlistParser(text).value()


def parse_pbxproj(text: str, rel: str) -> dict:
    """Targets, their source files / synchronized folders and dependencies.
    `rel` is the repo-relative path of the project.pbxproj file."""
    try:
        root = parse_openstep_plist(text)
        objects = root["objects"]  # type: Dict[str, dict]
        project = objects[root["rootObject"]]
    except (KeyError, TypeError, ValueError, IndexError):
        return {"targets": []}
    proj_dir = os.path.dirname(os.path.dirname(rel))  # …/App.xcodeproj/project.pbxproj
    paths = {}  # type: Dict[str, str]

    def join(base: str, p: str) -> str:
        return os.path.normpath(os.path.join(base, p)).replace(os.sep, "/") if p else base

    def walk(gid: str, base: str) -> None:
        g = objects.get(gid) or {}
        tree = g.get("sourceTree", "<group>")
        if tree == "SOURCE_ROOT":
            base = proj_dir
        elif tree not in ("<group>",):
            return  # SDK / absolute / build products
        here = join(base, g.get("path", ""))
        if here == "":
            here = "."
        paths[gid] = here
        for child in g.get("children", ()) or ():
            c = objects.get(child) or {}
            isa = c.get("isa", "")
            if isa in ("PBXGroup", "PBXVariantGroup", "PBXFileSystemSynchronizedRootGroup",
                       "XCVersionGroup"):
                walk(child, here if isa != "PBXVariantGroup" else here)
            elif isa == "PBXFileReference":
                ctree = c.get("sourceTree", "<group>")
                if ctree == "<group>":
                    paths[child] = join(here, c.get("path", ""))
                elif ctree == "SOURCE_ROOT":
                    paths[child] = join(proj_dir, c.get("path", ""))

    walk(project.get("mainGroup", ""), proj_dir)

    def files_of(phase_ids: List[str]) -> List[str]:
        out = []
        for ph in phase_ids or ():
            phase = objects.get(ph) or {}
            if phase.get("isa") not in ("PBXSourcesBuildPhase", "PBXResourcesBuildPhase"):
                continue
            for bf in phase.get("files", ()) or ():
                ref = (objects.get(bf) or {}).get("fileRef")
                refobj = objects.get(ref) or {}
                if refobj.get("isa") == "PBXVariantGroup":
                    out.extend(paths[c] for c in refobj.get("children", ()) if c in paths)
                elif ref in paths:
                    out.append(paths[ref])
        return out

    targets = []
    for tid in project.get("targets", ()) or ():
        t = objects.get(tid) or {}
        if t.get("isa") not in ("PBXNativeTarget", "PBXAggregateTarget"):
            continue
        name = t.get("name", "")
        ptype = t.get("productType", "")
        kind = "test" if "test" in ptype else "app" if "application" in ptype else "target"
        folders = [paths[g] for g in t.get("fileSystemSynchronizedGroups", ()) or () if g in paths]
        deps = []
        for dep in t.get("dependencies", ()) or ():
            tgt = objects.get((objects.get(dep) or {}).get("target", "")) or {}
            if tgt.get("name"):
                deps.append(_module_name(tgt["name"]))
        for pd in t.get("packageProductDependencies", ()) or ():
            pn = (objects.get(pd) or {}).get("productName")
            if pn:
                deps.append(pn)
        line = _line_at(text, text.find(tid + " /*")) if tid + " /*" in text else 1
        targets.append([_module_name(name), kind, sorted(set(files_of(t.get("buildPhases")))),
                        folders, deps, line])
    return {"targets": targets}


# ---- Interface Builder ----------------------------------------------------------

_CUSTOM_CLASS_RE = re.compile(r"""<(\w+)\b[^>]*?\bcustomClass="([\w.]+)"[^>]*>""", re.S)
_MODULE_ATTR_RE = re.compile(r"""\bcustomModule="(\w+)\"""")
_SB_ID_RE = re.compile(r"""\bstoryboardIdentifier="([^"]+)\"""")
_ACTION_RE = re.compile(r"""<action\b[^>]*\bselector="([\w]+)""")


def parse_ib(text: str, rel: str) -> dict:
    text = re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    base = os.path.basename(rel)
    stem, ext = os.path.splitext(base)
    facts = {"rtype": "storyboard" if ext == ".storyboard" else "nib", "rname": stem,
             "ids": [], "classes": [], "actions": []}  # type: dict
    for m in _CUSTOM_CLASS_RE.finditer(text):
        mod = _MODULE_ATTR_RE.search(m.group(0))
        name = f"{mod.group(1)}.{m.group(2)}" if mod else m.group(2)
        facts["classes"].append([name, _line_at(text, m.start())])
    for m in _SB_ID_RE.finditer(text):
        facts["ids"].append([m.group(1), _line_at(text, m.start())])
    for m in _ACTION_RE.finditer(text):
        facts["actions"].append([m.group(1), _line_at(text, m.start())])
    return facts


# ---- strings / catalogs / assets / Info.plist ---------------------------------------

_STRINGS_RE = re.compile(r'^\s*"((?:[^"\\]|\\.)+)"\s*=', re.M)


def parse_strings(text: str) -> dict:
    return {"keys": [[m.group(1), _line_at(text, m.start())] for m in _STRINGS_RE.finditer(text)]}


def parse_xcstrings(text: str) -> dict:
    try:
        data = json.loads(text)
        keys = list((data.get("strings") or {}).keys())
    except (ValueError, AttributeError):
        return {"keys": []}
    out = []
    for k in keys:
        pos = text.find(json.dumps(k))
        out.append([k, _line_at(text, pos) if pos >= 0 else 1])
    return {"keys": out}


def asset_kind(rel: str) -> Optional[Tuple[str, str]]:
    """'App/Assets.xcassets/logo.imageset/Contents.json' -> ('image', 'logo')."""
    parts = rel.split("/")
    if len(parts) < 2 or parts[-1] != "Contents.json":
        return None
    folder = parts[-2]
    stem, ext = os.path.splitext(folder)
    if ext in ASSET_SETS:
        return ASSET_SETS[ext], stem
    return None


_PLIST_CLASS_RE = re.compile(r"<string>\$\(PRODUCT_MODULE_NAME\)\.(\w+)</string>")


def parse_info_plist(text: str) -> dict:
    return {"classes": [[m.group(1), _line_at(text, m.start())] for m in _PLIST_CLASS_RE.finditer(text)]}
