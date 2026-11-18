"""TypeScript / JavaScript project files and module-specifier resolution.

Per-file facts (cached like the others):
  npm       package.json  {"name", "main", "module", "types", "exports":
                           {subpath: [targets]}, "deps": [[name, line]]}
  tsconfig  tsconfig*.json / jsconfig*.json (JSON with comments and trailing
            commas)  {"extends": [..], "baseUrl": str|None, "paths": {..}|None}

TsModules.resolve(from_rel, spec) maps an import specifier to a repo file:
  ./x, ../x       relative (x.ts, x.tsx, x.d.ts, x.js…, x/index.*; an
                  explicit .js that names a .ts source, as in ESM TypeScript)
  @/x, ~/x, …     tsconfig `paths` of the nearest tsconfig (following
                  relative `extends`; solution-style configs fall back to a
                  sibling tsconfig.*.json that has paths), then `baseUrl`
  @scope/pkg[/x]  a workspace package by package.json name: exports / types /
                  module / main, with build output (dist/, lib/, …) mapped
                  back to src/, else src/index / index
Anything else (react, node:fs, CSS/asset imports) is external: None.
"""
import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

TS_SOURCE_EXTS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_LOOKUP_EXTS = (".ts", ".tsx", ".d.ts", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")
_BUILD_DIRS = ("dist/", "build/", "lib/", "out/", "esm/", "cjs/", "es/", "types/")
_DEP_KEYS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")
_CONDITIONS = ("types", "typings", "source", "import", "module", "default", "require", "node",
               "browser")


def strip_ext(rel: str) -> str:
    """'src/a/user.service.ts' -> 'src/a/user.service'; '.d.ts' counts as one."""
    low = rel.lower()
    for ext in (".d.ts", ".d.mts", ".d.cts") + TS_SOURCE_EXTS:
        if low.endswith(ext):
            return rel[:-len(ext)]
    return rel


def _norm(path: str) -> Optional[str]:
    parts = []  # type: List[str]
    for seg in path.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None  # outside the repo
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts) or "."


def _join(base: str, rel: str) -> Optional[str]:
    return _norm(rel if base in ("", ".") else base + "/" + rel)


def _dirname(rel: str) -> str:
    return rel.rsplit("/", 1)[0] if "/" in rel else "."


# ---- JSON with comments ------------------------------------------------------

def _strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments outside strings, and trailing commas."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def parse_tsconfig(text: str) -> dict:
    try:
        data = json.loads(_strip_jsonc(text))
    except ValueError:
        return {"extends": [], "baseUrl": None, "paths": None}
    if not isinstance(data, dict):
        data = {}
    ext = data.get("extends")
    exts = [ext] if isinstance(ext, str) else [e for e in ext if isinstance(e, str)] \
        if isinstance(ext, list) else []
    opts = data.get("compilerOptions") if isinstance(data.get("compilerOptions"), dict) else {}
    base = opts.get("baseUrl") if isinstance(opts.get("baseUrl"), str) else None
    paths = opts.get("paths") if isinstance(opts.get("paths"), dict) else None
    if paths is not None:
        paths = {k: [t for t in v if isinstance(t, str)] for k, v in paths.items()
                 if isinstance(k, str) and isinstance(v, list)}
    return {"extends": exts, "baseUrl": base, "paths": paths}


def _targets(value) -> List[str]:
    """Flatten an exports condition value into candidate paths, best first."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [t for v in value for t in _targets(v)]
    if isinstance(value, dict):
        out = []
        for key in _CONDITIONS:
            if key in value:
                out.extend(_targets(value[key]))
        for key, v in value.items():
            if key not in _CONDITIONS:
                out.extend(_targets(v))
        return out
    return []


def parse_package_json(text: str) -> dict:
    facts = {"name": "", "main": "", "module": "", "types": "", "exports": {}, "deps": []}
    try:
        data = json.loads(text)
    except ValueError:
        return facts
    if not isinstance(data, dict):
        return facts
    for key in ("name", "main", "module"):
        if isinstance(data.get(key), str):
            facts[key] = data[key]
    for key in ("types", "typings", "source"):
        if isinstance(data.get(key), str) and not facts["types"]:
            facts["types"] = data[key]
    ex = data.get("exports")
    if isinstance(ex, (str, list)):
        facts["exports"] = {".": _targets(ex)}
    elif isinstance(ex, dict):
        if any(k.startswith(".") for k in ex):
            facts["exports"] = {k: _targets(v) for k, v in ex.items() if k.startswith(".")}
        else:
            facts["exports"] = {".": _targets(ex)}
    seen = set()
    for key in _DEP_KEYS:
        deps = data.get(key)
        if not isinstance(deps, dict):
            continue
        for name in deps:
            if name in seen or not isinstance(name, str):
                continue
            seen.add(name)
            m = re.search(r'"%s"\s*:' % re.escape(name), text)
            facts["deps"].append([name, text.count("\n", 0, m.start()) + 1 if m else 1])
    return facts


# ---- resolution --------------------------------------------------------------

class TsModules:
    def __init__(self, facts: Dict[str, dict], files: Iterable[str]):
        self.files = set(files)  # type: Set[str]
        self.configs = {rel: f for rel, f in facts.items() if f.get("lang") == "tsconfig"}
        self.cfg_dirs = defaultdict(list)  # type: Dict[str, List[str]]
        for rel in sorted(self.configs):
            self.cfg_dirs[_dirname(rel)].append(rel)
        self.packages = {}  # type: Dict[str, Tuple[str, dict]]   name -> (dir, facts)
        for rel, f in sorted(facts.items()):
            if f.get("lang") == "npm" and f.get("name"):
                self.packages.setdefault(f["name"], (_dirname(rel), f))
        self._by_len = sorted(self.packages, key=len, reverse=True)
        self._cache = {}  # type: Dict[Tuple[str, str], Optional[str]]
        self._dir_cfg = {}  # type: Dict[str, Optional[tuple]]
        self._eff = {}  # type: Dict[str, tuple]

    # -- files --

    def file(self, base: Optional[str]) -> Optional[str]:
        """The source file a path (without or with an extension) names."""
        if not base or base == ".":
            return None
        files = self.files
        if base in files:
            return base
        low = base.lower()
        if low.endswith(_JS_EXTS):  # import './x.js' from TypeScript means x.ts
            stem = base[:base.rfind(".")]
            for e in (".ts", ".tsx", ".mts", ".cts", ".d.ts"):
                if stem + e in files:
                    return stem + e
        for e in _LOOKUP_EXTS:
            if base + e in files:
                return base + e
        for e in _LOOKUP_EXTS:
            if base + "/index" + e in files:
                return base + "/index" + e
        return None

    # -- tsconfig --

    def _effective(self, rel: str, seen: Set[str]) -> tuple:
        """(baseUrl dir or None, paths or None, dir that paths are relative to)."""
        if rel in self._eff:
            return self._eff[rel]
        seen = seen | {rel}
        f = self.configs[rel]
        cdir = _dirname(rel)
        base_url, paths, paths_dir = None, None, cdir
        for ext in f.get("extends", ()):
            if not ext.startswith("."):
                continue
            p = _join(cdir, ext)
            if p and not p.endswith(".json"):
                p += ".json"
            if p in self.configs and p not in seen:
                pb, pp, pd = self._effective(p, seen)
                if pb is not None:
                    base_url = pb
                if pp is not None:
                    paths, paths_dir = pp, pd
        if f.get("baseUrl") is not None:
            base_url = _join(cdir, f["baseUrl"])
        if f.get("paths") is not None:
            paths, paths_dir = f["paths"], cdir
        res = (base_url, paths, paths_dir)
        self._eff[rel] = res
        return res

    def _config_for(self, d: str) -> Optional[tuple]:
        if d in self._dir_cfg:
            return self._dir_cfg[d]
        res = None
        cands = self.cfg_dirs.get(d)
        if cands:
            def order(rel):
                base = rel.rsplit("/", 1)[-1]
                return (base not in ("tsconfig.json", "jsconfig.json"), "app" not in base, base)
            effs = [self._effective(r, set()) for r in sorted(cands, key=order)]
            res = next((e for e in effs if e[1] is not None or e[0] is not None), effs[0])
        elif d != ".":
            res = self._config_for(_dirname(d))
        self._dir_cfg[d] = res
        return res

    def _paths(self, spec: str, cfg: tuple) -> Optional[str]:
        base_url, paths, paths_dir = cfg
        best, best_score, best_match = None, -1, ""
        for pat, targets in (paths or {}).items():
            if "*" in pat:
                pre, suf = pat.split("*", 1)
                if spec.startswith(pre) and spec.endswith(suf) and len(spec) >= len(pre) + len(suf):
                    if len(pre) > best_score:
                        best, best_score, best_match = targets, len(pre), \
                            spec[len(pre):len(spec) - len(suf)]
            elif pat == spec:
                best, best_score, best_match = targets, 10 ** 6, ""
        if best is None:
            return None
        root = base_url if base_url is not None else paths_dir
        for tgt in best:
            r = self.file(_join(root, tgt.replace("*", best_match)))
            if r:
                return r
        return None

    # -- workspace packages --

    def _package(self, spec: str) -> Optional[str]:
        name = next((n for n in self._by_len if spec == n or spec.startswith(n + "/")), None)
        if name is None:
            return None
        d, f = self.packages[name]
        sub = spec[len(name):].lstrip("/")
        exports = f.get("exports") or {}
        cands = []  # type: List[str]
        if sub:
            cands += exports.get("./" + sub, [])
            cands += [sub, "src/" + sub]
        else:
            cands += exports.get(".", [])
            cands += [f.get("types", ""), f.get("module", ""), f.get("main", ""), "src/index", "index"]
        for c in cands:
            if not c or "*" in c:
                continue
            c = c[2:] if c.startswith("./") else c
            variants = [c]
            for b in _BUILD_DIRS:
                if c.startswith(b):
                    variants.append("src/" + strip_ext(c[len(b):]))
            for v in variants:
                r = self.file(_join(d, v))
                if r:
                    return r
        return None

    def package_of(self, spec: str) -> Optional[str]:
        """Workspace package name a bare specifier refers to, if any."""
        return next((n for n in self._by_len if spec == n or spec.startswith(n + "/")), None)

    # -- entry point --

    def resolve(self, from_rel: str, spec: str) -> Optional[str]:
        if not spec:
            return None
        spec = spec.split("?", 1)[0].split("#", 1)[0] if not spec.startswith("#") else spec
        d = _dirname(from_rel)
        key = (d, spec)
        if key in self._cache:
            return self._cache[key]
        r = None
        if spec.startswith("."):
            r = self.file(_join(d, spec))
        elif spec.startswith("/"):
            r = self.file(_norm(spec))
        elif not spec.startswith("node:"):
            cfg = self._config_for(d)
            if cfg is not None:
                r = self._paths(spec, cfg)
                if r is None and cfg[0] is not None:
                    r = self.file(_join(cfg[0], spec))
            if r is None:
                r = self._package(spec)
        self._cache[key] = r
        return r
