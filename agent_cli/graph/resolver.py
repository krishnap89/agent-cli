"""Turn per-file facts into a resolved CodeGraph.

Name resolution follows Kotlin's scoping, in order, and records how sure it is:
  1. local declarations, enclosing classes (incl. inherited members), the
     file itself                                     -> extracted
  2. explicit imports (incl. aliases), same package, star imports -> extracted
  3. receiver types: locals / params / properties / constructor params with
     a declared type (or `val x = Foo()`), then member or extension lookup on
     that type, following chains like a.b().c        -> inferred
  4. a project-wide match on the bare name          -> inferred (one candidate)
                                                       ambiguous (several; kept)
Names that resolve to something outside the project (an explicit import of
an external class, a member of an external type, stdlib types) are left
unresolved rather than guessed.
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..android import map_file_to_module
from .kotlin_parser import _head
from .model import (AMBIGUOUS, CALLS, CLASS_KINDS, CODE_KINDS, CONTAINS, DEPENDS_ON, EXTRACTED,
                    IMPORTS, INFERRED, INHERITS, INJECTS, OVERRIDES, REFERENCES, USES_TYPE,
                    CodeGraph, Edge, Node)

_KIND = {"fun": "function", "prop": "property", "init": "constructor", "entry": "entry"}
SCOPE_KINDS = CLASS_KINDS | {"entry"}
# Kotlin's default imports that projects commonly shadow-match by accident.
BUILTIN_TYPES = frozenset("""
Any Nothing Unit String CharSequence Int Long Short Byte Float Double Boolean Char Number
Array IntArray LongArray ByteArray FloatArray DoubleArray BooleanArray CharArray
List MutableList ArrayList Map MutableMap HashMap LinkedHashMap Set MutableSet HashSet
Collection MutableCollection Iterable Iterator Sequence Pair Triple Lazy Result Comparable
Comparator Throwable Exception RuntimeException Error IllegalStateException
IllegalArgumentException UnsupportedOperationException IndexOutOfBoundsException
NullPointerException Regex StringBuilder Enum Annotation Function Deprecated Suppress
JvmStatic JvmOverloads JvmField
""".split())
INJECT_DELEGATES = frozenset({
    "inject", "viewModels", "activityViewModels", "viewModel", "navGraphViewModels",
    "hiltNavGraphViewModels", "sharedViewModel", "activityViewModel", "lazyInject",
})
# Member names so common on stdlib / coroutine / Android types that, with an
# unknown receiver, matching them to a project method is mostly wrong.
COMMON_MEMBER_NAMES = frozenset("""
map mapNotNull mapIndexed flatMap filter filterNot filterIsInstance forEach forEachIndexed
onEach let also apply run with takeIf takeUnless use first firstOrNull last lastOrNull
single singleOrNull find any all none count sum sumOf sortedBy sortedByDescending sorted
groupBy associate associateBy distinct distinctBy reversed take drop chunked windowed zip
plus minus contains containsKey indexOf joinToString toList toMutableList toSet
toMutableSet toMap toMutableMap toTypedArray orEmpty isEmpty isNotEmpty isNullOrEmpty
isBlank isNotBlank isNullOrBlank get set getOrNull getOrElse getOrDefault getOrPut put
putAll add addAll remove removeAll clear size copy equals hashCode toString compareTo
invoke trim split replace substring startsWith endsWith lowercase uppercase format
toInt toLong toDouble toFloat toBoolean toIntOrNull toLongOrNull toDoubleOrNull
launch async await collect collectLatest emit tryEmit stateIn shareIn combine update
catch flowOn debounce distinctUntilChanged filterNotNull observe postValue setValue
getValue value getString getInt getLong getBoolean putString putInt putExtra
getStringExtra getSerializable getParcelable findViewById inflate build execute
enqueue close start stop cancel resume dispose subscribe
""".split())
_TYPE_NAME_RE = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*")
_MAX_CANDS = 8


class _Res:
    """A resolution: targets (first is the edge dst), provenance, confidence."""
    __slots__ = ("targets", "prov", "conf")

    def __init__(self, targets: List[str], prov: str, conf: float):
        self.targets = targets
        self.prov = prov
        self.conf = conf


class _FileCtx:
    __slots__ = ("rel", "pkg", "imports", "stars", "module", "facts", "ids")

    def __init__(self, rel: str, facts: dict, module: str):
        self.rel = rel
        self.facts = facts
        self.pkg = facts.get("package", "")
        self.module = module
        self.imports = {}  # type: Dict[str, str]   simple name / alias -> fq
        self.stars = []  # type: List[str]
        for path, alias, star, _line in facts.get("imports", ()):
            if star:
                self.stars.append(path)
            else:
                self.imports[alias or path.rsplit(".", 1)[-1]] = path
        self.ids = []  # type: List[str]   decl index -> node id


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def _sig(d: dict) -> str:
    k = d["kind"]
    mods = " ".join(m for m in d.get("mods", ()) if m in (
        "override", "suspend", "abstract", "open", "data", "sealed", "inline", "private",
        "internal", "companion", "inner", "value", "lateinit", "const"))
    mods = mods + " " if mods else ""
    if k == "fun":
        ps = ", ".join(f"{'vararg ' if p[5] else ''}{p[0]}: {p[1]}{' = …' if p[3] else ''}"
                       for p in d.get("params", ()))
        recv = d.get("recv") or ""
        s = f"{mods}fun {recv + '.' if recv else ''}{d['name']}({ps})"
        if d.get("ret"):
            s += f": {d['ret']}"
    elif k in ("class", "interface", "object", "enum", "annotation"):
        word = {"enum": "enum class", "annotation": "annotation class"}.get(k, k)
        mods = mods.replace("companion ", "") if k != "object" else mods
        s = f"{mods}{word} {d['name']}"
        if d.get("params"):
            s += "(" + ", ".join(f"{'val ' if p[2] else ''}{p[0]}: {p[1]}" for p in d["params"]) + ")"
        if d.get("supers"):
            s += " : " + ", ".join(x[0] + ("()" if x[1] else "") for x in d["supers"])
    elif k == "prop":
        recv = d.get("recv") or ""
        s = f"{mods}val {recv + '.' if recv else ''}{d['name']}"
        if d.get("type"):
            s += f": {d['type']}"
        if d.get("delegate"):
            s += f" by {d['delegate']}"
    elif k == "typealias":
        s = f"typealias {d['name']} = {d.get('type', '')}"
    elif k == "init":
        s = "constructor / init"
    else:
        s = d["name"]
    return s if len(s) <= 180 else s[:177] + "..."


class Resolver:
    def __init__(self, facts: Dict[str, dict]):
        self.facts = facts
        self.g = CodeGraph()
        self.files = {}  # type: Dict[str, _FileCtx]
        self.modules = {}  # type: Dict[str, dict]    ":app" -> {"path", "build_file", "namespace"}
        self.reach = {}  # type: Dict[str, Set[str]]
        self.fq_class = {}  # type: Dict[str, str]
        self.fq_top = defaultdict(list)  # type: Dict[str, List[str]]
        self.members = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[str]]]
        self.top_by_file = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[str]]]
        self.top_by_pkg = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[str]]]
        self.classes_by_name = defaultdict(list)  # type: Dict[str, List[str]]
        self.funcs_by_name = defaultdict(list)  # type: Dict[str, List[str]]
        self.extensions = defaultdict(list)  # type: Dict[str, List[str]]
        self.supers = {}  # type: Dict[str, List[Tuple[str, str, float]]]
        self._type_cache = {}  # type: Dict[tuple, Optional[_Res]]
        self._scope_cache = {}  # type: Dict[str, List[str]]
        self._ancestor_cache = {}  # type: Dict[str, List[str]]
        self._edge_seen = set()  # type: Set[tuple]
        self.call_stats = defaultdict(int)  # type: Dict[str, int]
        self.unresolved_names = defaultdict(int)  # type: Dict[str, int]

    # ================================================================ build

    def run(self) -> CodeGraph:
        self._modules()
        self._declare()
        self._imports()
        self._inherits()
        self._calls()
        self._injects_and_types()
        self._overrides()
        self._references()
        self.g.stats["calls"] = dict(self.call_stats)
        self.g.stats["top_unresolved"] = sorted(
            self.unresolved_names.items(), key=lambda kv: -kv[1])[:15]
        return self.g

    def edge(self, src: str, dst: str, kind: str, prov: str = EXTRACTED, conf: float = 1.0,
             file: str = "", line: int = 0, cands: Optional[List[str]] = None,
             dedupe: bool = False) -> None:
        if src == dst and kind != CALLS:
            return
        if dedupe:
            key = (src, dst, kind)
            if key in self._edge_seen:
                return
            self._edge_seen.add(key)
        self.g.add_edge(Edge(src, dst, kind, prov, round(conf, 3), file, line,
                             cands[:_MAX_CANDS] if cands and len(cands) > 1 else None))

    def edge_res(self, src: str, res: _Res, kind: str, file: str, line: int,
                 dedupe: bool = False) -> None:
        self.edge(src, res.targets[0], kind, res.prov, res.conf, file, line, res.targets, dedupe)

    # ---------------------------------------------------------------- modules

    def _modules(self) -> None:
        for rel, f in self.facts.items():
            if f.get("lang") != "gradle":
                continue
            d = rel.rsplit("/", 1)[0] if "/" in rel else "."
            name = ":" if d == "." else ":" + d.replace("/", ":")
            base = rel.rsplit("/", 1)[-1]
            if base.startswith("settings."):
                for inc, _line in f.get("includes", ()):
                    path = inc.strip(":").replace(":", "/")
                    if d != ".":
                        path = d + "/" + path
                    self.modules.setdefault(inc if d == "." else ":" + path.replace("/", ":"),
                                            {"path": path, "build_file": "", "namespace": ""})
                continue
            m = self.modules.setdefault(name, {"path": d, "build_file": "", "namespace": ""})
            m["build_file"] = rel
            m["namespace"] = f.get("namespace", "") or m["namespace"]
        for name, m in self.modules.items():
            self.g.add_node(Node("module:" + name, "module", name, file=m["build_file"],
                                 line=1 if m["build_file"] else 0))
        # depends_on edges
        norm = {re.sub(r"[-_]", "", n).lower(): n for n in self.modules}
        deps = defaultdict(set)  # type: Dict[str, Set[str]]
        for name, m in self.modules.items():
            f = self.facts.get(m["build_file"]) if m["build_file"] else None
            if not f:
                continue
            for conf_name, target, line in f.get("deps", ()):
                if target.startswith("~"):
                    tgt = norm.get(re.sub(r"[-_]", "", target[1:]).lower())
                    prov = INFERRED
                else:
                    tgt = target if target in self.modules else None
                    prov = EXTRACTED
                if tgt and tgt != name:
                    deps[name].add(tgt)
                    self.edge("module:" + name, "module:" + tgt, DEPENDS_ON, prov,
                              1.0 if prov == EXTRACTED else 0.9, m["build_file"], line,
                              dedupe=True)
        for name in self.modules:
            seen, stack = {name}, [name]
            while stack:
                for nxt in deps.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            self.reach[name] = seen

    def module_of(self, rel: str) -> str:
        if not self.modules:
            return ""
        return map_file_to_module(rel, self.modules) or ""

    def _reachable_filter(self, cands: List[str], module: str) -> List[str]:
        if not module or module not in self.reach or len(cands) < 2:
            return cands
        ok = self.reach[module]
        kept = [c for c in cands if not self.g.nodes[c].module or self.g.nodes[c].module in ok]
        return kept or cands

    # ---------------------------------------------------------------- declare

    def _declare(self) -> None:
        g = self.g
        for rel in sorted(self.facts):
            f = self.facts[rel]
            lang = f.get("lang")
            module = self.module_of(rel)
            fnode = g.add_node(Node("file:" + rel, "file", rel, file=rel, module=module))
            if module and "module:" + module in g.nodes:
                self.edge("module:" + module, fnode.id, CONTAINS)
            if lang == "kotlin":
                self._declare_kotlin(rel, f, module)
            elif lang == "res":
                self._declare_res(rel, f, module)

    def _declare_kotlin(self, rel: str, f: dict, module: str) -> None:
        g = self.g
        ctx = _FileCtx(rel, f, module)
        self.files[rel] = ctx
        pkg = ctx.pkg
        for d in f.get("decls", ()):
            parent_id = ctx.ids[d["parent"]] if d["parent"] >= 0 else None
            base = parent_id.split("#", 1)[0] if parent_id else pkg
            fq = f"{base}.{d['name']}" if base else d["name"]
            nid = fq
            if nid in g.nodes:
                nid = f"{fq}#{d['line']}"
                k = 2
                while nid in g.nodes:
                    nid = f"{fq}#{d['line']}.{k}"
                    k += 1
            kind = _KIND.get(d["kind"], d["kind"])
            node = g.add_node(Node(nid, kind, d["name"], rel, d["line"], d.get("end", d["line"]),
                                   parent_id, _sig(d), module))
            node.extra["decl"] = d
            node.extra["pkg"] = pkg
            ctx.ids.append(nid)
            self.edge(parent_id or "file:" + rel, nid, CONTAINS, file=rel, line=d["line"])
            if kind in CLASS_KINDS:
                self.fq_class.setdefault(fq, nid)
                self.classes_by_name[d["name"]].append(nid)
            if kind == "function":
                self.funcs_by_name[d["name"]].append(nid)
            if kind in ("function", "property") and d.get("recv"):
                self.extensions[d["name"]].append(nid)
            if parent_id is None:
                self.top_by_file[rel][d["name"]].append(nid)
                self.top_by_pkg[pkg][d["name"]].append(nid)
                if kind not in CLASS_KINDS:
                    self.fq_top[fq].append(nid)
            else:
                self.members[parent_id][d["name"]].append(nid)

    def _declare_res(self, rel: str, f: dict, module: str) -> None:
        g = self.g
        fid = "file:" + rel

        def res_node(rtype: str, name: str, line: int) -> str:
            nid = f"res:{rtype}/{name}"
            node = g.nodes.get(nid)
            if node is None:
                node = g.add_node(Node(nid, "resource", name, rel, line, module=module))
                node.extra["rtype"] = rtype
                node.extra["files"] = []
            if rel not in node.extra["files"]:
                node.extra["files"].append(rel)
            self.edge(fid, nid, CONTAINS, file=rel, line=line, dedupe=True)
            return nid

        if f.get("rname"):
            res_node(f["rtype"], f["rname"], 1)
        for name, line in f.get("ids", ()):
            res_node("id", name, line)
        for rtype, name, line in f.get("values", ()):
            res_node(rtype, name, line)

    # ---------------------------------------------------------------- scopes

    def scope(self, nid: Optional[str]) -> List[str]:
        """[nid, parent, grandparent, ...] (declarations only)."""
        if not nid:
            return []
        s = self._scope_cache.get(nid)
        if s is None:
            node = self.g.nodes[nid]
            s = [nid] + self.scope(node.parent)
            self._scope_cache[nid] = s
        return s

    def decl(self, nid: str) -> dict:
        return self.g.nodes[nid].extra.get("decl", {})

    def ctx_of(self, nid: str) -> Optional[_FileCtx]:
        return self.files.get(self.g.nodes[nid].file)

    # ---------------------------------------------------------------- imports

    def _imports(self) -> None:
        for rel, ctx in self.files.items():
            for path, alias, star, line in ctx.facts.get("imports", ()):
                if star:
                    continue
                targets = [self.fq_class[path]] if path in self.fq_class else self.fq_top.get(path, [])
                for t in targets:
                    self.edge("file:" + rel, t, IMPORTS, EXTRACTED, 1.0, rel, line)

    # ---------------------------------------------------------------- types

    def _tparams_in_scope(self, scope: List[str]) -> Set[str]:
        out = set()  # type: Set[str]
        for sid in scope:
            out.update(self.decl(sid).get("tparams", ()))
        return out

    def resolve_type(self, text: str, ctx: Optional[_FileCtx], scope: List[str]) -> Optional[_Res]:
        head = _head(text)
        if not head or ctx is None:
            return None
        key = (ctx.rel, scope[0] if scope else None, head)
        if key in self._type_cache:
            return self._type_cache[key]
        res = self._resolve_type(head, ctx, scope)
        self._type_cache[key] = res
        return res

    def _resolve_type(self, head: str, ctx: _FileCtx, scope: List[str]) -> Optional[_Res]:
        parts = head.split(".")
        if len(parts) > 1:
            for k in range(len(parts), 0, -1):
                fq = ".".join(parts[:k])
                if fq in self.fq_class:
                    return self._descend(_Res([self.fq_class[fq]], EXTRACTED, 1.0), parts[k:])
        first = self._resolve_simple_type(parts[0], ctx, scope)
        if first is None or len(parts) == 1:
            return first
        return self._descend(first, parts[1:])

    def _descend(self, res: _Res, rest: List[str]) -> Optional[_Res]:
        cur = res.targets[0]
        for seg in rest:
            nxt = [m for m in self.members.get(cur, {}).get(seg, ())
                   if self.g.nodes[m].kind in SCOPE_KINDS]
            if not nxt:
                return None
            cur = nxt[0]
        return _Res([cur], res.prov, res.conf)

    def _resolve_simple_type(self, name: str, ctx: _FileCtx, scope: List[str]) -> Optional[_Res]:
        if name in self._tparams_in_scope(scope):
            return None
        nodes = self.g.nodes
        # 1. enclosing classes: themselves and their nested types (incl. inherited)
        for sid in scope:
            if nodes[sid].kind in SCOPE_KINDS:
                if nodes[sid].name == name and nodes[sid].kind in CLASS_KINDS:
                    return _Res([sid], EXTRACTED, 1.0)
                nested = [m for m in self.find_member(sid, name) if nodes[m].kind in CLASS_KINDS]
                if nested:
                    return _Res(nested[:1], EXTRACTED, 1.0)
        # 2. same file
        same = [i for i in self.top_by_file[ctx.rel].get(name, ()) if nodes[i].kind in CLASS_KINDS]
        if same:
            return _Res(same[:1], EXTRACTED, 1.0)
        # 3. explicit import (an import of something external stops the search)
        fq = ctx.imports.get(name)
        if fq is not None:
            if fq in self.fq_class:
                return _Res([self.fq_class[fq]], EXTRACTED, 1.0)
            return None
        # 4. same package, 5. star imports
        if (ctx.pkg + "." + name if ctx.pkg else name) in self.fq_class:
            return _Res([self.fq_class[ctx.pkg + "." + name if ctx.pkg else name]], EXTRACTED, 1.0)
        star = [self.fq_class[s + "." + name] for s in ctx.stars if s + "." + name in self.fq_class]
        if star:
            return _Res(star, EXTRACTED if len(star) == 1 else AMBIGUOUS, 1.0 / len(star))
        if name in BUILTIN_TYPES:
            return None
        # 6. project-wide by simple name (missing import, nested type, ...)
        cands = self._reachable_filter(self.classes_by_name.get(name, []), ctx.module)
        if not cands:
            return None
        if len(cands) == 1:
            return _Res(cands, INFERRED, 0.6)
        top = [c for c in cands if nodes[c].parent is None]
        if len(top) == 1:
            return _Res(top, INFERRED, 0.5)
        return _Res(cands, AMBIGUOUS, round(1.0 / len(cands), 3))

    def type_names(self, text: str) -> List[str]:
        """Every class-looking name in a type: 'Map<String, List<User>>' ->
        ['Map', 'String', 'List', 'User']."""
        out = []
        for m in _TYPE_NAME_RE.findall(text or ""):
            if m.rsplit(".", 1)[-1][:1].isupper():
                out.append(m)
        return out

    # ---------------------------------------------------------------- inheritance

    def _inherits(self) -> None:
        for nid, node in list(self.g.nodes.items()):
            if node.kind not in CLASS_KINDS:
                continue
            d = self.decl(nid)
            ctx = self.ctx_of(nid)
            # resolve supertypes from the class's own scope (nested types of the
            # enclosing class are visible; the class's own members are not yet)
            outer = self.scope(node.parent)
            sups = []
            for text, _has_args, line in d.get("supers", ()):
                res = self.resolve_type(text, ctx, outer)
                if res is None or res.targets[0] == nid:
                    continue
                sups.append((res.targets[0], res.prov, res.conf))
                self.edge_res(nid, res, INHERITS, node.file, line, dedupe=True)
            if d.get("kind") == "typealias" and d.get("type"):
                res = self.resolve_type(d["type"], ctx, outer)
                if res:
                    sups.append((res.targets[0], res.prov, res.conf))
            self.supers[nid] = sups

    def ancestors(self, cls: str) -> List[str]:
        """cls and its supertypes, breadth-first (resolved ones only)."""
        a = self._ancestor_cache.get(cls)
        if a is None:
            a, seen, queue = [], {cls}, [cls]
            while queue:
                c = queue.pop(0)
                a.append(c)
                for s, _p, _c in self.supers.get(c, ()):
                    if s not in seen:
                        seen.add(s)
                        queue.append(s)
            self._ancestor_cache[cls] = a
        return a

    def has_external_super(self, cls: str) -> bool:
        for c in self.ancestors(cls):
            if len(self.supers.get(c, ())) < len(self.decl(c).get("supers", ())):
                return True
        return False

    def find_member(self, cls: str, name: str, include_self: bool = True) -> List[str]:
        """Members named `name` of cls or the nearest supertype that has one.
        For typealiases, look in the aliased class."""
        for c in self.ancestors(cls):
            if c == cls and not include_self:
                continue
            hits = self.members.get(c, {}).get(name)
            if hits:
                return list(hits)
        return []

    def companion_member(self, cls: str, name: str) -> List[str]:
        for child in self.members.get(cls, {}).values():
            for c in child:
                n = self.g.nodes[c]
                if n.kind == "object" and "companion" in self.decl(c).get("mods", ()):
                    hits = self.find_member(c, name)
                    if hits:
                        return hits
        return []

    # ---------------------------------------------------------------- values

    def value_type(self, name: str, scope: List[str]) -> Optional[Tuple[str, str, float]]:
        """Declared type of a value visible from `scope`:
        (type_text, node id whose scope the text resolves in, confidence)."""
        nodes = self.g.nodes
        for sid in scope:
            node = nodes[sid]
            d = self.decl(sid)
            if node.kind in CODE_KINDS:
                loc = d.get("locals", {}).get(name)
                if loc:
                    return loc[0], sid, 0.9 if loc[1] == "decl" else 0.8
                for p in d.get("params", ()):
                    if p[0] == name and p[1]:
                        return p[1], sid, 0.9
                if node.kind == "constructor" and node.parent:
                    for p in self.decl(node.parent).get("params", ()):
                        if p[0] == name and p[1]:
                            return p[1], node.parent, 0.9
            elif node.kind in SCOPE_KINDS:
                for p in d.get("params", ()):
                    if p[0] == name and p[1]:
                        return p[1], sid, 0.9
                for m in self.find_member(sid, name):
                    md = self.decl(m)
                    if nodes[m].kind == "property":
                        t = md.get("type") or md.get("dtype")
                        if t:
                            return t, m, 0.9
                        return (md["itype"], m, 0.8) if md.get("itype") else None
                    if nodes[m].kind == "entry":
                        return nodes[sid].name, sid, 1.0
                for anc in self.ancestors(sid)[1:]:
                    for p in self.decl(anc).get("params", ()):
                        if p[0] == name and p[2] and p[1]:
                            return p[1], anc, 0.85
        ctx = self.ctx_of(scope[0]) if scope else None
        if ctx is not None:
            for nid in self._visible_top(name, ctx):
                if nodes[nid].kind == "property":
                    pd = self.decl(nid)
                    t = pd.get("type") or pd.get("itype")
                    return (t, nid, 0.9 if pd.get("type") else 0.8) if t else None
        return None

    def _visible_top(self, name: str, ctx: _FileCtx) -> List[str]:
        """Top-level declarations named `name` visible in a file."""
        hits = list(self.top_by_file[ctx.rel].get(name, ()))
        if hits:
            return hits
        fq = ctx.imports.get(name)
        if fq is not None:
            return list(self.fq_top.get(fq, ())) or ([self.fq_class[fq]] if fq in self.fq_class else [])
        hits = list(self.top_by_pkg[ctx.pkg].get(name, ()))
        if hits:
            return hits
        for s in ctx.stars:
            hits.extend(self.top_by_pkg.get(s, {}).get(name, ()))
        return hits

    def type_of_node(self, nid: str) -> Optional[_Res]:
        """The class a declaration evaluates to: a property's type, a
        function's return type, a class itself."""
        node = self.g.nodes[nid]
        if node.kind in CLASS_KINDS:
            return _Res([nid], EXTRACTED, 1.0)
        if node.kind == "entry":
            return _Res([node.parent], EXTRACTED, 1.0) if node.parent else None
        d = self.decl(nid)
        text = d.get("ret") if node.kind == "function" else \
            d.get("type") or d.get("dtype") or d.get("itype")
        if not text:
            return None
        return self.resolve_type(text, self.ctx_of(nid), self.scope(nid))

    # ---------------------------------------------------------------- calls

    def _arity_ok(self, nid: str, nargs: int) -> bool:
        node = self.g.nodes[nid]
        if nargs < 0 or node.kind != "function":
            return True
        params = self.decl(nid).get("params", ())
        if any(p[5] for p in params):
            return True
        required = sum(1 for p in params if not p[3])
        return required <= nargs <= len(params)

    def _pick(self, cands: List[str], nargs: int, prov: str, conf: float,
              module: str = "", strict: bool = False) -> Optional[_Res]:
        """Narrow candidates by arity and module reachability, then decide
        whether the result is unique, a set of overloads, or ambiguous.
        strict: for guesses, drop candidates whose arity does not fit instead
        of keeping them all (a scoped lookup keeps them: named/default args
        may defeat the count, but the scope is still right)."""
        if not cands:
            return None
        cands = list(dict.fromkeys(cands))
        fit = [c for c in cands if self._arity_ok(c, nargs)]
        if strict and not fit:
            return None
        cands = fit or cands
        cands = self._reachable_filter(cands, module)
        if len(cands) == 1:
            return _Res(cands, prov, conf)
        parents = {self.g.nodes[c].parent for c in cands}
        if len(parents) == 1:  # overloads in one scope: right place, unsure which
            return _Res(cands, prov, round(conf * 0.9, 3))
        return _Res(cands, AMBIGUOUS, round(min(conf, 1.0 / len(cands)), 3))

    def _callables(self, ids: List[str]) -> List[str]:
        return [i for i in ids if self.g.nodes[i].kind in ("function",) or
                self.g.nodes[i].kind in CLASS_KINDS]

    def resolve_bare(self, name: str, nargs: int, owner: str, ctx: _FileCtx) -> Optional[object]:
        """A call without a receiver. Returns _Res, "invoke" (calling a
        function-typed value), or None."""
        nodes = self.g.nodes
        scope = self.scope(owner)
        # local functions of the enclosing code declarations
        for sid in scope:
            if nodes[sid].kind not in CODE_KINDS:
                break
            hits = self._callables(self.members.get(sid, {}).get(name, []))
            if hits:
                return self._pick(hits, nargs, EXTRACTED, 1.0)
        # calling a function-typed parameter / local / property
        vt = self.value_type(name, scope)
        if vt is not None and ("->" in vt[0] or vt[0].startswith("(")):
            return "invoke"
        # members of enclosing classes, and of extension receivers in scope
        external_recv = False
        for sid in scope:
            node = nodes[sid]
            if node.kind in SCOPE_KINDS:
                hits = self._callables(self.find_member(sid, name))
                if hits:
                    return self._pick(hits, nargs, EXTRACTED, 1.0)
                hits = self._callables(self.companion_member(sid, name))
                if hits:
                    return self._pick(hits, nargs, EXTRACTED, 1.0)
            elif node.kind in CODE_KINDS and self.decl(sid).get("recv"):
                rt = self.resolve_type(self.decl(sid)["recv"], ctx, self.scope(node.parent))
                if rt:
                    hits = self._callables(self.find_member(rt.targets[0], name))
                    if hits:
                        return self._pick(hits, nargs, INFERRED, 0.9 * rt.conf)
                else:
                    external_recv = True
        # file, imports, package, star imports
        hits = self._callables(self._visible_top(name, ctx))
        if hits:
            return self._pick(hits, nargs, EXTRACTED, 1.0, ctx.module)
        if name[:1].isupper():
            rt = self.resolve_type(name, ctx, scope)
            if rt is not None:
                return rt
            if name in ctx.imports or name in BUILTIN_TYPES:
                return None
        elif name in ctx.imports:
            return None  # imported from outside the project
        if external_recv:
            return None  # inside `fun String.x()`, a bare call is most likely String's
        # project-wide fallback (scope functions like apply/with hide receivers)
        cands = self.funcs_by_name.get(name, []) + (
            self.classes_by_name.get(name, []) if name[:1].isupper() else [])
        if not cands:
            return None
        res = self._pick(cands, nargs, INFERRED, 0.5, ctx.module, strict=True)
        if res is not None and res.prov != AMBIGUOUS:
            res.conf = min(res.conf, 0.5)
        return res

    def eval_receiver(self, segs: List[list], owner: str, ctx: _FileCtx) -> Optional[tuple]:
        """Evaluate a receiver chain. Returns (kind, target, prov, conf):
          ("class", class_id, ...)   a project class (instance or static)
          ("super", class_id, ...)   super.<call> inside class_id
          ("ext", "String", ...)     an external type, by simple name
          ("pkg", "com.x.util", ...) a package-qualified call
        or None when unknown."""
        nodes = self.g.nodes
        scope = self.scope(owner)
        s0, is_call = segs[0]
        cur = None  # type: Optional[str]
        prov, conf = INFERRED, 0.9
        rest = segs[1:]
        if s0 in ("this", "super"):
            for sid in scope:
                d = self.decl(sid)
                if nodes[sid].kind in CODE_KINDS and d.get("recv") and s0 == "this":
                    rt = self.resolve_type(d["recv"], ctx, self.scope(nodes[sid].parent))
                    if rt is None:
                        return ("ext", _head(d["recv"]).rsplit(".", 1)[-1], INFERRED, 0.8) \
                            if not rest else None
                    cur, prov, conf = rt.targets[0], rt.prov, rt.conf
                    break
                if nodes[sid].kind in SCOPE_KINDS:
                    if s0 == "super" and not rest:
                        return ("super", sid, EXTRACTED, 1.0)
                    cur, prov, conf = sid, EXTRACTED, 1.0
                    break
            if cur is None:
                return None
        elif s0 in ('""', "#num"):
            lit = "String" if s0 == '""' else "Int"
            return ("ext", lit, INFERRED, 0.8) if not rest else None
        elif s0 == "?":
            return None
        elif is_call:
            r = self.resolve_bare(s0, -1, owner, ctx)
            if not isinstance(r, _Res) or r.prov == AMBIGUOUS:
                return None
            t = self.type_of_node(r.targets[0])
            if t is None:
                return None
            cur, prov, conf = t.targets[0], INFERRED, r.conf * t.conf * 0.9
        else:
            vt = self.value_type(s0, scope)
            if vt is not None:
                t = self.resolve_type(vt[0], self.ctx_of(vt[1]), self.scope(vt[1]))
                if t is None:
                    if not rest and not ("->" in vt[0]):
                        return ("ext", _head(vt[0]).rsplit(".", 1)[-1], INFERRED, vt[2] * 0.9)
                    return None
                cur, prov, conf = t.targets[0], INFERRED, vt[2] * t.conf
            else:
                names = [s for s, _c in segs]
                t = self.resolve_type(s0, ctx, scope) if s0[:1].isupper() else None
                if t is not None and t.prov != AMBIGUOUS:
                    cur, prov, conf = t.targets[0], t.prov, t.conf * 0.95
                else:
                    # fully qualified: com.example.Foo.bar() / com.example.util.helper()
                    for k in range(len(names), 0, -1):
                        fq = ".".join(names[:k])
                        if fq in self.fq_class:
                            cur, prov, conf = self.fq_class[fq], EXTRACTED, 1.0
                            rest = segs[k:]
                            break
                    else:
                        pkg = ".".join(names)
                        if pkg in self.top_by_pkg and not any(c for _s, c in segs):
                            return ("pkg", pkg, EXTRACTED, 1.0)
                        return None
        for seg, seg_call in rest:
            hits = self.find_member(cur, seg) or self.companion_member(cur, seg)
            if seg_call:
                hits = [h for h in hits if nodes[h].kind == "function" or nodes[h].kind in CLASS_KINDS]
            else:
                hits = [h for h in hits if nodes[h].kind != "function"]
            if not hits:
                return None
            t = self.type_of_node(hits[0])
            if t is None:
                return None
            cur = t.targets[0]
            prov = INFERRED if prov != EXTRACTED or nodes[hits[0]].kind != "object" else prov
            conf *= 0.9 * t.conf
        return ("class", cur, prov, conf)

    def resolve_member_call(self, recv: tuple, name: str, nargs: int, ctx: _FileCtx) -> Optional[_Res]:
        kind, target, prov, conf = recv
        nodes = self.g.nodes
        if kind == "pkg":
            hits = self._callables(self.top_by_pkg[target].get(name, []))
            return self._pick(hits, nargs, EXTRACTED, 1.0)
        if kind == "super":
            hits = self._callables(self.find_member(target, name, include_self=False))
            return self._pick(hits, nargs, EXTRACTED, 1.0)
        if kind == "ext":
            exts = [e for e in self.extensions.get(name, [])
                    if _head(self.decl(e).get("recv", "")).rsplit(".", 1)[-1] == target]
            return self._pick(exts, nargs, INFERRED, round(conf * 0.9, 3), ctx.module)
        hits = self._callables(self.find_member(target, name) or self.companion_member(target, name))
        if hits:
            return self._pick(hits, nargs, prov, round(conf, 3))
        # extension functions declared on the class or one of its supertypes
        anc = set(self.ancestors(target))
        names = {nodes[a].name for a in anc}
        exts = []
        for e in self.extensions.get(name, []):
            recv_text = self.decl(e).get("recv", "")
            simple = _head(recv_text).rsplit(".", 1)[-1]
            if simple not in names:
                continue
            rt = self.resolve_type(recv_text, self.ctx_of(e), self.scope(nodes[e].parent))
            if rt is None or rt.targets[0] in anc:
                exts.append(e)
        if exts:
            return self._pick(exts, nargs, INFERRED, round(conf * 0.9, 3), ctx.module)
        return None

    def resolve_unknown_receiver(self, name: str, nargs: int, ctx: _FileCtx) -> Optional[_Res]:
        """Receiver type unknown (lambda `it`, chained external calls...):
        match the name against project members and extensions, except for
        names that are everywhere in the standard library."""
        if name in COMMON_MEMBER_NAMES:
            self.call_stats["common_skipped"] += 1
            return None
        cands = [c for c in self.funcs_by_name.get(name, []) if self.g.nodes[c].parent is not None
                 or self.decl(c).get("recv")]
        if not cands:
            return None
        res = self._pick(cands, nargs, INFERRED, 0.4, ctx.module, strict=True)
        if res is not None and res.prov != AMBIGUOUS:
            res.conf = min(res.conf, 0.4)
        return res

    def _calls(self) -> None:
        st = self.call_stats
        for rel, ctx in self.files.items():
            ids = ctx.ids
            for owner_i, name, recv, line, nargs, is_ref in ctx.facts.get("calls", ()):
                if owner_i < 0 or owner_i >= len(ids):
                    continue
                owner = ids[owner_i]
                st["total"] += 1
                res = None  # type: Optional[object]
                external = False
                skipped = False
                if recv is None:
                    res = self.resolve_bare(name, nargs, owner, ctx)
                else:
                    r = self.eval_receiver(recv, owner, ctx)
                    if r is not None:
                        res = self.resolve_member_call(r, name, nargs, ctx)
                        if res is None and r[0] in ("class", "super", "ext", "pkg"):
                            # known receiver without such a member: an external
                            # (library / inherited-from-framework) call
                            external = True
                    elif recv[0][0] not in ('""', "#num"):
                        skipped = name in COMMON_MEMBER_NAMES
                        res = self.resolve_unknown_receiver(name, nargs, ctx)
                if res == "invoke":
                    st["invoke"] += 1
                    continue
                if res is None and recv is None:
                    fq = ctx.imports.get(name)
                    if (fq is not None and fq not in self.fq_class and fq not in self.fq_top) \
                            or name in BUILTIN_TYPES:
                        external = True  # explicitly imported from a library
                if isinstance(res, _Res):
                    st[res.prov] += 1
                    if name in self.funcs_by_name or name in self.classes_by_name:
                        st["internal_resolved"] += 1
                    self.edge_res(owner, res, CALLS, rel, line)
                else:
                    st["unresolved"] += 1
                    if external:
                        st["external"] += 1
                    elif skipped:
                        pass  # not guessed on purpose (counted in common_skipped)
                    elif name in self.funcs_by_name or name in self.classes_by_name:
                        st["internal_unresolved"] += 1
                        self.unresolved_names[name] += 1

    # ---------------------------------------------------------------- DI + types

    def _injects_and_types(self) -> None:
        nodes = self.g.nodes
        for nid, node in list(nodes.items()):
            d = node.extra.get("decl")
            if d is None:
                continue
            ctx = self.ctx_of(nid)
            scope = self.scope(nid)
            if node.kind in CLASS_KINDS:
                outer = self.scope(node.parent)
                for p in d.get("params", ()):
                    if not p[1]:
                        continue
                    if d.get("inject_ctor"):
                        res = self.resolve_type(p[1], ctx, scope)
                        if res:
                            self.edge_res(nid, res, INJECTS, node.file, node.line, dedupe=True)
                    self._uses_types(nid, p[1], ctx, scope, node.line)
                if node.kind == "typealias":
                    self._uses_types(nid, d.get("type", ""), ctx, outer, node.line)
            elif node.kind == "property":
                t = d.get("type") or d.get("dtype") or ""
                injected = "Inject" in d.get("annos", ()) or d.get("delegate") in INJECT_DELEGATES
                if injected and t:
                    holder = next((s for s in scope[1:] if nodes[s].kind in SCOPE_KINDS), None)
                    res = self.resolve_type(t, ctx, scope)
                    if res and holder:
                        self.edge_res(holder, res, INJECTS, node.file, node.line, dedupe=True)
                self._uses_types(nid, t, ctx, scope, node.line)
                if d.get("recv"):
                    self._uses_types(nid, d["recv"], ctx, scope, node.line)
            elif node.kind == "function":
                for p in d.get("params", ()):
                    self._uses_types(nid, p[1], ctx, scope, node.line)
                self._uses_types(nid, d.get("ret", ""), ctx, scope, node.line)
                if d.get("recv"):
                    self._uses_types(nid, d["recv"], ctx, scope, node.line)
            for loc_type, _how in d.get("locals", {}).values():
                self._uses_types(nid, loc_type, ctx, scope, node.line)
        for rel, ctx in self.files.items():
            for owner_i, name, line, _how in ctx.facts.get("trefs", ()):
                if 0 <= owner_i < len(ctx.ids):
                    owner = ctx.ids[owner_i]
                    res = self.resolve_type(name, ctx, self.scope(owner))
                    if res:
                        self.edge_res(owner, res, USES_TYPE, rel, line, dedupe=True)

    def _uses_types(self, src: str, text: str, ctx: Optional[_FileCtx], scope: List[str],
                    line: int) -> None:
        for name in self.type_names(text):
            res = self.resolve_type(name, ctx, scope)
            if res:
                self.edge_res(src, res, USES_TYPE, self.g.nodes[src].file, line, dedupe=True)
            elif name.endswith("Binding") and len(name) > 7:
                # generated ViewBinding class: ActivityMainBinding -> layout/activity_main
                rid = f"res:layout/{_snake(name.rsplit('.', 1)[-1][:-len('Binding')])}"
                if rid in self.g.nodes:
                    self.edge(src, rid, REFERENCES, INFERRED, 0.8, self.g.nodes[src].file, line,
                              dedupe=True)

    # ---------------------------------------------------------------- overrides

    def _overrides(self) -> None:
        nodes = self.g.nodes
        for nid, node in list(nodes.items()):
            if node.kind not in ("function", "property"):
                continue
            d = self.decl(nid)
            if "override" not in d.get("mods", ()):
                continue
            parent = node.parent
            if parent is None:
                continue
            pnode = nodes[parent]
            bases = []  # type: List[Tuple[str, str, float]]
            if pnode.kind == "entry":  # enum entry body overrides the enum's members
                bases = [(pnode.parent, EXTRACTED, 1.0)] if pnode.parent else []
            elif pnode.kind in CLASS_KINDS:
                bases = self.supers.get(parent, [])
            else:  # member of an anonymous object inside code
                ctx = self.ctx_of(nid)
                objs = [t for t in ctx.facts.get("trefs", ()) if t[3] == "obj"
                        and 0 <= t[0] < len(ctx.ids) and ctx.ids[t[0]] == parent
                        and t[2] <= node.line]
                if objs:
                    res = self.resolve_type(objs[-1][1], ctx, self.scope(parent))
                    if res:
                        bases = [(res.targets[0], res.prov, res.conf)]
            nargs = len(d.get("params", ()))
            for base, prov, conf in bases:
                hits = [h for h in self.find_member(base, node.name) if nodes[h].kind == node.kind]
                if node.kind == "function":
                    same = [h for h in hits if len(self.decl(h).get("params", ())) == nargs]
                    hits = same or hits
                if hits:
                    self.edge(nid, hits[0], OVERRIDES, prov if prov != AMBIGUOUS else INFERRED,
                              conf, node.file, node.line, hits)
                    break

    # ---------------------------------------------------------------- resources

    def _references(self) -> None:
        nodes = self.g.nodes
        for rel, ctx in self.files.items():
            for owner_i, rtype, rname, line in ctx.facts.get("rrefs", ()):
                if 0 <= owner_i < len(ctx.ids):
                    rid = f"res:{rtype}/{rname}"
                    if rid in nodes:
                        self.edge(ctx.ids[owner_i], rid, REFERENCES, EXTRACTED, 1.0, rel, line)
            for owner_i, cls, line in ctx.facts.get("brefs", ()):
                if 0 <= owner_i < len(ctx.ids):
                    rid = f"res:layout/{_snake(cls[:-len('Binding')])}"
                    if rid in nodes:
                        self.edge(ctx.ids[owner_i], rid, REFERENCES, INFERRED, 0.8, rel, line,
                                  dedupe=True)
        # XML -> classes (custom views, tools:context, fragments, manifest components)
        for rel, f in self.facts.items():
            lang = f.get("lang")
            if lang not in ("res", "manifest"):
                continue
            module = self.module_of(rel)
            ns = (self.modules.get(module, {}).get("namespace") or f.get("package", "")) \
                if module or f.get("package") else ""
            if lang == "res" and f.get("rname"):
                src = f"res:{f['rtype']}/{f['rname']}"
            else:
                src = "file:" + rel
            for name, line in f.get("classes", ()):
                fq = (ns + name) if name.startswith(".") else name
                fq = fq.replace("$", ".")
                if fq in self.fq_class:
                    self.edge(src, self.fq_class[fq], REFERENCES, EXTRACTED, 1.0, rel, line,
                              dedupe=True)
                    continue
                simple = fq.rsplit(".", 1)[-1]
                cands = self._reachable_filter(self.classes_by_name.get(simple, []), module)
                if len(cands) == 1:
                    self.edge(src, cands[0], REFERENCES, INFERRED, 0.6, rel, line, dedupe=True)


def resolve(facts: Dict[str, dict]) -> CodeGraph:
    return Resolver(facts).run()
