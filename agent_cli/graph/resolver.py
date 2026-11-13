"""Turn per-file facts into a resolved CodeGraph (Kotlin and Swift).

Name resolution follows the language's scoping, in order, and records how sure
it is (for Swift, "package" means the module: Gradle-less projects get their
modules from Package.swift targets and Xcode targets, and `import Foo` makes
module Foo visible like a star import):
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

Swift specifics: `extension Foo` adds its members (and conformances) to Foo;
an extension of an external type (String, UIView) behaves like a Kotlin
extension function on it. Overloads are told apart by argument labels;
methods that implement a protocol requirement get an inferred `overrides`.
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

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
Void AnyObject Never Bool Character Substring Dictionary Optional Data Date URL UUID
CGFloat CGRect CGPoint CGSize NSObject NSString Decimal Task MainActor Error
View Text Image Color Button VStack HStack ZStack LazyVStack LazyHStack ScrollView
NavigationView NavigationStack NavigationLink Spacer Divider Label Toggle TextField
SecureField Picker Form Section ForEach GeometryReader Group EmptyView AnyView
UIView UIViewController UIColor UIImage UILabel UIButton UITableView UICollectionView
UINavigationController UIStoryboard UINib UIApplication UIScreen UIFont NSLayoutConstraint
ObservableObject Published State Binding StateObject ObservedObject EnvironmentObject
Identifiable Codable Decodable Encodable Hashable Equatable Comparable CaseIterable
Sendable CustomStringConvertible DispatchQueue NotificationCenter UserDefaults
JSONDecoder JSONEncoder URLSession URLRequest Bundle FileManager DateFormatter
XCTestCase
""".split())
# Swift property wrappers / attributes that mean "this dependency is provided".
SWIFT_INJECT_WRAPPERS = frozenset({
    "Inject", "Injected", "LazyInjected", "WeakLazyInjected", "OptionalInjected",
    "InjectedObject", "Dependency", "EnvironmentObject", "StateObject", "ObservedObject",
    "Environment", "Autowired",
})
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
append insert removeFirst removeLast removeValue compactMap reduce allSatisfy
sink store assign receive eraseToAnyPublisher dataTask decode encode dismiss present
pushViewController popViewController reloadData register dequeueReusableCell addSubview
removeFromSuperview layoutIfNeeded setNeedsLayout addTarget setTitle setImage padding
foregroundColor foregroundStyle font background frame overlay cornerRadius clipShape
onAppear onDisappear onTapGesture onChange onReceive task sheet alert fullScreenCover
navigationTitle navigationBarTitleDisplayMode toolbar listStyle buttonStyle
environmentObject environment tag opacity shadow offset animation transition
sync asyncAfter post addObserver removeObserver object bool integer string data
sorted joined lowercased uppercased trimmingCharacters components hasPrefix hasSuffix
first last prefix suffix dropFirst dropLast index enumerated reversed contains
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
    __slots__ = ("rel", "pkg", "imports", "stars", "module", "facts", "ids", "lang")

    def __init__(self, rel: str, facts: dict, module: str, pkg: Optional[str] = None):
        self.rel = rel
        self.facts = facts
        self.lang = facts.get("lang", "kotlin")
        self.pkg = facts.get("package", "") if pkg is None else pkg
        self.module = module
        self.imports = {}  # type: Dict[str, str]   simple name / alias -> fq
        self.stars = []  # type: List[str]
        for path, alias, star, _line in facts.get("imports", ()):
            if star:
                self.stars.append(path)
            else:
                self.imports[alias or path.rsplit(".", 1)[-1]] = path
        if self.lang == "swift":  # `import Foo` exposes module Foo; own module is visible
            self.stars = [x.split(".", 1)[0] for x in self.stars]
        self.ids = [None] * len(facts.get("decls", ()))  # type: List[Optional[str]]

    def owner(self, i: int) -> Optional[str]:
        """Node id of decl i ('' / None when it has no node)."""
        return self.ids[i] if 0 <= i < len(self.ids) and self.ids[i] else None


def _labels_sig(params) -> str:
    return "".join((p[6] if len(p) > 6 else p[0]) + ":" for p in params)


def _labels_ok(params, labels: List[str]) -> bool:
    """Do call-site argument labels fit a Swift parameter list? '_' is an
    unlabeled argument, '*' a trailing closure (fits any parameter)."""
    i, n = 0, len(params)
    for lab in labels:
        while i < n:
            p = params[i]
            plabel = p[6] if len(p) > 6 else p[0]
            if lab == "*" or lab == plabel:
                if not p[5]:  # a variadic param may take more args
                    i += 1
                break
            if p[3] or p[5]:  # has default / variadic: may be skipped
                i += 1
                continue
            return False
        else:
            return False
    return all(p[3] or p[5] for p in params[i:]) or (i < n and params[i][5] and
                                                     all(q[3] or q[5] for q in params[i + 1:]))


def _swift_sig(d: dict) -> str:
    k = d["kind"]
    mods = " ".join(m for m in d.get("mods", ()) if m in (
        "override", "static", "class", "private", "fileprivate", "public", "open", "final",
        "mutating", "lazy", "weak", "required", "convenience"))
    attrs = " ".join("@" + a for a in d.get("annos", ()) if a not in ("objc", "discardableResult",
                                                                       "available", "MainActor"))
    pre = " ".join(x for x in (attrs, mods) if x)
    pre = pre + " " if pre else ""

    def param(p):
        label, name = (p[6] if len(p) > 6 else p[0]), p[0]
        head = name if label == name else f"{label} {name}"
        return f"{head}: {p[1]}{'...' if p[5] else ''}{' = …' if p[3] else ''}"

    if k in ("fun", "init"):
        ps = ", ".join(param(p) for p in d.get("params", ()))
        recv = d.get("recv") or ""
        s = f"{pre}{'init' if k == 'init' else 'func ' + d['name']}({ps})"
        if recv:
            s = f"{pre}extension {recv}: " + s[len(pre):]
        if d.get("ret"):
            s += f" -> {d['ret']}"
    elif k in ("class", "struct", "enum", "protocol", "actor"):
        s = f"{pre}{k} {d['name']}"
        if d.get("supers"):
            s += ": " + ", ".join(x[0] for x in d["supers"])
    elif k == "prop":
        recv = d.get("recv") or ""
        s = f"{pre}var {d['name']}"
        if d.get("type") or d.get("itype"):
            s += f": {d.get('type') or d.get('itype')}"
        if recv:
            s = f"extension {recv}: " + s
    elif k == "typealias":
        s = f"typealias {d['name']} = {d.get('type', '')}"
    elif k == "entry":
        s = f"case {d['name']}"
    else:
        s = d["name"]
    return s if len(s) <= 180 else s[:177] + "..."


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
        self.modules = {}  # type: Dict[str, dict]    ":app" / "CoreKit" -> {"paths", "build_file", ...}
        self.file_module = {}  # type: Dict[str, str]   explicit membership (Xcode targets)
        self.dir_module = {}  # type: Dict[str, str]    folders owned by one Xcode target
        self._prefixes = []  # type: List[Tuple[str, str]]   (dir, module), deepest first
        self.ext_supers = defaultdict(list)  # type: Dict[str, List[tuple]]   Swift extension conformances
        self._pending_ext = []  # type: List[tuple]
        self._cur_labels = None  # type: Optional[List[str]]   labels of the call being resolved
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
        self._swift_modules()
        self._finish_modules()
        self._declare()
        self._declare_swift_extensions()
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
        for m in self.modules.values():
            m.setdefault("paths", [m["path"]] if m.get("path") else [])
            m.setdefault("deps", [])
            m.setdefault("line", 1 if m["build_file"] else 0)
        # Gradle depends_on (Swift modules add theirs in _swift_modules)
        norm = {re.sub(r"[-_]", "", n).lower(): n for n in self.modules}
        self._deps = defaultdict(set)  # type: Dict[str, Set[str]]
        for name, m in list(self.modules.items()):
            f = self.facts.get(m["build_file"]) if m["build_file"] else None
            if not f or f.get("lang") != "gradle":
                continue
            for conf_name, target, line in f.get("deps", ()):
                if target.startswith("~"):
                    tgt = norm.get(re.sub(r"[-_]", "", target[1:]).lower())
                    prov = INFERRED
                else:
                    tgt = target if target in self.modules else None
                    prov = EXTRACTED
                if tgt and tgt != name:
                    self._deps[name].add(tgt)
                    m["deps"].append((tgt, prov, line))

    def _swift_modules(self) -> None:
        """Modules from Package.swift targets and Xcode targets."""
        def norm(path: str) -> str:
            parts = []
            for seg in path.replace("\\", "/").split("/"):
                if seg in ("", "."):
                    continue
                if seg == ".." and parts:
                    parts.pop()
                else:
                    parts.append(seg)
            return "/".join(parts) or "."

        def module(name: str, build_file: str, line: int, kind: str) -> dict:
            return self.modules.setdefault(name, {"path": "", "paths": [], "build_file": build_file,
                                                  "namespace": "", "deps": [], "line": line,
                                                  "swift": True, "kind": kind})
        # app/library targets claim files before test targets do
        xcode = []
        for rel, f in sorted(self.facts.items()):
            lang = f.get("lang")
            if lang == "spm":
                pkg_dir = rel.rsplit("/", 1)[0] if "/" in rel else "."
                for name, path, kind, deps, line in f.get("targets", ()):
                    m = module(name, rel, line, kind)
                    full = norm(pkg_dir + "/" + path if pkg_dir != "." else path)
                    if full not in m["paths"]:
                        m["paths"].append(full)
                    m["deps"].extend((d, EXTRACTED, line) for d in deps)
            elif lang == "xcodeproj":
                xcode.extend((rel, t) for t in f.get("targets", ()))
        for rel, (name, kind, files, folders, deps, line) in sorted(
                xcode, key=lambda x: (x[1][1] == "test", x[0])):
            m = module(name, rel, line, kind)
            for fd in folders:
                if norm(fd) not in m["paths"]:
                    m["paths"].append(norm(fd))
            for fp in files:
                self.file_module.setdefault(norm(fp), name)
            m["deps"].extend((d, EXTRACTED, line) for d in deps)
        # Folders whose listed files all belong to one target: unlisted files
        # there (Info.plist, files not yet added to the project) belong to it too.
        owners = defaultdict(set)  # type: Dict[str, Set[str]]
        for fp, mod in self.file_module.items():
            d = fp.rsplit("/", 1)[0] if "/" in fp else "."
            while True:
                owners[d].add(mod)
                if d == ".":
                    break
                d = d.rsplit("/", 1)[0] if "/" in d else "."
        self.dir_module = {d: next(iter(ms)) for d, ms in owners.items() if len(ms) == 1}
        for name, m in self.modules.items():
            if m.get("swift"):
                for tgt, _prov, _line in m["deps"]:
                    if tgt in self.modules and tgt != name:
                        self._deps[name].add(tgt)

    def _finish_modules(self) -> None:
        for name, m in self.modules.items():
            self.g.add_node(Node("module:" + name, "module", name, file=m["build_file"],
                                 line=m.get("line", 0)))
            for path in m["paths"]:
                self._prefixes.append((path, name))
        self._prefixes.sort(key=lambda pm: -(0 if pm[0] == "." else pm[0].count("/") + 1))
        for name, m in self.modules.items():
            seen_dep = set()
            for tgt, prov, line in m["deps"]:
                if tgt in self.modules and tgt != name and tgt not in seen_dep:
                    seen_dep.add(tgt)
                    self.edge("module:" + name, "module:" + tgt, DEPENDS_ON, prov,
                              1.0 if prov == EXTRACTED else 0.9, m["build_file"], line,
                              dedupe=True)
        for name in self.modules:
            seen, stack = {name}, [name]
            while stack:
                for nxt in self._deps.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            self.reach[name] = seen

    def module_of(self, rel: str) -> str:
        """Explicit target membership (also via a containing folder reference
        like Assets.xcassets), else the deepest module directory."""
        if not self.modules:
            return ""
        if self.file_module:
            probe = rel
            while probe:
                if probe in self.file_module:
                    return self.file_module[probe]
                probe = probe.rsplit("/", 1)[0] if "/" in probe else ""
        for path, name in self._prefixes:
            if path == "." or rel == path or rel.startswith(path + "/"):
                return name
        probe = rel.rsplit("/", 1)[0] if "/" in rel else ""
        while probe:
            if probe in self.dir_module:
                return self.dir_module[probe]
            probe = probe.rsplit("/", 1)[0] if "/" in probe else ""
        return ""

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
                ctx = _FileCtx(rel, f, module)
                self.files[rel] = ctx
                for i, d in enumerate(f.get("decls", ())):
                    self._declare_decl(ctx, i, d, ctx.owner(d["parent"]))
            elif lang == "swift":
                self._declare_swift(rel, f, module)
            elif lang == "res":
                self._declare_res(rel, f, module)
            elif lang in ("xcasset", "strings", "xcstrings", "ib"):
                self._declare_ios_res(rel, f, module)

    def _declare_swift(self, rel: str, f: dict, module: str) -> None:
        """Pass 1: everything except extension contents (their target types
        may live in files not declared yet)."""
        swift_module = module if module and not module.startswith(":") else ""
        ctx = _FileCtx(rel, f, module, pkg=swift_module)
        if swift_module:
            ctx.stars.append(swift_module)
        self.files[rel] = ctx
        decls = f.get("decls", ())
        in_ext = [False] * len(decls)
        for i, d in enumerate(decls):
            p = d["parent"]
            in_ext[i] = d["kind"] == "extension" or (0 <= p < i and in_ext[p])
            if not in_ext[i]:
                self._declare_decl(ctx, i, d, ctx.owner(p))
        if any(in_ext):
            self._pending_ext.append((ctx, in_ext))

    def _declare_swift_extensions(self) -> None:
        """Pass 2: attach extension members to their (now declared) types;
        members of extensions on external types become receiver-typed
        top-level declarations, like Kotlin extension functions."""
        ext_recv = {}  # type: Dict[Tuple[str, int], str]
        for ctx, in_ext in self._pending_ext:
            decls = ctx.facts["decls"]
            for i, d in enumerate(decls):
                if not in_ext[i]:
                    continue
                if d["kind"] == "extension":
                    res = self.resolve_type(d["name"], ctx, [])
                    if res is not None and res.prov != AMBIGUOUS:
                        ctx.ids[i] = res.targets[0]
                        for text, _a, line in d.get("supers", ()):
                            self.ext_supers[res.targets[0]].append((text, line, ctx))
                    else:
                        ctx.ids[i] = ""
                        ext_recv[(ctx.rel, i)] = d["name"]
                    continue
                p = d["parent"]
                if (ctx.rel, p) in ext_recv:
                    recv = ext_recv[(ctx.rel, p)]
                    d = dict(d)
                    if d["kind"] in ("fun", "prop"):
                        d["recv"] = recv
                    simple = _head(recv).rsplit(".", 1)[-1]
                    base = f"{ctx.pkg}.{simple}" if ctx.pkg else simple
                    self._declare_decl(ctx, i, d, None, base=base)
                else:
                    self._declare_decl(ctx, i, d, ctx.owner(p))
        self._type_cache.clear()

    def _declare_decl(self, ctx: _FileCtx, i: int, d: dict, parent_id: Optional[str],
                      base: Optional[str] = None) -> None:
        g = self.g
        rel, pkg, module = ctx.rel, ctx.pkg, ctx.module
        swift = ctx.lang == "swift"
        if base is None:
            base = parent_id.split("#", 1)[0] if parent_id else pkg
        fq = f"{base}.{d['name']}" if base else d["name"]
        nid = fq
        if swift and d["kind"] in ("fun", "init"):
            nid = f"{fq}({_labels_sig(d.get('params', ()))})"
        if nid in g.nodes:
            stem = nid
            nid = f"{stem}#{d['line']}"
            k = 2
            while nid in g.nodes:
                nid = f"{stem}#{d['line']}.{k}"
                k += 1
        kind = _KIND.get(d["kind"], d["kind"])
        node = g.add_node(Node(nid, kind, d["name"], rel, d["line"], d.get("end", d["line"]),
                               parent_id, _swift_sig(d) if swift else _sig(d), module))
        node.extra["decl"] = d
        node.extra["pkg"] = pkg
        node.extra["lang"] = ctx.lang
        ctx.ids[i] = nid
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
        return res_node

    def _declare_ios_res(self, rel: str, f: dict, module: str) -> None:
        """Asset-catalog entries, .strings / String Catalog keys, storyboards
        and nibs (+ their storyboard identifiers)."""
        lang = f.get("lang")
        res_node = self._declare_res(rel, {}, module)
        if lang == "xcasset" and f.get("rname"):
            res_node(f["rtype"], f["rname"], 1)
        elif lang in ("strings", "xcstrings"):
            for key, line in f.get("keys", ()):
                res_node("string", key, line)
        elif lang == "ib":
            res_node(f["rtype"], f["rname"], 1)
            for ident, line in f.get("ids", ()):
                res_node("vcid", ident, line)

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
            if ctx.lang == "swift":
                for path, alias, star, line in ctx.facts.get("imports", ()):
                    mod = path.split(".", 1)[0]
                    if "module:" + mod in self.g.nodes and mod != ctx.pkg:
                        self.edge("file:" + rel, "module:" + mod, IMPORTS, EXTRACTED, 1.0, rel,
                                  line, dedupe=True)
                    if star:
                        continue
                    if path in self.fq_class:
                        self.edge("file:" + rel, self.fq_class[path], IMPORTS, EXTRACTED, 1.0,
                                  rel, line)
                continue
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
            for text, line, ectx in self.ext_supers.get(nid, ()):
                res = self.resolve_type(text, ectx, [])
                if res is None or res.targets[0] == nid:
                    continue
                sups.append((res.targets[0], res.prov, res.conf))
                self.edge_res(nid, res, INHERITS, ectx.rel, line, dedupe=True)
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
        function's return type, a class itself (or an init's class)."""
        node = self.g.nodes[nid]
        if node.kind in CLASS_KINDS:
            return _Res([nid], EXTRACTED, 1.0)
        if node.kind == "constructor":
            return _Res([node.parent], EXTRACTED, 1.0) if node.parent else None
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
        if nargs < 0 or node.kind not in ("function", "constructor"):
            return True
        params = self.decl(nid).get("params", ())
        if node.extra.get("lang") == "swift":
            if self._cur_labels is None:
                return True
            return _labels_ok(params, self._cur_labels)
        if node.kind != "function":
            return True
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
        return [i for i in ids if self.g.nodes[i].kind in ("function", "constructor") or
                self.g.nodes[i].kind in CLASS_KINDS]

    def _ext_visible(self, cand: str, scope: List[str]) -> bool:
        """A receiver-typed (extension) declaration can be called without a
        receiver only from code whose implicit `this`/`self` has that type."""
        recv = self.decl(cand).get("recv")
        if not recv:
            return True
        want = _head(recv).rsplit(".", 1)[-1]
        nodes = self.g.nodes
        for sid in scope:
            d = self.decl(sid)
            if d.get("recv") and _head(d["recv"]).rsplit(".", 1)[-1] == want:
                return True
            if nodes[sid].kind in SCOPE_KINDS and want in {nodes[a].name for a in self.ancestors(sid)}:
                return True
        return False

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
        hits = [h for h in self._callables(self._visible_top(name, ctx))
                if ctx.lang != "swift" or self._ext_visible(h, scope)]
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
        if s0 == "Self":
            cls = next((sid for sid in scope if nodes[sid].kind in SCOPE_KINDS), None)
            if cls is None:
                return None
            cur, prov, conf = cls, EXTRACTED, 1.0
        elif s0 in ("this", "super"):
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
                hits = [h for h in hits if nodes[h].kind in ("function", "constructor")
                        or nodes[h].kind in CLASS_KINDS]
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
            for call in ctx.facts.get("calls", ()):
                owner_i, name, recv, line, nargs, is_ref = call[:6]
                owner = ctx.owner(owner_i)
                if owner is None:
                    continue
                self._cur_labels = call[6] if len(call) > 6 else None
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
        self._cur_labels = None

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
                annos = d.get("annos", ())
                injected = "Inject" in annos or d.get("delegate") in INJECT_DELEGATES or (
                    node.extra.get("lang") == "swift" and any(a in SWIFT_INJECT_WRAPPERS for a in annos))
                if injected and not t:
                    t = d.get("itype", "")
                if injected and t:
                    holder = next((s for s in scope[1:] if nodes[s].kind in SCOPE_KINDS), None)
                    res = self.resolve_type(t, ctx, scope)
                    if res and holder:
                        self.edge_res(holder, res, INJECTS, node.file, node.line, dedupe=True)
                self._uses_types(nid, t, ctx, scope, node.line)
                if d.get("recv"):
                    self._uses_types(nid, d["recv"], ctx, scope, node.line)
            elif node.kind in ("function", "constructor"):
                for p in d.get("params", ()):
                    self._uses_types(nid, p[1], ctx, scope, node.line)
                self._uses_types(nid, d.get("ret", ""), ctx, scope, node.line)
                if d.get("recv"):
                    self._uses_types(nid, d["recv"], ctx, scope, node.line)
            for loc_type, _how in d.get("locals", {}).values():
                self._uses_types(nid, loc_type, ctx, scope, node.line)
        for rel, ctx in self.files.items():
            for owner_i, name, line, how in ctx.facts.get("trefs", ()):
                owner = ctx.owner(owner_i)
                if owner is None:
                    continue
                scope = self.scope(owner)
                res = self.resolve_type(name, ctx, scope)
                if not res:
                    continue
                self.edge_res(owner, res, USES_TYPE, rel, line, dedupe=True)
                if how == "inject":  # container.resolve(Foo.self)
                    holder = next((s for s in scope if nodes[s].kind in SCOPE_KINDS), owner)
                    self.edge_res(holder, res, INJECTS, rel, line, dedupe=True)

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
            parent = node.parent
            if parent is None:
                continue
            pnode = nodes[parent]
            swift = node.extra.get("lang") == "swift"
            if "override" not in d.get("mods", ()):
                if swift and pnode.kind in CLASS_KINDS and pnode.kind != "protocol":
                    self._protocol_witness(nid, node, parent)
                continue
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
                    if swift:
                        sig = _labels_sig(d.get("params", ()))
                        same = [h for h in hits if _labels_sig(self.decl(h).get("params", ())) == sig]
                    else:
                        same = [h for h in hits if len(self.decl(h).get("params", ())) == nargs]
                    hits = same or hits
                if hits:
                    self.edge(nid, hits[0], OVERRIDES, prov if prov != AMBIGUOUS else INFERRED,
                              conf, node.file, node.line, hits)
                    break

    def _protocol_witness(self, nid: str, node: Node, parent: str) -> None:
        """Swift: a member implementing a protocol requirement (no `override`
        keyword) -> inferred overrides edge to the requirement."""
        nodes = self.g.nodes
        d = self.decl(nid)
        sig = _labels_sig(d.get("params", ())) if node.kind == "function" else None
        for proto in self.ancestors(parent)[1:]:
            if nodes[proto].kind != "protocol":
                continue
            for h in self.members.get(proto, {}).get(node.name, ()):
                hn = nodes[h]
                if hn.kind != node.kind or hn.file == node.file and hn.parent == parent:
                    continue
                if sig is not None and _labels_sig(self.decl(h).get("params", ())) != sig:
                    continue
                self.edge(nid, h, OVERRIDES, INFERRED, 0.9, node.file, node.line, dedupe=True)
                return

    # ---------------------------------------------------------------- resources

    def _resolve_class_name(self, name: str, module: str) -> Tuple[Optional[str], str]:
        """'App.LoginViewController' or 'LoginViewController' -> (node id, provenance)."""
        if name in self.fq_class:
            return self.fq_class[name], EXTRACTED
        simple = name.rsplit(".", 1)[-1]
        if module and f"{module}.{simple}" in self.fq_class:
            return self.fq_class[f"{module}.{simple}"], EXTRACTED
        cands = self._reachable_filter(self.classes_by_name.get(simple, []), module)
        if len(cands) == 1:
            return cands[0], INFERRED
        return None, ""


    def _ios_references(self) -> None:
        """Storyboards / xibs -> customClass classes and @IBAction methods;
        Info.plist -> $(PRODUCT_MODULE_NAME).Class."""
        nodes = self.g.nodes
        for rel, f in self.facts.items():
            lang = f.get("lang")
            if lang not in ("ib", "plist"):
                continue
            module = self.module_of(rel)
            src = f"res:{f['rtype']}/{f['rname']}" if lang == "ib" else "file:" + rel
            if src not in nodes:
                src = "file:" + rel
            classes = []
            for name, line in f.get("classes", ()):
                cid, prov = self._resolve_class_name(name, module)
                if cid:
                    classes.append(cid)
                    self.edge(src, cid, REFERENCES, prov, 1.0 if prov == EXTRACTED else 0.7, rel, line,
                              dedupe=True)
            for selector, line in f.get("actions", ()):
                for cid in classes:
                    hits = [h for h in self.find_member(cid, selector) if nodes[h].kind == "function"]
                    if hits:
                        self.edge(src, hits[0], REFERENCES, INFERRED, 0.8, rel, line, dedupe=True)
                        break


    def _references(self) -> None:
        nodes = self.g.nodes
        for rel, ctx in self.files.items():
            for rr in ctx.facts.get("rrefs", ()):
                owner_i, rtype, rname, line = rr[:4]
                literal = len(rr) > 4 and rr[4] == "literal"
                owner = ctx.owner(owner_i)
                if owner is None:
                    continue
                rid = f"res:{rtype}/{rname}"
                if rid in nodes:
                    # a string literal that happens to match a resource name is
                    # very likely, not certainly, that resource
                    self.edge(owner, rid, REFERENCES, INFERRED if literal else EXTRACTED,
                              0.9 if literal else 1.0, rel, line)
            for owner_i, cls, line in ctx.facts.get("brefs", ()):
                if ctx.owner(owner_i):
                    rid = f"res:layout/{_snake(cls[:-len('Binding')])}"
                    if rid in nodes:
                        self.edge(ctx.ids[owner_i], rid, REFERENCES, INFERRED, 0.8, rel, line,
                                  dedupe=True)
        self._ios_references()
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
