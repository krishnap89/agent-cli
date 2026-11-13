"""Kotlin declaration + call-site extractor for the code graph.

A hand-written recursive-descent pass over the tokens from kotlin_lexer. It is
not a full Kotlin parser: it understands declarations exactly (package,
imports, classes/objects/interfaces/enums, primary-constructor params,
supertypes, functions incl. extension receivers, properties, accessors, init
blocks, enum entries) and scans expression code for call sites, without
building expression trees.

parse_kotlin(source) returns a JSON-serializable dict ("facts"):

  package  "com.example.ui"
  imports  [[path, alias, is_star, line]]
  decls    [decl, ...]  flat; decl["parent"] is an index into decls or -1
  calls    [[owner, name, recv, line, nargs, is_ref]]
             owner  index of the enclosing decl (fun/prop/init)
             recv   None for a bare call, else [[segment, is_call], ...]
                    e.g. repo.load().items.map(...) -> map with recv
                    [["repo",0],["load",1],["items",0]]; [["?",0]] = unknown
             nargs  argument count incl. a trailing lambda (-1 for ::refs)
  trefs    [[owner, TypeName, line, how]]   how: is|as|qual|obj|cast
  rrefs    [[owner, res_type, res_name, line]]   R.layout.main etc.
  brefs    [[owner, BindingClassName, line]]     ViewBinding classes

A decl is a dict with kind in
  class interface object enum annotation typealias fun prop init entry
and name, parent, line, end, mods, annos plus kind-specific keys:
  class-like: tparams, params (primary ctor), supers [[Type, has_call_args, line]],
              inject_ctor
  fun:        tparams, recv, params, ret
  prop:       recv, type, delegate, has_init, [itype: type of a `= Foo()`
              initializer], [dtype: type argument of `by inject<Foo>()`]
  any owner of code may get locals {name: [type, how]} (how: decl|ctor).
params are [name, type, is_property, has_default, annos, is_vararg].
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from .kotlin_lexer import ID, NUM, OP, STR, Token, tokenize

MODIFIERS = frozenset({
    "public", "private", "protected", "internal", "open", "final", "abstract",
    "sealed", "data", "inner", "enum", "annotation", "companion", "override",
    "lateinit", "const", "inline", "noinline", "crossinline", "suspend",
    "tailrec", "operator", "infix", "external", "actual", "expect", "value",
    "vararg", "reified",
})
# Modifiers worth keeping in the facts (the rest only matter to the compiler).
_KEEP_MODS = frozenset({
    "private", "protected", "internal", "open", "abstract", "sealed", "data",
    "enum", "annotation", "companion", "override", "lateinit", "const",
    "inline", "suspend", "inner", "value", "operator", "infix",
})
DECL_KEYWORDS = frozenset({
    "class", "interface", "object", "fun", "val", "var", "typealias", "init",
    "constructor", "package", "import",
})
NOT_CALLS = frozenset({
    "if", "when", "while", "for", "catch", "return", "throw", "fun", "object",
    "class", "interface", "try", "else", "do", "in", "is", "as", "super",
    "this", "constructor", "init", "val", "var", "by", "where", "finally",
    "typealias", "package", "import", "null", "true", "false", "break",
    "continue", "get", "set",
})
# A newline does not end an expression when the previous token is one of these…
_CONT_AFTER = frozenset({
    "=", ".", "?.", ",", "(", "[", "{", "->", "&&", "||", "?:", "+", "-", "*",
    "/", "%", "+=", "-=", "*=", "/=", "%=", "==", "!=", "===", "!==", "<", ">",
    "<=", ">=", "..", "..<", "::", "!", "as", "is", "in", "!!.", ":", "@",
})
# …or when the next line starts with one of these.
_CONT_BEFORE = frozenset({
    ".", "?.", "?:", "&&", "||", "->", "as", "else", "catch", "finally", ":",
    "==", "!=", "===", "!==", "*", "/", "%", "..", "..<", "::", "=",
})
_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset({")", "]", "}"})
_TYPE_TOKENS = frozenset({",", ".", "?", "<", ">", "*", "out", "in", "(", ")", "->"})
_CAP_RE = re.compile(r"^[A-Z]")


def _match_table(toks: List[Token]) -> Dict[int, int]:
    """Index of the matching closer for every balanced ( [ {."""
    match = {}  # type: Dict[int, int]
    stack = []  # type: List[int]
    for i, t in enumerate(toks):
        if t[0] != OP:
            continue
        if t[1] in _OPENERS:
            stack.append(i)
        elif t[1] in _CLOSERS:
            # Pop to the nearest opener of the same family (tolerates junk).
            for k in range(len(stack) - 1, -1, -1):
                if _OPENERS[toks[stack[k]][1]] == t[1]:
                    match[stack[k]] = i
                    del stack[k:]
                    break
    return match


class _Parser:
    def __init__(self, source: str):
        self.t = tokenize(source)
        self.n = len(self.t)
        self.p = 0
        self.match = _match_table(self.t)
        self.package = ""
        self.imports = []  # type: List[list]
        self.decls = []  # type: List[dict]
        self.calls = []  # type: List[list]
        self.trefs = []  # type: List[list]
        self.rrefs = []  # type: List[list]
        self.brefs = []  # type: List[list]
        self.rmatch = {v: k for k, v in self.match.items()}
        self._inits = {}  # type: Dict[int, int]
        self._seen_brefs = set()  # type: Set[Tuple[int, str]]

    # ---- token helpers ----

    def text(self, k: int = 0) -> str:
        i = self.p + k
        return self.t[i][1] if 0 <= i < self.n else ""

    def kind(self, k: int = 0) -> str:
        i = self.p + k
        return self.t[i][0] if 0 <= i < self.n else ""

    def nl(self, k: int = 0) -> bool:
        """Is there a real newline before token p+k?"""
        i = self.p + k
        return 0 <= i < self.n and self.t[i][3]

    def line(self, k: int = 0) -> int:
        i = min(max(self.p + k, 0), self.n - 1)
        return self.t[i][2] if self.n else 1

    def is_op(self, s: str, k: int = 0) -> bool:
        i = self.p + k
        return 0 <= i < self.n and self.t[i][0] == OP and self.t[i][1] == s

    def is_id(self, k: int = 0, s: Optional[str] = None) -> bool:
        i = self.p + k
        return 0 <= i < self.n and self.t[i][0] == ID and (s is None or self.t[i][1] == s)

    def eat(self, s: str) -> bool:
        if self.p < self.n and self.t[self.p][1] == s:
            self.p += 1
            return True
        return False

    def skip_balanced(self) -> None:
        """At an opener: jump past its matching closer (or one token)."""
        m = self.match.get(self.p)
        self.p = (m + 1) if m is not None else self.p + 1

    def skip_angle(self) -> List[str]:
        """At '<': skip a type-parameter/argument list; return its identifiers."""
        depth = 0
        names = []
        while self.p < self.n:
            s = self.t[self.p][1]
            if s == "<":
                depth += 1
            elif s == ">":
                depth -= 1
                if depth == 0:
                    self.p += 1
                    break
            elif s in ("{", "}", ";", "=") or (s == ")" and depth <= 0):
                break  # not a type list after all
            elif self.t[self.p][0] == ID:
                names.append(s)
            self.p += 1
        return names

    # ---- declarations ----

    def add_decl(self, kind: str, name: str, parent: int, line: int,
                 mods: List[str], annos: List[str], **extra) -> int:
        d = {"kind": kind, "name": name, "parent": parent, "line": line, "end": line,
             "mods": [m for m in mods if m in _KEEP_MODS], "annos": annos}
        d.update(extra)
        self.decls.append(d)
        return len(self.decls) - 1

    def init_of(self, cls: int) -> int:
        """The synthetic <init> decl of a class (owner of init blocks,
        property-less constructor defaults, super-constructor args)."""
        if cls < 0:
            return cls
        if cls not in self._inits:
            d = self.decls[cls]
            self._inits[cls] = self.add_decl("init", "<init>", cls, d["line"], [], [])
        return self._inits[cls]

    def parse_annotations(self) -> List[str]:
        names = []
        while self.is_op("@"):
            self.p += 1
            if self.is_op("["):  # @[A B]
                self.skip_balanced()
                continue
            # use-site target: @field:Inject, @get:Rule
            if self.is_id() and self.is_op(":", 1) and self.is_id(2):
                self.p += 2
            last = ""
            while self.is_id():
                last = self.text()
                self.p += 1
                if self.is_op(".") and self.is_id(1):
                    self.p += 1
                    continue
                break
            if last:
                names.append(last)
            if self.is_op("<"):
                self.skip_angle()
            if self.is_op("(") and not self.nl():
                self.skip_balanced()
        return names

    def parse_modifiers(self) -> List[str]:
        mods = []
        while self.is_id() and self.text() in MODIFIERS and (self.is_id(1) or self.is_op("@", 1)):
            # `fun interface` is handled by the caller; `enum class` etc. stay here
            mods.append(self.text())
            self.p += 1
            if self.is_op("@"):
                self.parse_annotations()
        return mods

    def parse_type(self) -> str:
        """Parse a type and return its normalized text ('' if none)."""
        start = self.p
        while True:
            if self.is_op("@"):
                self.parse_annotations()
            elif self.is_id(0, "suspend"):
                self.p += 1
            else:
                break
        if self.is_op("("):
            self.skip_balanced()
            if self.eat("->"):
                self.parse_type()
            else:
                while self.eat("?"):
                    pass
        elif self.is_id():
            while self.is_id():
                self.p += 1
                if self.is_op("<"):
                    self.skip_angle()
                if self.is_op(".") and (self.is_id(1) or self.is_op("(", 1)):
                    self.p += 1
                    if self.is_op("("):  # receiver function type A.() -> B
                        self.skip_balanced()
                        if self.eat("->"):
                            self.parse_type()
                        break
                    continue
                break
            while self.eat("?"):
                pass
            if self.is_op("?.") and self.is_op("(", 1):  # A?.() -> B
                self.p += 1
                self.skip_balanced()
                if self.eat("->"):
                    self.parse_type()
        return _join(self.t[start:self.p])

    def parse_params(self, default_owner: int, ctor: bool = False) -> List[list]:
        """At '(': parse a parameter list. Default values are scanned for calls
        owned by default_owner."""
        params = []  # type: List[list]
        if not self.eat("("):
            return params
        while self.p < self.n and not self.is_op(")"):
            before = self.p
            annos = self.parse_annotations()
            mods = self.parse_modifiers()
            annos += self.parse_annotations()
            prop = False
            if self.is_id() and self.text() in ("val", "var"):
                prop = True
                self.p += 1
            name = ""
            if self.is_id():
                name = self.text()
                self.p += 1
            ptype = ""
            if self.eat(":"):
                ptype = self.parse_type()
            has_default = False
            if self.eat("="):
                has_default = True
                self.scan(default_owner, until=")", stops=(",",))
            if name:
                params.append([name, ptype, prop and ctor, has_default, annos, "vararg" in mods])
            if not self.eat(",") and self.p == before:
                self.p += 1  # junk; keep moving
        self.eat(")")
        return params

    def parse_receiver_and_name(self) -> Tuple[str, str]:
        """`String.isEmail` / `List<T>.second` / `Foo?.bar` / `name` ->
        (receiver_text, name)."""
        start = self.p
        last_name_at = -1
        while self.p < self.n:
            if self.is_id():
                last_name_at = self.p
                self.p += 1
                if self.is_op("<"):
                    self.skip_angle()
            else:
                break
            while self.eat("?"):
                pass
            if self.is_op(".") or self.is_op("?."):
                self.p += 1
                continue
            break
        if last_name_at < 0:
            return "", ""
        recv_toks = self.t[start:last_name_at]
        while recv_toks and recv_toks[-1][1] in (".", "?."):
            if recv_toks[-1][1] == "?.":
                recv_toks = recv_toks[:-1] + [(OP, "?", recv_toks[-1][2], False)]
                break
            recv_toks = recv_toks[:-1]
        return _join(recv_toks), self.t[last_name_at][1]

    def parse_class(self, parent: int, mods: List[str], annos: List[str]) -> int:
        kw = self.text()
        line = self.line()
        self.p += 1
        if kw == "fun":  # fun interface
            self.p += 1
            kw = "interface"
        if kw == "class":
            kind = "enum" if "enum" in mods else "annotation" if "annotation" in mods else "class"
        else:
            kind = kw
        name = ""
        if self.is_id() and not (kw == "object" and self.text() in ("by",)):
            name = self.text()
            self.p += 1
        elif "companion" in mods:
            name = "Companion"
        if not name:
            name = "<anonymous>"
        idx = self.add_decl(kind, name, parent, line, mods, annos, tparams=[], params=[],
                            supers=[], inject_ctor=False)
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = [x for x in self.skip_angle() if x not in ("out", "in", "reified")]
        # Primary constructor: [annotations] [modifiers] [constructor] ( params )
        save = self.p
        ctor_annos = self.parse_annotations()
        self.parse_modifiers()
        ctor_annos += self.parse_annotations()
        if self.is_id(0, "constructor") and self.is_op("(", 1):
            self.p += 1
        if self.is_op("("):
            d["params"] = self.parse_params(self.init_of(idx), ctor=True)
            d["inject_ctor"] = "Inject" in ctor_annos or "AssistedInject" in ctor_annos
        else:
            self.p = save  # those annotations belong to the next declaration
        if self.eat(":"):
            self.parse_supertypes(idx)
        if self.is_id(0, "where"):
            while self.p < self.n and not self.is_op("{") and not self._decl_starts_line():
                self.p += 1
        if self.is_op("{"):
            self.p += 1
            if kind == "enum":
                self.parse_enum_entries(idx)
            self.parse_members(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_supertypes(self, cls: int) -> None:
        d = self.decls[cls]
        while self.p < self.n:
            self.parse_annotations()
            line = self.line()
            tname = self.parse_type()
            if not tname:
                break
            has_args = False
            if self.is_op("("):
                has_args = True
                self.p += 1
                self.scan(self.init_of(cls), until=")")
                self.eat(")")
            if self.is_id(0, "by"):
                self.p += 1
                self.scan(self.init_of(cls), stops=(",", "{"), expr=True)
            d["supers"].append([tname, has_args, line])
            if not self.eat(","):
                break

    def parse_enum_entries(self, cls: int) -> None:
        while self.p < self.n:
            self.parse_annotations()
            if self.eat(";"):
                return
            if self.is_op("}"):
                return
            if not self.is_id() or self.text() in DECL_KEYWORDS or self.text() in MODIFIERS:
                return
            if not (self.is_op(",", 1) or self.is_op(";", 1) or self.is_op("(", 1)
                    or self.is_op("{", 1) or self.is_op("}", 1)):
                return
            entry = self.add_decl("entry", self.text(), cls, self.line(), [], [])
            self.p += 1
            if self.is_op("("):
                self.p += 1
                self.scan(self.init_of(cls), until=")")
                self.eat(")")
            if self.is_op("{"):
                self.p += 1
                self.parse_members(entry)
                self.decls[entry]["end"] = self.line(-1)
            self.eat(",")

    def parse_fun(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1  # 'fun'
        tparams = []  # type: List[str]
        if self.is_op("<"):
            tparams = [x for x in self.skip_angle() if x not in ("out", "in", "reified")]
        recv, name = self.parse_receiver_and_name()
        idx = self.add_decl("fun", name or "<anonymous>", parent, line, mods, annos,
                            tparams=tparams, recv=recv, params=[], ret="")
        d = self.decls[idx]
        if self.is_op("("):
            d["params"] = self.parse_params(idx)
        if self.eat(":"):
            d["ret"] = self.parse_type()
        if self.is_id(0, "where"):
            self.p += 1
            while self.p < self.n and not self.is_op("{") and not self.is_op("=") \
                    and not self._decl_starts_line():
                self.p += 1
        self.parse_code_body(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_code_body(self, owner: int) -> None:
        """`{ block }` or `= expression` (or nothing, for abstract members)."""
        if self.is_op("{"):
            self.p += 1
            self.scan(owner, until="}")
            self.eat("}")
        elif self.is_op("="):
            self.p += 1
            self.scan(owner, expr=True)

    def parse_prop(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1  # val/var
        if self.is_op("<"):
            self.skip_angle()
        if self.is_op("("):  # destructuring (only legal locally)
            self.skip_balanced()
            recv, name = "", "<destructured>"
        else:
            recv, name = self.parse_receiver_and_name()
        idx = self.add_decl("prop", name or "<anonymous>", parent, line, mods, annos,
                            recv=recv, type="", delegate="", has_init=False)
        d = self.decls[idx]
        if self.eat(":"):
            d["type"] = self.parse_type()
        if self.eat("="):
            d["has_init"] = True
            if not d["type"] and self.is_id() and _CAP_RE.match(self.text()) \
                    and (self.is_op("(", 1) or self.is_op("<", 1) or self.is_op("{", 1)):
                d["itype"] = self.text()  # val x = Foo(...): type inferred from the constructor
            self.scan(idx, expr=True)
        elif self.is_id(0, "by"):
            self.p += 1
            if self.is_id():
                d["delegate"] = self.text()
                if self.is_op("<", 1) and self.is_id(2):  # by inject<Repo>()
                    d["dtype"] = self.text(2)
            self.scan(idx, expr=True)
        self.parse_accessors(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_accessors(self, prop: int) -> None:
        for _ in range(2):
            save = self.p
            self.parse_annotations()
            while self.is_id() and self.text() in ("private", "protected", "internal",
                                                   "public", "override", "inline"):
                self.p += 1
            if self.is_id() and self.text() in ("get", "set") and (
                    self.is_op("(", 1) or self.nl(1) or self.is_op("}", 1)
                    or self.is_op(";", 1) or self.p + 1 >= self.n):
                self.p += 1
                if self.is_op("("):
                    self.skip_balanced()
                if self.eat(":"):
                    self.parse_type()
                self.parse_code_body(prop)
            else:
                self.p = save
                return

    def parse_typealias(self, parent: int, mods: List[str], annos: List[str]) -> None:
        line = self.line()
        self.p += 1
        name = self.text() if self.is_id() else "<anonymous>"
        self.p += 1
        if self.is_op("<"):
            self.skip_angle()
        target = ""
        if self.eat("="):
            target = self.parse_type()
        self.add_decl("typealias", name, parent, line, mods, annos, type=target)

    def _decl_starts_line(self) -> bool:
        return self.p < self.n and self.p > 0 and self.nl() and (
            self.text() in DECL_KEYWORDS or self.is_op("@") or
            (self.text() in MODIFIERS and self.is_id(1)))

    def parse_members(self, parent: int) -> None:
        """Member scope (file top level or class body); stops after the
        closing '}' of a class body."""
        while self.p < self.n:
            if self.is_op("}"):
                self.p += 1
                if parent >= 0:
                    return
                continue
            if self.eat(";"):
                continue
            before = self.p
            if self.parse_member(parent):
                continue
            if self.p == before:
                if self.is_op("{") or self.is_op("(") or self.is_op("["):
                    self.skip_balanced()
                else:
                    self.p += 1

    def parse_member(self, parent: int) -> bool:
        """Parse one declaration at the cursor. False if none was found."""
        annos = self.parse_annotations()
        mods = self.parse_modifiers()
        s = self.text() if self.is_id() else ""
        if s in ("class", "interface", "object") or (s == "fun" and self.is_id(1, "interface")):
            self.parse_class(parent, mods, annos)
        elif s == "fun":
            self.parse_fun(parent, mods, annos)
        elif s in ("val", "var"):
            self.parse_prop(parent, mods, annos)
        elif s == "init" and self.is_op("{", 1):
            self.p += 1
            owner = self.init_of(parent)
            self.parse_code_body(owner)
        elif s == "constructor" and self.is_op("(", 1):
            self.p += 1
            owner = self.init_of(parent)
            self.parse_params(owner)
            if self.eat(":") and self.is_id() and self.is_op("(", 1):
                self.p += 1  # this(...) / super(...)
                self.p += 1
                self.scan(owner, until=")")
                self.eat(")")
            self.parse_code_body(owner)
        elif s == "typealias":
            self.parse_typealias(parent, mods, annos)
        elif s == "package":
            self.p += 1
            self.package = self.qualified_name()
        elif s == "import":
            self.parse_import()
        else:
            return bool(annos or mods)
        return True

    def qualified_name(self) -> str:
        parts = []
        while self.is_id():
            parts.append(self.text())
            self.p += 1
            if self.is_op(".") and self.is_id(1):
                self.p += 1
                continue
            break
        return ".".join(parts)

    def parse_import(self) -> None:
        line = self.line()
        self.p += 1
        path = self.qualified_name()
        star = False
        if self.is_op(".") and self.is_op("*", 1):
            star = True
            self.p += 2
        alias = None
        if self.is_id(0, "as") and self.is_id(1):
            alias = self.text(1)
            self.p += 2
        if path:
            self.imports.append([path, alias, star, line])

    # ---- expression scanning ----

    def _ends_expression(self) -> bool:
        if self.p == 0 or not self.nl():
            return False
        prev = self.t[self.p - 1]
        cur = self.t[self.p]
        if prev[0] == OP and prev[1] in _CONT_AFTER or prev[0] == ID and prev[1] in _CONT_AFTER:
            return False
        if cur[1] in _CONT_BEFORE:
            return False
        return True

    def scan(self, owner: int, until: Optional[str] = None, stops: Tuple[str, ...] = (),
             expr: bool = False) -> None:
        """Scan code, recording calls and references for `owner`.

        until: closer that ends the region at depth 0 (not consumed).
        stops: extra depth-0 tokens that end the region (not consumed).
        expr:  the region is one expression; a newline that does not continue
               it (Kotlin's rules, approximately) ends it.
        """
        depth = 0
        t = self.t
        while self.p < self.n:
            kind, s, line, _ = t[self.p]
            if depth == 0:
                if s in stops and kind == OP:
                    return
                if kind == OP and s in _CLOSERS:
                    if s == until or expr or stops or until is None:
                        return
                    if until == ")" and s == "}":
                        return
                    self.p += 1  # stray closer inside a block
                    continue
                if expr and (s == ";" or self._ends_expression()):
                    return
            if kind == OP:
                if s in _OPENERS:
                    depth += 1
                elif s in _CLOSERS:
                    depth -= 1
                elif s == "@":
                    self.parse_annotations()
                    continue
                elif s == "::":
                    self._callable_ref(owner)
                self.p += 1
                continue
            if kind != ID:
                self.p += 1
                continue
            # -- identifiers / keywords --
            prev = t[self.p - 1][1] if self.p > 0 else ""
            if s == "fun" and prev != "::" and (self.is_id(1) or self.is_op("<", 1)):
                self.parse_fun(owner, [], [])
                continue
            if s in ("class", "interface") and prev != "::" and self.is_id(1):
                self.parse_class(owner, [], [])
                continue
            if s == "object" and prev != "::":
                if self.is_id(1) and not self.is_id(1, "by"):
                    self.parse_class(owner, [], [])
                    continue
                self._anonymous_object(owner)
                continue
            if s in ("val", "var") and prev != ".":
                self._local_var(owner)
                continue
            if s in ("is", "as"):
                k = 2 if self.is_op("?", 1) else 1
                if self.is_id(k) and _CAP_RE.match(self.text(k)):
                    self.trefs.append([owner, self.text(k), line, s])
            if s == "R" and prev != "." and self.is_op(".", 1) and self.is_id(2) \
                    and self.is_op(".", 3) and self.is_id(4):
                self.rrefs.append([owner, self.text(2), self.text(4), line])
                self.p += 5
                continue
            if s == "R" and prev == "." and self.p >= 2 and t[self.p - 2][1] == "android":
                self.p += 5 if self.is_id(4) else 1
                continue
            if s.endswith("Binding") and len(s) > 7 and _CAP_RE.match(s):
                key = (owner, s)
                if key not in self._seen_brefs:
                    self._seen_brefs.add(key)
                    self.brefs.append([owner, s, line])
            if s not in NOT_CALLS and prev not in (".", "?.", "::") and _CAP_RE.match(s) \
                    and (self.is_op(".", 1) or self.is_op("::", 1)) and prev != "@":
                self.trefs.append([owner, s, line, "qual"])
            if s not in NOT_CALLS and prev != "::":
                nxt = self.text(1) if self.kind(1) == OP else ""
                if nxt == "(":
                    self._record_call(owner, self.p, self.p + 1)
                elif nxt == "{" and prev != ":" and s not in MODIFIERS:
                    self._record_call(owner, self.p, self.p + 1)
                elif nxt == "<":
                    close = self._generic_call_close(self.p + 1)
                    if close is not None:
                        self._record_call(owner, self.p, close + 1)
            self.p += 1

    def _generic_call_close(self, i: int) -> Optional[int]:
        """At '<' (index i): if this is foo<T, U>( or foo<T> { , return the
        index of the closing '>'."""
        depth = 0
        t = self.t
        for j in range(i, min(i + 40, self.n)):
            kind, s, _, _ = t[j]
            if s == "<":
                depth += 1
            elif s == ">":
                depth -= 1
                if depth == 0:
                    nxt = t[j + 1][1] if j + 1 < self.n else ""
                    return j if nxt in ("(", "{") else None
            elif kind == ID or s in _TYPE_TOKENS:
                continue
            else:
                return None
        return None

    def _record_call(self, owner: int, name_at: int, open_at: int) -> None:
        """open_at: index of the '(' or trailing-lambda '{' after the name
        (and after any type arguments)."""
        nargs = 0
        after = open_at
        if open_at < self.n and self.t[open_at][1] == "(":
            close = self.match.get(open_at)
            if close is not None:
                nargs = self._count_args(open_at, close)
                after = close + 1
        if after < self.n and self.t[after][1] == "{" and not self.t[after][3]:
            nargs += 1  # trailing lambda
        self.calls.append([owner, self.t[name_at][1], self._receiver(name_at),
                           self.t[name_at][2], nargs, False])

    def _count_args(self, open_at: int, close_at: int) -> int:
        if close_at == open_at + 1:
            return 0
        n = 1
        j = open_at + 1
        while j < close_at:
            s = self.t[j][1]
            if s in _OPENERS and j in self.match:
                j = self.match[j] + 1
                continue
            if s == ",":
                n += 1
            j += 1
        return n

    def _receiver(self, name_at: int) -> Optional[List[list]]:
        """Walk back from a call name over `a.b().c.` to the receiver chain."""
        j = name_at - 1
        if j < 0 or self.t[j][1] not in (".", "?."):
            return None
        segs = []  # type: List[list]
        j -= 1
        while j >= 0 and len(segs) < 8:
            kind, s, _, _ = self.t[j]
            if s == "!!":
                j -= 1
                continue
            if s == ")":
                opener = self.rmatch.get(j)
                if opener is None or opener == 0 or self.t[opener - 1][0] != ID \
                        or self.t[opener - 1][1] in NOT_CALLS:
                    segs.append(["?", 0])
                    break
                segs.append([self.t[opener - 1][1], 1])
                j = opener - 2
            elif kind == ID:
                segs.append([s, 0])
                j -= 1
            elif kind == STR:
                segs.append(['""', 0])
                break
            elif kind == NUM:
                segs.append(["#num", 0])
                break
            else:
                segs.append(["?", 0])
                break
            if j >= 0 and self.t[j][1] in (".", "?."):
                j -= 1
                continue
            break
        segs.reverse()
        return segs or [["?", 0]]

    def _callable_ref(self, owner: int) -> None:
        """At '::'. Foo::bar, this::bar, ::bar (not ::class)."""
        if not self.is_id(1) or self.text(1) == "class":
            return
        recv = None
        if self.p > 0 and self.t[self.p - 1][0] == ID:
            recv = [[self.t[self.p - 1][1], 0]]
        self.calls.append([owner, self.text(1), recv, self.line(1), -1, True])

    def _anonymous_object(self, owner: int) -> None:
        """`object : Runnable { override fun run() {} }` inside code. Its
        members become nested decls of the owner; supertypes are type refs."""
        self.p += 1
        if self.eat(":"):
            while self.p < self.n:
                line = self.line()
                tname = self.parse_type()
                if not tname:
                    break
                self.trefs.append([owner, _head(tname), line, "obj"])
                if self.is_op("("):
                    self.p += 1
                    self.scan(owner, until=")")
                    self.eat(")")
                if not self.eat(","):
                    break
        if self.is_op("{"):
            self.p += 1
            self.parse_members(owner if owner >= 0 else -1)
            if owner < 0:
                self.eat("}")

    def _local_var(self, owner: int) -> None:
        """`val x: Foo = …` / `val x = Foo(…)` inside code: remember the type
        of x for receiver resolution. The initializer is left for scan()."""
        self.p += 1
        if self.is_op("("):  # destructuring
            self.skip_balanced()
            return
        if not self.is_id():
            return
        name = self.text()
        self.p += 1
        locals_ = self.decls[owner].setdefault("locals", {}) if owner >= 0 else {}
        if self.eat(":"):
            tname = self.parse_type()
            if tname:
                locals_[name] = [tname, "decl"]
        elif self.is_op("=") and self.is_id(1) and _CAP_RE.match(self.text(1)) \
                and (self.is_op("(", 2) or self.is_op("{", 2) or self.is_op("<", 2)):
            locals_[name] = [self.text(1), "ctor"]

    # ---- entry point ----

    def run(self) -> dict:
        self.parse_members(-1)
        return {"package": self.package, "imports": self.imports, "decls": self.decls,
                "calls": self.calls, "trefs": self.trefs, "rrefs": self.rrefs,
                "brefs": self.brefs}


_SPACE_AFTER = frozenset({",", "->", "suspend", "out", "in"})
_SPACE_BEFORE = frozenset({"->"})


def _join(toks: List[Token]) -> str:
    out = []
    for i, tok in enumerate(toks):
        s = tok[1]
        if out and (s in _SPACE_BEFORE or toks[i - 1][1] in _SPACE_AFTER):
            out.append(" ")
        out.append(s)
    return "".join(out).strip()


def _head(type_text: str) -> str:
    """'List<User>?' -> 'List'; 'a.b.Foo<T>' -> 'a.b.Foo'; '(A) -> B' -> ''."""
    m = re.match(r"\s*(?:suspend\s+)?([A-Za-z_][\w.]*)", type_text)
    return m.group(1) if m else ""


def parse_kotlin(source: str) -> dict:
    """Parse one Kotlin file into facts (see module docstring)."""
    return _Parser(source).run()
