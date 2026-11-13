"""Swift declaration + call-site extractor for the code graph.

Reuses the Kotlin parser's token machinery and produces the same facts
shape (see kotlin_parser), with these Swift specifics:

  package   always "" (Swift has no packages; the resolver uses the module)
  imports   [module, None, True, line] for `import UIKit` / `@testable import App`,
            [Module.Type, None, False, line] for `import struct Module.Type`
  decl kinds  class struct enum protocol actor extension typealias fun prop
              init entry
              extension: name is the extended type text; children are its
              members; supers are the conformances it adds
  params    [name, type, False, has_default, attrs, variadic, label]
            label is the argument label ("_" for none)
  calls     [owner, name, recv, line, nargs, is_ref, labels]
            labels: one per argument, "_" for unlabeled, "*" for a trailing closure
  rrefs     [owner, res_type, name, line, how]   how: "R" (R.swift) or "literal"
            (UIImage(named: "logo"), Text("title_key"), NSLocalizedString("k", …), …)
  trefs     how is|as|qual|inject   (inject: Swinject/Resolver `resolve(Foo.self)`)
"""
import re
from typing import List, Optional, Tuple

from .kotlin_lexer import ID, OP, STR
from .kotlin_parser import _CAP_RE, _Parser, _join
from .swift_lexer import tokenize_swift

SWIFT_MODIFIERS = frozenset({
    "public", "private", "fileprivate", "internal", "open", "final", "static",
    "override", "mutating", "nonmutating", "lazy", "weak", "unowned", "required",
    "convenience", "dynamic", "optional", "indirect", "prefix", "postfix", "infix",
    "nonisolated", "isolated", "distributed", "package", "consuming", "borrowing",
})
_KEEP = frozenset({
    "private", "fileprivate", "public", "open", "final", "static", "class", "override",
    "mutating", "lazy", "weak", "required", "convenience", "indirect",
})
TYPE_KEYWORDS = {"class": "class", "struct": "struct", "enum": "enum", "protocol": "protocol",
                 "actor": "actor", "extension": "extension"}
# After `class` these mean `class` is a modifier (class func / class var ...).
_CLASS_MEMBER_FOLLOW = frozenset({"func", "var", "let", "subscript", "override", "final",
                                  "private", "public", "open", "internal", "fileprivate"})
SWIFT_NOT_CALLS = frozenset({
    "if", "guard", "while", "for", "switch", "case", "default", "catch", "do", "repeat",
    "return", "throw", "throws", "rethrows", "try", "await", "defer", "else", "in", "is",
    "as", "func", "let", "var", "class", "struct", "enum", "protocol", "extension",
    "actor", "import", "typealias", "where", "willSet", "didSet", "self",
    "Self", "super", "nil", "true", "false", "break", "continue", "fallthrough", "some",
    "any", "inout", "async", "lazy", "weak", "unowned", "static", "private", "public",
    "fileprivate", "internal", "open", "final", "override", "mutating", "subscript",
    "deinit", "associatedtype", "operator", "precedencegroup", "selector", "available",
    "keyPath", "unavailable", "consume", "copy", "each", "repeat",
})
_DECL_WORDS = frozenset({"func", "var", "let", "class", "struct", "enum", "protocol", "actor",
                         "extension", "init", "deinit", "subscript", "case", "typealias",
                         "associatedtype", "import", "}"})
_CONDITION_KEYWORDS = frozenset({"if", "guard", "while", "for", "switch", "catch"})
_NO_TRAILING = frozenset({"else", "do", "defer", "repeat", "get", "set", "willSet", "didSet",
                          "in", "throws", "rethrows", "async", "try", "await", "return"})
_CONT_AFTER = frozenset({
    "=", ".", "?.", ",", "(", "[", "{", "->", "&&", "||", "??", "?", ":", "+", "-", "*",
    "/", "%", "+=", "-=", "*=", "/=", "%=", "==", "!=", "===", "!==", "<", ">", "<=", ">=",
    "...", "..<", "&", "|", "^", "!", "as", "is", "in", "try", "await", "@", "\\",
})
_CONT_BEFORE = frozenset({
    ".", "?.", "??", "&&", "||", "->", "as", "is", "else", "catch", ":", "?", "==", "!=",
    "===", "!==", "*", "/", "%", "...", "..<", "=", "&", "|", "^",
})
_TYPE_PREFIX = frozenset({"some", "any", "inout", "borrowing", "consuming", "sending",
                          "__owned", "__shared", "isolated"})
_EFFECTS = frozenset({"async", "throws", "rethrows", "reasync"})
# call name -> resource type for string-literal resource lookups, and labels
# that disqualify (systemName images are SF Symbols, not assets).
RES_CALLS = {
    "UIImage": "image", "Image": "image", "NSImage": "image",
    "UIColor": "color", "Color": "color", "NSColor": "color",
    "NSLocalizedString": "string", "Text": "string", "LocalizedStringKey": "string",
    "LocalizedStringResource": "string", "String": "string",
    "UIStoryboard": "storyboard", "UINib": "nib", "loadNibNamed": "nib",
    "instantiateViewController": "vcid",
}
_RES_BAD_LABELS = frozenset({"systemName", "verbatim", "format", "cgColor", "red", "white",
                             "hue", "uiImage", "cgImage", "nsImage", "decoding", "contentsOf"})


class _SwiftParser(_Parser):
    def __init__(self, source: str):
        super().__init__(source, tokens=tokenize_swift(source))

    # ---- declarations ----

    def parse_modifiers(self) -> List[str]:
        mods = []
        while self.is_id():
            s = self.text()
            if s == "class" and self.is_id(1) and self.text(1) in _CLASS_MEMBER_FOLLOW:
                mods.append("class")
                self.p += 1
                continue
            if s not in SWIFT_MODIFIERS or not (self.is_id(1) or self.is_op("@", 1) or self.is_op("(", 1)):
                break
            if self.is_op("(", 1):  # private(set), unowned(unsafe)
                if not (self.is_id(2) and self.is_op(")", 3)):
                    break
                mods.append(s)
                self.p += 4
                continue
            mods.append(s)
            self.p += 1
            if self.is_op("@"):
                self.parse_annotations()
        return mods

    def add_decl(self, kind, name, parent, line, mods, annos, **extra):
        idx = super().add_decl(kind, name, parent, line, [], annos, **extra)
        self.decls[idx]["mods"] = [m for m in mods if m in _KEEP]
        return idx

    def parse_type(self) -> str:
        start = self.p
        self._type_one()
        while self.is_op("&"):  # protocol composition
            self.p += 1
            self._type_one()
        return _join(self.t[start:self.p])

    def _generic_params(self) -> List[str]:
        """At '<': `<T: Equatable, U>` -> ['T', 'U'] (constraints dropped)."""
        start = self.p
        self.skip_angle()
        names = []
        depth = 0
        for j in range(start, self.p):
            s = self.t[j][1]
            if s == "<":
                depth += 1
            elif s == ">":
                depth -= 1
            elif depth == 1 and self.t[j][0] == ID and self.t[j - 1][1] in ("<", ","):
                names.append(s)
        return names

    def _type_one(self) -> None:
        while True:
            if self.is_op("@"):  # type attributes take no arguments except @convention(...)
                self.p += 1
                if self.is_id():
                    word = self.text()
                    self.p += 1
                    if word == "convention" and self.is_op("("):
                        self.skip_balanced()
            elif self.is_id() and self.text() in _TYPE_PREFIX:
                self.p += 1
            else:
                break
        if self.is_op("(") or self.is_op("["):
            self.skip_balanced()
            while self.is_id() and self.text() in _EFFECTS:
                self.p += 1
                if self.is_op("("):  # typed throws
                    self.skip_balanced()
            if self.eat("->"):
                self._type_one()
                return
        elif self.is_id():
            while self.is_id():
                self.p += 1
                if self.is_op("<"):
                    self.skip_angle()
                if (self.is_op(".") or self.is_op("?.")) and self.is_id(1):
                    self.p += 1
                    continue
                break
        else:
            return
        while self.is_op("?") or self.is_op("!"):
            self.p += 1

    def parse_swift_params(self, owner: int) -> List[list]:
        params = []  # type: List[list]
        if not self.eat("("):
            return params
        while self.p < self.n and not self.is_op(")"):
            before = self.p
            annos = self.parse_annotations()
            names = []
            while self.is_id() and len(names) < 2:
                names.append(self.text())
                self.p += 1
            ptype, variadic, has_default = "", False, False
            if self.eat(":"):
                ptype = self.parse_type()
                if self.eat("..."):
                    variadic = True
            if self.eat("="):
                has_default = True
                self.scan(owner, until=")", stops=(",",))
            if names:
                label = names[0]
                name = names[-1]
                params.append([name, ptype, False, has_default, annos, variadic, label])
            if not self.eat(",") and self.p == before:
                self.p += 1
        self.eat(")")
        return params

    def _skip_effects_and_where(self) -> None:
        while self.is_id() and self.text() in _EFFECTS:
            self.p += 1
            if self.is_op("("):
                self.skip_balanced()

    def _skip_where(self) -> None:
        if self.is_id(0, "where"):
            while self.p < self.n and not self.is_op("{") and not self._decl_starts_line():
                self.p += 1

    def _body(self, owner: int) -> None:
        if self.is_op("{"):
            self.p += 1
            self.scan(owner, until="}")
            self.eat("}")

    def parse_type_decl(self, parent: int, mods: List[str], annos: List[str]) -> int:
        kw = self.text()
        line = self.line()
        self.p += 1
        kind = TYPE_KEYWORDS[kw]
        if kind == "extension":
            name = self.parse_type()
        else:
            name = self.text() if self.is_id() else "<anonymous>"
            if self.is_id():
                self.p += 1
        idx = self.add_decl(kind, name or "<anonymous>", parent, line, mods, annos,
                            tparams=[], params=[], supers=[], inject_ctor=False)
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = self._generic_params()
        if self.eat(":"):
            while self.p < self.n:
                tline = self.line()
                start = self.p
                self._type_one()
                if self.p == start:
                    break
                d["supers"].append([_join(self.t[start:self.p]), False, tline])
                while self.eat("&"):
                    start = self.p
                    self._type_one()
                    d["supers"].append([_join(self.t[start:self.p]), False, tline])
                if not self.eat(","):
                    break
        self._skip_where()
        if self.is_op("{"):
            self.p += 1
            self.parse_members(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_fun(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        kw = self.text()
        self.p += 1
        if kw == "func":
            if self.is_id():
                name = self.text()
                self.p += 1
            else:  # operator function: func == (lhs:, rhs:)
                parts = []
                while self.p < self.n and self.kind() == OP and not self.is_op("(") and not self.is_op("<"):
                    parts.append(self.text())
                    self.p += 1
                name = "".join(parts) or "<operator>"
        else:
            name = kw  # subscript / deinit
        idx = self.add_decl("fun", name, parent, line, mods, annos, tparams=[], recv="",
                            params=[], ret="")
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = self._generic_params()
        if self.is_op("("):
            d["params"] = self.parse_swift_params(idx)
        self._skip_effects_and_where()
        if self.eat("->"):
            d["ret"] = self.parse_type()
        self._skip_where()
        self._body(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_init(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1
        while self.is_op("?") or self.is_op("!"):
            self.p += 1
        idx = self.add_decl("init", "init", parent, line, mods, annos, params=[])
        d = self.decls[idx]
        if self.is_op("<"):
            self.skip_angle()
        if self.is_op("("):
            d["params"] = self.parse_swift_params(idx)
        self._skip_effects_and_where()
        self._skip_where()
        self._body(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_prop(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1  # var / let
        idx = -1
        while self.p < self.n:
            if self.is_op("("):  # tuple pattern
                self.skip_balanced()
                name = "<tuple>"
            elif self.is_id():
                name = self.text()
                self.p += 1
            else:
                break
            idx = self.add_decl("prop", name, parent, line, mods, annos, recv="", type="",
                                delegate="", has_init=False)
            d = self.decls[idx]
            if self.eat(":"):
                d["type"] = self.parse_type()
            if self.eat("="):
                d["has_init"] = True
                k = 0
                while self.is_id(k) and self.text(k) in ("try", "await"):
                    k += 1
                if not d["type"] and self.is_id(k) and _CAP_RE.match(self.text(k)) and \
                        (self.is_op("(", k + 1) or self.is_op("<", k + 1)):
                    d["itype"] = self.text(k)
                self.scan(idx, expr=True, stops=(",",))
            if self.is_op("{") and (not d["has_init"] or not self.nl()):
                self._body(idx)  # computed property / accessors / observers
            d["end"] = self.line(-1)
            if not self.eat(","):
                break
        return idx

    def parse_cases(self, parent: int, annos: List[str]) -> None:
        self.p += 1  # case
        while self.p < self.n and self.is_id():
            self.add_decl("entry", self.text(), parent, self.line(), [], annos)
            self.p += 1
            if self.is_op("("):
                self.skip_balanced()
            if self.eat("="):
                self.scan(parent, expr=True, stops=(",",))
            if not self.eat(","):
                break

    def parse_import(self) -> None:
        line = self.line()
        self.p += 1
        kind = None
        if self.is_id() and self.text() in ("struct", "class", "enum", "protocol", "func",
                                            "var", "let", "typealias", "actor"):
            kind = self.text()
            self.p += 1
        path = self.qualified_name()
        if path:
            self.imports.append([path, None, kind is None, line])

    def parse_member(self, parent: int) -> bool:
        annos = self.parse_annotations()
        mods = self.parse_modifiers()
        s = self.text() if self.is_id() else ""
        if s in TYPE_KEYWORDS and (self.is_id(1) or s == "extension"):
            self.parse_type_decl(parent, mods, annos)
        elif s in ("func", "subscript", "deinit"):
            self.parse_fun(parent, mods, annos)
        elif s == "init" and (self.is_op("(", 1) or self.is_op("?", 1) or self.is_op("!", 1)
                              or self.is_op("<", 1)):
            self.parse_init(parent, mods, annos)
        elif s in ("var", "let"):
            self.parse_prop(parent, mods, annos)
        elif s == "case" and parent >= 0:
            self.parse_cases(parent, annos)
        elif s in ("typealias", "associatedtype"):
            line = self.line()
            self.p += 1
            name = self.text() if self.is_id() else "<anonymous>"
            self.p += 1
            if self.is_op("<"):
                self.skip_angle()
            target = ""
            if self.eat("=") or self.eat(":"):
                target = self.parse_type()
            self.add_decl("typealias", name, parent, line, mods, annos, type=target)
        elif s == "import":
            self.parse_import()
        elif s in ("operator", "precedencegroup"):
            self.p += 1
            while self.p < self.n and not self.nl() and not self.is_op("{"):
                self.p += 1
            if self.is_op("{"):
                self.skip_balanced()
        elif self.is_op("#"):  # #Preview { … } and other freestanding macros
            self.p += 1
            if self.is_id():
                self.p += 1
            if self.is_op("("):
                self.skip_balanced()
            if self.is_op("{"):
                self.skip_balanced()
        else:
            return bool(annos or mods)
        return True

    # ---- code ----

    def _decl_starts_line(self) -> bool:
        if not (self.p < self.n and self.p > 0 and self.nl()):
            return False
        s = self.text()
        return s in _DECL_WORDS or self.is_op("@") or (s in SWIFT_MODIFIERS and self.is_id(1))

    def _ends_expression(self) -> bool:
        if self.p == 0 or not self.nl():
            return False
        prev = self.t[self.p - 1][1]
        cur = self.t[self.p][1]
        return prev not in _CONT_AFTER and cur not in _CONT_BEFORE

    def scan(self, owner: int, until: Optional[str] = None, stops: Tuple[str, ...] = (),
             expr: bool = False) -> None:
        depth = 0
        cond = []  # type: List[int]   depths at which a statement condition is open
        t = self.t
        while self.p < self.n:
            kind, s, line, _nl = t[self.p]
            if depth == 0:
                if s in stops and kind == OP:
                    return
                if kind == OP and s in (")", "]", "}"):
                    if s == until or expr or stops or until is None:
                        return
                    if until == ")" and s == "}":
                        return
                    self.p += 1
                    continue
                if expr and (s == ";" or self._ends_expression()):
                    return
            if kind == OP:
                if s in ("(", "[", "{"):
                    if s == "{" and cond and cond[-1] == depth:
                        cond.pop()  # the statement's body
                    depth += 1
                elif s in (")", "]", "}"):
                    depth -= 1
                elif s == "@":
                    self.parse_annotations()
                    continue
                elif s == "#":
                    self.p += 1
                    if self.is_id():
                        self.p += 1
                    if self.is_op("("):
                        self.skip_balanced()
                    continue
                self.p += 1
                continue
            if kind != ID:
                self.p += 1
                continue
            prev = t[self.p - 1][1] if self.p > 0 else ""
            if prev in (".", "?."):
                pass
            elif s == "func" and (self.is_id(1) or self.kind(1) == OP):
                self.parse_fun(owner, [], [])
                continue
            elif s in TYPE_KEYWORDS and s != "extension" and self.is_id(1):
                self.parse_type_decl(owner, [], [])
                continue
            elif s in ("let", "var"):
                self._local_var(owner)
                continue
            elif s in _CONDITION_KEYWORDS:
                cond.append(depth)
            elif s in ("is", "as"):
                k = 2 if (self.is_op("?", 1) or self.is_op("!", 1)) else 1
                if self.is_id(k) and _CAP_RE.match(self.text(k)):
                    self.trefs.append([owner, self.text(k), line, s])
            if s == "R" and prev not in (".", "?.") and self.is_op(".", 1) and self.is_id(2) \
                    and self.is_op(".", 3) and self.is_id(4):
                rtype, name = self.text(2), self.text(4)
                if rtype == "string" and self.is_op(".", 5) and self.is_id(6):
                    name = self.text(6)  # R.string.localizable.key
                self.rrefs.append([owner, rtype, name, line, "R"])
                self.p += 5
                continue
            if s == "resolve" and self.is_op("(", 1) and self.is_id(2) and self.is_op(".", 3) \
                    and self.is_id(4, "self") and _CAP_RE.match(self.text(2)):
                self.trefs.append([owner, self.text(2), line, "inject"])
            if prev not in (".", "?.", "\\") and _CAP_RE.match(s) and s != "Self" \
                    and self.is_op(".", 1) and prev != "@":
                self.trefs.append([owner, s, line, "qual"])
            if s in ("get", "set") and prev not in (".", "?.") and self._is_accessor():
                self.p += 1  # accessor block `get {` / `set(v) {`, not a call
                continue
            if s not in SWIFT_NOT_CALLS or (s == "init" and prev in (".", "?.")):
                if prev == "#" or prev == "func":
                    self.p += 1
                    continue
                nxt = self.text(1) if self.kind(1) == OP else ""
                if nxt == "(":
                    self._record_call(owner, self.p, self.p + 1)
                    if s in RES_CALLS:
                        self._resource_literal(owner, s, self.p + 1, line)
                elif nxt == "{" and not (cond and cond[-1] == depth) and s not in _NO_TRAILING \
                        and prev not in (":", "->") and not self.nl(1):
                    self._record_call(owner, self.p, self.p + 1)
                elif nxt == "<":
                    close = self._generic_call_close(self.p + 1)
                    if close is not None:
                        self._record_call(owner, self.p, close + 1)
            self.p += 1

    def _is_accessor(self) -> bool:
        if self.is_op("{", 1):
            return True
        if self.is_op("(", 1):
            close = self.match.get(self.p + 1)
            return close is not None and close + 1 < self.n and self.t[close + 1][1] == "{" \
                and not self.t[close + 1][3]
        return False

    def _record_call(self, owner: int, name_at: int, open_at: int) -> None:
        labels = []  # type: List[str]
        after = open_at
        t = self.t
        if open_at < self.n and t[open_at][1] == "(":
            close = self.match.get(open_at)
            if close is not None:
                labels = self._arg_labels(open_at, close)
                after = close + 1
        # trailing closures: `{…}` then optionally `label: {…}` …
        while after < self.n and t[after][1] == "{" and not t[after][3]:
            labels.append("*")
            close = self.match.get(after)
            if close is None:
                break
            after = close + 1
            if after + 2 < self.n and t[after][0] == ID and t[after + 1][1] == ":" \
                    and t[after + 2][1] == "{":
                after += 2
                continue
            break
        self.calls.append([owner, t[name_at][1], self._receiver(name_at), t[name_at][2],
                           len(labels), False, labels])

    def _arg_labels(self, open_at: int, close_at: int) -> List[str]:
        if close_at == open_at + 1:
            return []
        labels = []
        j = open_at + 1
        start = True
        t = self.t
        while j < close_at:
            if start:
                if t[j][0] == ID and j + 1 < close_at and t[j + 1][1] == ":":
                    labels.append(t[j][1])
                else:
                    labels.append("_")
                start = False
            s = t[j][1]
            if s in ("(", "[", "{") and j in self.match:
                j = self.match[j] + 1
                continue
            if s == ",":
                start = True
            j += 1
        return labels

    def _resource_literal(self, owner: int, name: str, open_at: int, line: int) -> None:
        """UIImage(named: "logo"), Text("key"), NSLocalizedString("key", comment: ""), …"""
        t = self.t
        j = open_at + 1
        label = None
        if j + 1 < self.n and t[j][0] == ID and t[j + 1][1] == ":":
            label = t[j][1]
            j += 2
        if j >= self.n or t[j][0] != STR or not t[j][1]:
            return
        if j + 1 < self.n and t[j + 1][1] not in (",", ")"):
            return
        if label in _RES_BAD_LABELS or (name == "String" and label != "localized"):
            return
        if name == "instantiateViewController" and label != "withIdentifier":
            return
        self.rrefs.append([owner, RES_CALLS[name], t[j][1], line, "literal"])

    def _receiver(self, name_at: int):
        segs = super()._receiver(name_at)
        if segs:
            for seg in segs:
                if seg[0] == "self":
                    seg[0] = "this"
        return segs

    def _local_var(self, owner: int) -> None:
        """`let x: Foo`, `let x = Foo(…)`, `if let x = y as? Foo` -> locals."""
        self.p += 1
        if self.is_op("("):
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
            return
        if not self.is_op("="):
            return
        k = 1
        while self.is_id(k) and self.text(k) in ("try", "await"):
            k += 1
        if self.is_op("?", k) or self.is_op("!", k):
            k += 1
        # Foo(…) / Outer.Inner(…)
        parts = []
        while self.is_id(k) and _CAP_RE.match(self.text(k)):
            parts.append(self.text(k))
            if self.is_op(".", k + 1) and self.is_id(k + 2):
                k += 2
                continue
            k += 1
            break
        if parts and (self.is_op("(", k) or self.is_op("<", k)):
            locals_[name] = [".".join(parts), "ctor"]
            return
        # `= expr as? Foo` before the end of the clause
        for j in range(self.p + 1, min(self.p + 20, self.n)):
            s = self.t[j][1]
            if s in ("{", ",", "else") or self.t[j][3]:
                break
            if s == "as" and j + 2 < self.n:
                k2 = j + 2 if self.t[j + 1][1] in ("?", "!") else j + 1
                if self.t[k2][0] == ID and _CAP_RE.match(self.t[k2][1]):
                    locals_[name] = [self.t[k2][1], "decl"]
                break


def parse_swift(source: str) -> dict:
    """Parse one Swift file into facts (see module docstring)."""
    return _SwiftParser(source).run()
