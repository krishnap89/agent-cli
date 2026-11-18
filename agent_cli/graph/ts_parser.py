"""TypeScript / JavaScript declaration + call-site extractor for the code graph.

Reuses the Kotlin parser's token machinery (_Parser) over ts_lexer tokens and
produces the same facts shape (see kotlin_parser), with these specifics:

  package   always "" (the resolver names declarations after the file path)
  module    True when the file has ES/CommonJS imports or exports (else it is
            a script whose top-level declarations are global, e.g. a .d.ts)
  imports   [[spec, imported, local, line]]
              import X from 'm'            -> ['m', 'default', 'X']
              import { a as b } from 'm'   -> ['m', 'a', 'b']
              import * as ns from 'm'      -> ['m', '*', 'ns']
              const x = require('m')       -> ['m', '=', 'x']
              const { a } = require('m')   -> ['m', 'a', 'a']
              import 'm' / import('m')     -> ['m', None, None]
  exports   [[exported, local, spec, line]]
              export class A / export { A }  -> ['A', 'A', None]
              export default foo             -> ['default', 'foo', None]
              export { a as b } from 'm'     -> ['b', 'a', 'm']
              export * from 'm'              -> ['*', None, 'm']
              export * as ns from 'm'        -> ['ns', '*', 'm']
              module.exports = X / exports.a = …  (CommonJS) likewise
  decl kinds  class interface enum namespace object typealias fun prop init entry
              object     a top-level object literal with methods (const api = {…},
                         export default {…}, module.exports = {…}); a member
                         `a` / `a: b` that just names a value gets "alias"
              fun        functions, methods, arrow functions / function
                         expressions assigned to a const or class field,
                         interface method signatures; "<module>" owns
                         top-level statements (calls made at load time)
              init       constructor; parameter properties also become props
  params    [name, type, is_property, optional_or_default, decorators, rest]
  calls     [owner, name, recv, line, nargs, is_ref]   (`new Foo()` is a call of
            Foo; a JSX element <Foo …/> is a call with nargs -1; an attribute
            value {handler} is a reference, is_ref True)
  trefs     how: as|satisfies|instanceof|qual
"""
import re
from typing import List, Optional, Tuple

from .kotlin_lexer import ID, NUM, OP, STR, Token
from .kotlin_parser import _CAP_RE, _Parser
from .ts_lexer import jsx_enabled, tokenize_ts

TS_MODIFIERS = frozenset({
    "public", "private", "protected", "readonly", "static", "abstract", "override",
    "declare", "async", "accessor", "export", "default", "const",
})
_KEEP = frozenset({"private", "protected", "static", "abstract", "override", "readonly",
                   "async", "declare"})
_PARAM_MODS = frozenset({"public", "private", "protected", "readonly", "override"})
TS_NOT_CALLS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "typeof", "instanceof", "in", "of",
    "new", "delete", "void", "throw", "case", "do", "else", "yield", "await", "function",
    "class", "const", "let", "var", "import", "export", "default", "super", "this", "null",
    "undefined", "true", "false", "constructor", "extends", "implements", "interface",
    "type", "enum", "async", "get", "set", "static", "as", "satisfies", "keyof", "readonly",
    "declare", "abstract", "namespace", "module", "try", "finally", "with", "debugger",
    "break", "continue", "require", "is", "infer", "unique", "asserts",
})
# Decorators that make a class's constructor parameters injected dependencies
# (Angular, NestJS, InversifyJS, tsyringe, TypeDI).
INJECT_CLASS_DECORATORS = frozenset({
    "Injectable", "injectable", "Component", "Directive", "Pipe", "Controller", "Resolver",
    "Service", "WebSocketGateway", "Processor", "singleton", "scoped", "autoInjectable",
    "Middleware", "Guard", "Interceptor",
})
_INJECT_PARAM_DECORATORS = re.compile(r"^(Inject|inject|Inject[A-Z]\w*|Optional|Self|Host)$")
# A newline does not end an expression after these tokens…
_CONT_AFTER = frozenset({
    "=", ".", "?.", ",", "(", "[", "{", "=>", "&&", "||", "??", "?", ":", "+", "-", "*", "/",
    "%", "**", "+=", "-=", "*=", "/=", "%=", "**=", "??=", "&&=", "||=", "|=", "&=", "^=",
    "==", "!=", "===", "!==", "<", ">", "<=", ">=", "&", "|", "^", "!", "~", "...", "<attr",
    "new", "await", "typeof", "in", "of", "instanceof", "as", "satisfies", "void",
    "delete", "yield", "return", "throw", "keyof", "extends", "@",
})
# …or before these.
_CONT_BEFORE = frozenset({
    ".", "?.", "??", "&&", "||", "=>", "?", ":", "=", "==", "!=", "===", "!==", "*", "/", "%",
    "**", "+", "-", "&", "|", "^", ">", "<=", ">=", "+=", "-=", "*=", "/=", "as", "satisfies",
    "in", "instanceof", ",", "??=", "&&=", "||=", "extends", "implements",
})
_STMT_DECL_WORDS = frozenset({"import", "export", "class", "function", "async", "interface",
                              "enum", "type", "const", "let", "var", "declare", "abstract",
                              "namespace", "module"})
_HANDLER_ATTR_RE = re.compile(r"^on[A-Z]")
_REF_ATTRS = frozenset({"component", "render", "element", "Component", "renderItem", "action",
                        "loader", "fallbackRender"})
_GENERIC_OK = frozenset({".", ",", "?", "|", "&", "[", "]", "{", "}", ":", ";", "(", ")",
                         "=>", "<", ">", "...", "-"})
_MAX_TYPE = 2000


_WORDISH = (ID, NUM, STR)


def _ts_join(toks: List[Token]) -> str:
    """Type tokens -> readable text: 'Promise<User | null>', '(id: string) => void'."""
    out = []
    for i, tok in enumerate(toks):
        s = '"%s"' % tok[1] if tok[0] == STR else tok[1]
        if out:
            prev = toks[i - 1]
            if s in ("|", "&", "=>", "=") or prev[1] in ("|", "&", "=>", ",", ":", "=", ";") \
                    or (tok[0] in _WORDISH and prev[0] in _WORDISH):
                out.append(" ")
        out.append(s)
    return "".join(out).strip()


def default_name(rel: str) -> str:
    """Name for an anonymous default export: the file stem ('UserCard' for
    UserCard.tsx; the folder name for index files)."""
    parts = rel.replace("\\", "/").split("/")
    stem = parts[-1].split(".", 1)[0]
    if stem == "index" and len(parts) > 1:
        stem = parts[-2]
    stem = re.sub(r"\W+", "_", stem).strip("_") or "default"
    return stem if not stem[0].isdigit() else "_" + stem


class _TsParser(_Parser):
    def __init__(self, source: str, rel: str = "x.ts"):
        super().__init__(source, tokens=tokenize_ts(source, jsx_enabled(rel)))
        self.default_name = default_name(rel)
        self.exports = []  # type: List[list]
        self.is_module = False
        self._mod = -1  # index of the <module> decl
        self._fields = {}  # type: dict   class index -> member names seen

    # ---- helpers ----

    def is_str(self, k: int = 0) -> bool:
        i = self.p + k
        return 0 <= i < self.n and self.t[i][0] == STR

    def module_owner(self) -> int:
        if self._mod < 0:
            self._mod = self.add_decl("fun", "<module>", -1, self.line(), [], [], tparams=[],
                                      recv="", params=[], ret="")
        return self._mod

    def add_decl(self, kind, name, parent, line, mods, annos, **extra):
        idx = super().add_decl(kind, name, parent, line, [], annos, **extra)
        self.decls[idx]["mods"] = [m for m in mods if m in _KEEP]
        return idx

    def skip_angle(self) -> List[str]:
        """At '<': skip type parameters/arguments; return the parameter names
        (identifiers right after '<' or ',' at depth 1)."""
        depth = 0
        names = []
        t = self.t
        while self.p < self.n:
            kind, s = t[self.p][0], t[self.p][1]
            if kind == OP and s in ("(", "[", "{"):
                self.skip_balanced()
                continue
            if s == "<":
                depth += 1
            elif s == ">":
                depth -= 1
                if depth <= 0:
                    self.p += 1
                    break
            elif s in (";", ")", "]", "}") or (kind == OP and s == "=>" and depth <= 0):
                break
            elif kind == ID and depth == 1 and t[self.p - 1][1] in ("<", ","):
                names.append(s)
            self.p += 1
        return names

    def parse_modifiers(self) -> List[str]:
        mods = []
        while self.is_id() and self.text() in TS_MODIFIERS and self.text() not in ("export", "default", "const"):
            nxt = self.p + 1
            if nxt >= self.n:
                break
            k, s = self.t[nxt][0], self.t[nxt][1]
            if not (k == ID or k == STR or s in ("[", "*", "#", "@")):
                break
            mods.append(self.text())
            self.p += 1
            if self.is_op("@"):
                self.parse_annotations()
        return mods

    # ---- types ----

    def parse_type(self) -> str:
        start = self.p
        self._type()
        return _ts_join(self.t[start:self.p])[:_MAX_TYPE]

    def _type(self) -> None:
        self._union()
        if self.is_id(0, "extends") and not self.nl():  # conditional type
            self.p += 1
            self._union()
            if self.eat("?"):
                self._type()
                if self.eat(":"):
                    self._type()

    def _union(self) -> None:
        if self.is_op("|") or self.is_op("&"):
            self.p += 1
        self._primary()
        while self.is_op("|") or self.is_op("&"):
            self.p += 1
            self._primary()

    def _primary(self) -> None:
        while self.is_id() and self.text() in ("keyof", "unique", "readonly", "infer", "asserts",
                                               "abstract") \
                and (self.is_id(1) or self.is_op("(", 1) or self.is_op("[", 1) or self.is_op("{", 1)):
            self.p += 1
        if self.is_id(0, "typeof"):
            self.p += 1
        if self.is_id(0, "new"):
            self.p += 1
        if self.is_op("<"):  # generic function type
            self.skip_angle()
        if self.is_op("("):
            self.skip_balanced()
            if self.eat("=>"):
                self._type()
                return
        elif self.is_op("{") or self.is_op("["):
            self.skip_balanced()
        elif self.kind() in (STR, NUM):
            self.p += 1
        elif self.is_op("-") and self.kind(1) == NUM:
            self.p += 2
        elif self.is_id(0, "import") and self.is_op("(", 1):
            self.p += 1
            self.skip_balanced()
            while self.is_op(".") and self.is_id(1):
                self.p += 2
            if self.is_op("<"):
                self.skip_angle()
        elif self.is_id():
            self.p += 1
            while True:
                if self.is_op("<"):
                    self.skip_angle()
                if self.is_op(".") and self.is_id(1):
                    self.p += 2
                    continue
                break
            if self.is_id(0, "is") and not self.nl():  # type predicate `x is Foo`
                self.p += 1
                self._type()
                return
        else:
            return
        while self.is_op("[") and not self.nl():
            self.skip_balanced()

    # ---- top level / statements ----

    def run(self) -> dict:
        self.parse_statements(-1)
        if self._mod >= 0:
            self.decls[self._mod]["end"] = self.line(-1) if self.n else 1
        return {"package": "", "module": self.is_module, "imports": self.imports,
                "exports": self.exports, "decls": self.decls, "calls": self.calls,
                "trefs": self.trefs, "rrefs": [], "brefs": []}

    def parse_statements(self, parent: int) -> None:
        """Top level (parent -1) or a namespace / `declare global` body (stops
        after its closing '}')."""
        while self.p < self.n:
            if self.is_op("}"):
                self.p += 1
                if parent >= 0 or parent == -2:
                    return
                continue
            if self.eat(";"):
                continue
            before = self.p
            real_parent = -1 if parent == -2 else parent
            if self.parse_decl(real_parent):
                continue
            if self.p == before and real_parent < 0 and self._cjs_export():
                continue
            if self.p == before:
                owner = self.module_owner() if real_parent < 0 else self.init_of(real_parent)
                self.scan(owner, top=True)
            if self.p == before:
                self.p += 1

    def _decl_here(self) -> bool:
        """Does a declaration start at the cursor (for top-level statement ends)?"""
        if self.is_op("@"):
            return True
        if not self.is_id():
            return False
        s = self.text()
        if s not in _STMT_DECL_WORDS:
            return False
        if s == "import":
            return not (self.is_op("(", 1) or self.is_op(".", 1))
        if s == "export":
            return True
        if s == "async":
            return self.is_id(1, "function")
        if s in ("class", "interface", "enum"):
            return self.is_id(1)
        if s == "function":
            return True
        if s == "type":
            return self.is_id(1) and (self.is_op("=", 2) or self.is_op("<", 2))
        if s in ("const", "let", "var"):
            return self.is_id(1) or self.is_op("{", 1) or self.is_op("[", 1)
        if s == "declare":
            return self.is_id(1)
        if s == "abstract":
            return self.is_id(1, "class")
        return (self.is_id(1) or self.is_str(1)) and not self.is_op(".", 1)  # namespace / module

    def _stmt_start(self) -> bool:
        return self.p > 0 and (self.nl() or self.t[self.p - 1][1] in (";", "}"))

    def parse_decl(self, parent: int) -> bool:
        start = self.p
        annos = self.parse_annotations()
        exported = default = False
        top = parent < 0
        if self.is_id(0, "export"):
            self.is_module = True
            if self.is_op("=", 1):  # export = foo
                self.p += 2
                if self._name_value():
                    self.exports.append(["default", self.text(), None, self.line()])
                    self.p += 1
                else:
                    self.scan(self.module_owner(), expr=True)
                return True
            if self.is_op("{", 1) or self.is_op("*", 1) or (
                    self.is_id(1, "type") and (self.is_op("{", 2) or self.is_op("*", 2))):
                self.p += 1
                self.eat("type")
                self._export_clause(top)
                return True
            if self.is_id(1, "import") or self.is_id(1, "as"):  # export import A = B / export as namespace X
                while self.p < self.n and not self.is_op(";") and not (self.p > start + 1 and self.nl()):
                    self.p += 1
                return True
            exported = True
            self.p += 1
            if self.is_id(0, "default"):
                default = True
                self.p += 1
            annos += self.parse_annotations()
        mods = self.parse_modifiers()
        annos += self.parse_annotations()
        if not self.is_id():
            if default and top:
                self._default_expression(annos)
                return True
            if self.p != start:
                return True
            return False
        s = self.text()
        names = []  # type: List[str]
        line = self.line()
        if s == "class":
            idx = self.parse_class(parent, mods, annos, self.default_name if default else None)
            names = [self.decls[idx]["name"]]
        elif s == "async" and self.is_id(1, "function"):
            self.p += 1
            names = [self.decls[self.parse_function(parent, mods + ["async"], annos,
                                                    default)]["name"]]
        elif s == "function":
            names = [self.decls[self.parse_function(parent, mods, annos, default)]["name"]]
        elif s == "interface" and self.is_id(1):
            names = [self.decls[self.parse_interface(parent, mods, annos)]["name"]]
        elif s == "enum" or (s == "const" and self.is_id(1, "enum")):
            if s == "const":
                self.p += 1
            names = [self.decls[self.parse_enum(parent, mods, annos)]["name"]]
        elif s == "type" and self.is_id(1) and (self.is_op("=", 2) or self.is_op("<", 2)):
            names = [self.decls[self.parse_typealias(parent, mods, annos)]["name"]]
        elif s in ("namespace", "module") and (self.is_id(1) or self.is_str(1)) and not self.is_op(".", 1):
            idx = self.parse_namespace(parent, mods, annos)
            if idx >= 0:
                names = [self.decls[idx]["name"]]
        elif s == "global" and "declare" in mods and self.is_op("{", 1):
            self.p += 2
            self.parse_statements(-2)
            return True
        elif s in ("const", "let", "var") and not default and (
                self.is_id(1) or self.is_op("{", 1) or self.is_op("[", 1)):
            names = self.parse_var_decl(parent, mods, annos)
        elif s == "import" and not (self.is_op("(", 1) or self.is_op(".", 1)):
            self.parse_import()
            return True
        elif default and top:
            self._default_expression(annos)
            return True
        else:
            if self.p != start:
                return True
            return False
        if exported and top:
            for nm in names:
                self.exports.append(["default" if default else nm, nm, None, line])
        return True

    def _value_ends(self, k: int) -> bool:
        i = self.p + k
        return i >= self.n or self.t[i][1] in (";", ")", "}", ",") or self.t[i][3]

    def _name_value(self) -> bool:
        """An identifier that is the whole value (`= foo;`), not a literal."""
        return self.is_id() and self.text() not in TS_NOT_CALLS and self._value_ends(1)

    def _default_expression(self, annos: List[str]) -> None:
        """`export default <expr>` (the keywords already consumed)."""
        line = self.line()
        if self._name_value():
            self.exports.append(["default", self.text(), None, line])
            self.p += 1
            return
        if self.is_op("{"):
            idx = self.parse_object_literal(-1, "default", line, [], annos)
        elif self._arrow_at():
            idx = self.add_decl("fun", self.default_name, -1, line, [], annos, tparams=[],
                                recv="", params=[], ret="")
            self.parse_arrow(idx)
        else:
            idx = self.add_decl("prop", "default", -1, line, [], annos, recv="", type="",
                                delegate="", has_init=True)
            self._init_type(self.decls[idx], 0)
            self.scan(idx, expr=True)
            self.decls[idx]["end"] = self.line(-1)
        self.exports.append(["default", self.decls[idx]["name"], None, line])

    def _export_clause(self, top: bool) -> None:
        """At '{' or '*' after `export`."""
        line = self.line()
        pairs = []  # type: List[Tuple[str, Optional[str]]]   (exported, local)
        if self.eat("*"):
            if self.is_id(0, "as") and (self.is_id(1) or self.is_str(1)):
                pairs.append((self.text(1), "*"))
                self.p += 2
            else:
                pairs.append(("*", None))
        elif self.is_op("{"):
            close = self.match.get(self.p)
            self.p += 1
            while self.p < self.n and not self.is_op("}"):
                before = self.p
                if self.is_id(0, "type") and (self.is_id(1) or self.is_str(1)) and not self.is_op(",", 1) \
                        and not self.is_id(1, "as"):
                    self.p += 1
                if self.is_id() or self.is_str():
                    local = self.text()
                    self.p += 1
                    exported = local
                    if self.is_id(0, "as") and (self.is_id(1) or self.is_str(1)):
                        exported = self.text(1)
                        self.p += 2
                    pairs.append((exported, local))
                if not self.eat(",") and self.p == before:
                    self.p += 1
            self.p = (close + 1) if close is not None else self.p + 1
        spec = None
        if self.is_id(0, "from") and self.is_str(1):
            spec = self.text(1)
            self.p += 2
        if top:
            for exported, local in pairs:
                self.exports.append([exported, local, spec, line])

    def _cjs_export(self) -> bool:
        """module.exports = … / exports.x = … / module.exports.x = … ."""
        k = 0
        named = None
        if self.is_id(0, "module") and self.is_op(".", 1) and self.is_id(2, "exports"):
            k = 3
            if self.is_op(".", 3) and self.is_id(4):
                named = self.text(4)
                k = 5
        elif self.is_id(0, "exports") and self.is_op(".", 1) and self.is_id(2):
            named = self.text(2)
            k = 3
        else:
            return False
        if not self.is_op("=", k):
            return False
        self.is_module = True
        line = self.line()
        self.p += k + 1
        if named is None:
            if self._name_value():
                self.exports.append(["default", self.text(), None, line])
                self.p += 1
                return True
            if self.is_op("{"):
                idx = self.parse_object_literal(-1, "default", line, [], [])
                self.exports.append(["default", "default", None, line])
                return True
            if self.is_id(0, "class"):
                idx = self.parse_class(-1, [], [], self.default_name)
                self.exports.append(["default", self.decls[idx]["name"], None, line])
                return True
            named = "default"
        if self._name_value() and named != "default":
            self.exports.append([named, self.text(), None, line])
            self.p += 1
            return True
        if self._function_value_at():
            idx = self.add_decl("fun", named if named != "default" else self.default_name, -1, line,
                                [], [], tparams=[], recv="", params=[], ret="")
            self._function_value(idx)
        else:
            idx = self.add_decl("prop", named, -1, line, [], [], recv="", type="", delegate="",
                                has_init=True)
            self._init_type(self.decls[idx], 0)
            self.scan(idx, expr=True)
            self.decls[idx]["end"] = self.line(-1)
        self.exports.append([named, self.decls[idx]["name"], None, line])
        return True

    # ---- imports ----

    def parse_import(self) -> None:
        line = self.line()
        self.p += 1
        self.is_module = True
        if self.is_id(0, "type") and not (self.is_op(",", 1) or self.is_id(1, "from") or self.is_op("=", 1)):
            self.p += 1
        if self.is_str():
            self.imports.append([self.text(), None, None, line])
            self.p += 1
            return
        bindings = []  # type: List[Tuple[str, str]]
        if self.is_id() and not self.is_id(0, "from"):
            local = self.text()
            self.p += 1
            if self.is_op("="):  # import x = require('m') / import x = A.B
                self.p += 1
                if self.is_id(0, "require") and self.is_op("(", 1) and self.is_str(2):
                    self.imports.append([self.text(2), "=", local, line])
                    self.p += 3
                    self.eat(")")
                else:
                    self.qualified_name()
                return
            bindings.append(("default", local))
            self.eat(",")
        if self.eat("*"):
            if self.is_id(0, "as") and self.is_id(1):
                bindings.append(("*", self.text(1)))
                self.p += 2
        elif self.is_op("{"):
            close = self.match.get(self.p)
            self.p += 1
            while self.p < self.n and not self.is_op("}"):
                before = self.p
                if self.is_id(0, "type") and (self.is_id(1) or self.is_str(1)) and not self.is_op(",", 1) \
                        and not self.is_id(1, "as"):
                    self.p += 1
                if self.is_id() or self.is_str():
                    imported = self.text()
                    self.p += 1
                    local = imported
                    if self.is_id(0, "as") and self.is_id(1):
                        local = self.text(1)
                        self.p += 2
                    bindings.append((imported, local))
                if not self.eat(",") and self.p == before:
                    self.p += 1
            self.p = (close + 1) if close is not None else self.p + 1
        if not (self.is_id(0, "from") and self.is_str(1)):
            return
        spec = self.text(1)
        self.p += 2
        if self.is_id() and self.text() in ("assert", "with") and self.is_op("{", 1):
            self.p += 1
            self.skip_balanced()
        for imported, local in bindings:
            self.imports.append([spec, imported, local, line])

    def _require_pattern(self, open_at: int, close_at: int, spec: str, line: int) -> None:
        """`const { a, b: c } = require('m')` -> named imports."""
        t = self.t
        j = open_at + 1
        while j < close_at:
            if t[j][0] == ID:
                imported = local = t[j][1]
                if j + 2 < close_at and t[j + 1][1] == ":" and t[j + 2][0] == ID:
                    local = t[j + 2][1]
                    j += 2
                self.imports.append([spec, imported, local, line])
            elif t[j][1] in ("{", "[", "(") and j in self.match:
                j = self.match[j]
            j += 1

    # ---- declarations ----

    def parse_class(self, parent: int, mods: List[str], annos: List[str],
                    anon_name: Optional[str] = None) -> int:
        line = self.line()
        self.p += 1  # class
        name = anon_name or "<anonymous>"
        if self.is_id() and self.text() not in ("extends", "implements"):
            name = self.text()
            self.p += 1
        idx = self.add_decl("class", name, parent, line, mods, annos, tparams=[], params=[],
                            supers=[], inject_ctor=bool(set(annos) & INJECT_CLASS_DECORATORS))
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = self.skip_angle()
        if self.is_id(0, "extends"):
            self.p += 1
            tline = self.line()
            text = self.parse_type()
            if self.is_op("("):  # mixin call: extends Mixin(Base)
                self.p += 1
                self.scan(self.init_of(idx), until=")")
                self.eat(")")
            if text:
                d["supers"].append([text, True, tline])
        if self.is_id(0, "implements"):
            self.p += 1
            while self.p < self.n:
                tline = self.line()
                text = self.parse_type()
                if not text:
                    break
                d["supers"].append([text, False, tline])
                if not self.eat(","):
                    break
        if self.is_op("{"):
            self.p += 1
            self.parse_class_members(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_class_members(self, cls: int) -> None:
        while self.p < self.n:
            if self.is_op("}"):
                self.p += 1
                return
            if self.eat(";") or self.eat(","):
                continue
            before = self.p
            self.parse_class_member(cls)
            if self.p == before:
                if self.is_op("{") or self.is_op("(") or self.is_op("["):
                    self.skip_balanced()
                else:
                    self.p += 1

    def _member_name(self) -> Optional[str]:
        if self.is_id():
            s = self.text()
            self.p += 1
            return s
        if self.kind() in (STR, NUM):
            s = self.text() or "<computed>"
            self.p += 1
            return s
        if self.is_op("["):
            self.skip_balanced()
            return "<computed>"
        return None

    def parse_class_member(self, cls: int) -> None:
        line = self.line()
        annos = self.parse_annotations()
        if self.is_id(0, "static") and self.is_op("{", 1):  # static block
            self.p += 2
            self.scan(self.init_of(cls), until="}")
            self.eat("}")
            return
        mods = self.parse_modifiers()
        if self.is_op("[") and self.is_id(1) and self.is_op(":", 2):  # index signature
            self.skip_balanced()
            if self.eat(":"):
                self.parse_type()
            return
        accessor = None
        if self.is_id() and self.text() in ("get", "set") and (
                self.is_id(1) or self.is_str(1) or self.is_op("[", 1)):
            accessor = self.text()
            self.p += 1
        self.eat("*")
        name = self._member_name()
        if name is None:
            return
        if name == "constructor" and self.is_op("("):
            self.parse_constructor(cls, mods, annos, line)
            return
        self.eat("?") or self.eat("!")
        if self.is_op("(") or self.is_op("<"):
            if accessor:  # get x() / set x(v): one property, code owned by it
                idx = self._accessor_prop(cls, name, line, mods, annos)
                d = self.decls[idx]
                had_type = d.get("type")
                self.parse_function_rest(idx)
                if accessor == "get" and d.get("ret") and not had_type:
                    d["type"] = d["ret"]
                return
            idx = self.add_decl("fun", name, cls, line, mods, annos, tparams=[], recv="",
                                params=[], ret="")
            self.parse_function_rest(idx)
            return
        ptype = self.parse_type() if self.eat(":") else ""
        if self.is_op("=") and self._function_value_at(1):
            self.p += 1
            idx = self.add_decl("fun", name, cls, line, mods, annos, tparams=[], recv="",
                                params=[], ret="", ftype=ptype)
            self._function_value(idx)
            return
        idx = self.add_decl("prop", name, cls, line, mods, annos, recv="", type=ptype,
                            delegate="", has_init=False)
        self._fields.setdefault(cls, set()).add(name)
        d = self.decls[idx]
        if self.eat("="):
            d["has_init"] = True
            self._init_type(d, 0)
            self.scan(idx, expr=True, stops=(",",))
        d["end"] = self.line(-1)

    def _accessor_prop(self, cls: int, name: str, line: int, mods, annos) -> int:
        for i in range(len(self.decls) - 1, -1, -1):
            d = self.decls[i]
            if d["parent"] == cls and d["name"] == name and d["kind"] == "prop":
                return i
        self._fields.setdefault(cls, set()).add(name)
        return self.add_decl("prop", name, cls, line, mods, annos, recv="", type="", delegate="",
                             has_init=False)

    def _init_type(self, d: dict, k: int) -> None:
        """Type of an initializer at cursor+k: `new Foo(…)` (itype), Angular
        `inject(Foo)` (dtype)."""
        while self.is_id(k, "await"):
            k += 1
        if self.is_id(k, "new"):
            parts = []
            k += 1
            while self.is_id(k):
                parts.append(self.text(k))
                if self.is_op(".", k + 1) and self.is_id(k + 2):
                    k += 2
                    continue
                k += 1
                break
            if parts and _CAP_RE.match(parts[-1]) and not d.get("type"):
                d["itype"] = ".".join(parts)
        elif self.is_id(k, "inject") and self.is_op("(", k + 1) and self.is_id(k + 2) \
                and _CAP_RE.match(self.text(k + 2)):
            d["delegate"] = "inject"
            d["dtype"] = self.text(k + 2)

    def parse_constructor(self, cls: int, mods: List[str], annos: List[str], line: int) -> None:
        idx = self.add_decl("init", "constructor", cls, line, mods, annos, params=[])
        d = self.decls[idx]
        d["params"] = self.parse_params(idx, ctor_cls=cls)
        cd = self.decls[cls]
        if any(_INJECT_PARAM_DECORATORS.match(a) for p in d["params"] for a in p[4]):
            cd["inject_ctor"] = True
        if cd.get("inject_ctor"):
            cd["params"] = [list(p) for p in d["params"]]
        if self.is_op("{"):
            self.p += 1
            self.scan(idx, until="}")
            self.eat("}")
        d["end"] = self.line(-1)

    def parse_params(self, owner: int, ctor: bool = False, ctor_cls: int = -1) -> List[list]:
        params = []  # type: List[list]
        if not self.eat("("):
            return params
        while self.p < self.n and not self.is_op(")"):
            before = self.p
            line = self.line()
            annos = self.parse_annotations()
            pmods = []
            while self.is_id() and self.text() in _PARAM_MODS and (
                    self.is_id(1) or self.is_op("{", 1) or self.is_op("[", 1)):
                pmods.append(self.text())
                self.p += 1
            rest = self.eat("...")
            name = ""
            names = []  # type: List[str]
            if self.is_id():
                name = self.text()
                self.p += 1
            elif self.is_op("{") or self.is_op("["):
                names = self._pattern_names(self.p)
                self.skip_balanced()
                name = "<destructured>"
            optional = bool(self.eat("?"))
            ptype = self.parse_type() if self.eat(":") else ""
            has_default = False
            if self.eat("="):
                has_default = True
                self.scan(owner, until=")", stops=(",",))
            if name:
                params.append([name, ptype, bool(pmods), has_default or optional, annos, rest])
                if names and owner >= 0:
                    self.decls[owner].setdefault("vars", []).extend(names)
            if pmods and ctor_cls >= 0 and name and name != "<destructured>":
                pidx = self.add_decl("prop", name, ctor_cls, line, pmods, annos, recv="",
                                     type=ptype, delegate="", has_init=False)
                self.decls[pidx]["param"] = True
                self._fields.setdefault(ctor_cls, set()).add(name)
            if not self.eat(",") and self.p == before:
                self.p += 1
        self.eat(")")
        return params

    def _pattern_names(self, open_at: int) -> List[str]:
        """Names bound by a destructuring pattern ({ a, b: c, ...d } / [x, y])."""
        close = self.match.get(open_at)
        if close is None:
            return []
        t = self.t
        out = []  # type: List[str]
        is_obj = t[open_at][1] == "{"
        j = open_at + 1
        while j < close:
            if t[j][1] == ",":
                j += 1
                continue
            if t[j][1] == "...":
                j += 1
            target = j
            if is_obj and j < close:
                if t[j][1] == "[" and j in self.match:  # computed key
                    j = self.match[j]
                if j + 1 < close and t[j + 1][1] == ":":
                    target = j + 2
            if target < close:
                if t[target][1] in ("{", "[") and target in self.match:
                    out.extend(self._pattern_names(target))
                elif t[target][0] == ID:
                    out.append(t[target][1])
            j = target
            while j < close and t[j][1] != ",":  # rest of the entry (defaults)
                if t[j][1] in ("(", "[", "{") and j in self.match:
                    j = self.match[j]
                j += 1
        return out

    def parse_function_rest(self, idx: int) -> None:
        """Cursor at '<' or '(' of a function: tparams, params, return type, body."""
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = self.skip_angle()
        if self.is_op("("):
            d["params"] = self.parse_params(idx)
        if self.eat(":"):
            d["ret"] = self.parse_type()
        if self.is_op("{"):
            self.p += 1
            self.scan(idx, until="}")
            self.eat("}")
        d["end"] = self.line(-1)

    def parse_function(self, parent: int, mods: List[str], annos: List[str],
                       default: bool = False) -> int:
        line = self.line()
        self.p += 1  # function
        self.eat("*")
        name = self.default_name if default else "<anonymous>"
        if self.is_id():
            name = self.text()
            self.p += 1
        idx = self.add_decl("fun", name, parent, line, mods, annos, tparams=[], recv="",
                            params=[], ret="")
        self.parse_function_rest(idx)
        return idx

    def _arrow_at(self, k: int = 0) -> bool:
        """Does an arrow function start at cursor+k?"""
        j = self.p + k
        t = self.t
        if j < self.n and t[j][0] == ID and t[j][1] == "async" and j + 1 < self.n \
                and t[j + 1][1] != "=>":
            j += 1
        if j < self.n and t[j][1] == "<":
            save = self.p
            self.p = j
            self.skip_angle()
            j = self.p
            self.p = save
        if j >= self.n:
            return False
        if t[j][1] == "(" and t[j][0] == OP:
            close = self.match.get(j)
            if close is None or close + 1 >= self.n:
                return False
            nxt = t[close + 1][1]
            if nxt == "=>":
                return True
            if nxt == ":":
                save = self.p
                self.p = close + 2
                self._type()
                ok = self.is_op("=>")
                self.p = save
                return ok
            return False
        return t[j][0] == ID and j + 1 < self.n and t[j + 1][1] == "=>"

    def _function_value_at(self, k: int = 0) -> bool:
        return self._arrow_at(k) or self.is_id(k, "function") or (
            self.is_id(k, "async") and self.is_id(k + 1, "function"))

    def _function_value(self, idx: int) -> None:
        """Cursor at an arrow function or function expression assigned to idx."""
        if self.is_id(0, "async") and self.is_id(1, "function"):
            self.p += 1
        if self.is_id(0, "function"):
            self.p += 1
            self.eat("*")
            if self.is_id():
                self.p += 1
            self.parse_function_rest(idx)
        else:
            self.parse_arrow(idx)

    def parse_arrow(self, idx: int) -> None:
        d = self.decls[idx]
        d.setdefault("tparams", [])
        if self.is_id(0, "async") and not self.is_op("=>", 1):
            self.p += 1
        if self.is_op("<"):
            d["tparams"] = self.skip_angle()
        if self.is_op("("):
            d["params"] = self.parse_params(idx)
        elif self.is_id():
            d["params"] = [[self.text(), "", False, False, [], False]]
            self.p += 1
        if self.eat(":"):
            d["ret"] = self.parse_type()
        self.eat("=>")
        if self.is_op("{"):
            self.p += 1
            self.scan(idx, until="}")
            self.eat("}")
        else:
            self.scan(idx, expr=True, stops=(",",))
        d["end"] = self.line(-1)

    def parse_interface(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1
        name = self.text()
        self.p += 1
        idx = self.add_decl("interface", name, parent, line, mods, annos, tparams=[], params=[],
                            supers=[], inject_ctor=False)
        d = self.decls[idx]
        if self.is_op("<"):
            d["tparams"] = self.skip_angle()
        if self.is_id(0, "extends"):
            self.p += 1
            while self.p < self.n:
                tline = self.line()
                text = self.parse_type()
                if not text:
                    break
                d["supers"].append([text, False, tline])
                if not self.eat(","):
                    break
        if self.is_op("{"):
            self.p += 1
            self.parse_type_members(idx)
        d["end"] = self.line(-1)
        return idx

    def parse_type_members(self, owner: int) -> None:
        while self.p < self.n:
            if self.is_op("}"):
                self.p += 1
                return
            if self.eat(";") or self.eat(","):
                continue
            before = self.p
            line = self.line()
            while self.is_id() and self.text() in ("readonly",) and (self.is_id(1) or self.is_op("[", 1)):
                self.p += 1
            if self.is_op("["):
                self.skip_balanced()
                self.eat("?")
                if self.eat(":"):
                    self.parse_type()
                continue
            if self.is_op("(") or self.is_op("<") or (self.is_id(0, "new") and self.is_op("(", 1)):
                if self.is_id():
                    self.p += 1
                if self.is_op("<"):
                    self.skip_angle()
                self.skip_balanced()
                if self.eat(":"):
                    self.parse_type()
                continue
            accessor = self.text() in ("get", "set") and self.is_id() and self.is_id(1)
            if accessor:
                self.p += 1
            name = self._member_name()
            if name is None:
                self.p = max(self.p, before + 1)
                continue
            self.eat("?")
            if (self.is_op("(") or self.is_op("<")) and not accessor:
                idx = self.add_decl("fun", name, owner, line, [], [], tparams=[], recv="",
                                    params=[], ret="")
                d = self.decls[idx]
                if self.is_op("<"):
                    d["tparams"] = self.skip_angle()
                if self.is_op("("):
                    d["params"] = self.parse_params(idx)
                if self.eat(":"):
                    d["ret"] = self.parse_type()
                d["end"] = self.line(-1)
            else:
                ptype = ""
                if self.is_op("("):
                    self.skip_balanced()
                if self.eat(":"):
                    ptype = self.parse_type()
                self.add_decl("prop", name, owner, line, [], [], recv="", type=ptype, delegate="",
                              has_init=False)
            if self.p == before:
                self.p += 1

    def parse_enum(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1  # enum
        name = self.text() if self.is_id() else "<anonymous>"
        self.p += 1
        idx = self.add_decl("enum", name, parent, line, mods, annos, tparams=[], params=[],
                            supers=[], inject_ctor=False)
        if self.is_op("{"):
            close = self.match.get(self.p)
            self.p += 1
            while self.p < self.n and not self.is_op("}"):
                before = self.p
                if self.is_id() or self.is_str():
                    self.add_decl("entry", self.text() or "<computed>", idx, self.line(), [], [])
                    self.p += 1
                    if self.eat("="):
                        self.scan(self.init_of(idx), until="}", stops=(",",))
                if not self.eat(",") and self.p == before:
                    self.p += 1
            self.p = (close + 1) if close is not None else self.p + 1
        self.decls[idx]["end"] = self.line(-1)
        return idx

    def parse_typealias(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1
        name = self.text()
        self.p += 1
        tparams = self.skip_angle() if self.is_op("<") else []
        target = ""
        if self.eat("="):
            target = self.parse_type()
        idx = self.add_decl("typealias", name, parent, line, mods, annos, type=target, tparams=tparams)
        self.decls[idx]["end"] = self.line(-1)
        return idx

    def parse_namespace(self, parent: int, mods: List[str], annos: List[str]) -> int:
        line = self.line()
        self.p += 1
        if self.is_str():  # declare module 'x' { … }: an ambient module, skipped
            self.p += 1
            if self.is_op("{"):
                self.skip_balanced()
            return -1
        name = self.qualified_name()
        idx = self.add_decl("namespace", name or "<anonymous>", parent, line, mods, annos,
                            tparams=[], params=[], supers=[], inject_ctor=False)
        if self.is_op("{"):
            self.p += 1
            self.parse_statements(idx)
        self.decls[idx]["end"] = self.line(-1)
        return idx

    def parse_var_decl(self, parent: int, mods: List[str], annos: List[str]) -> List[str]:
        """Top-level (or namespace-level) const / let / var."""
        kw = self.text()
        self.p += 1
        names = []  # type: List[str]
        while self.p < self.n:
            line = self.line()
            if self.is_op("{") or self.is_op("["):
                open_at = self.p
                self.skip_balanced()
                if self.eat(":"):
                    self.parse_type()
                if self.is_op("=") and self.is_id(1, "require") and self.is_op("(", 2) and self.is_str(3) \
                        and self._value_ends(5):
                    close = self.match.get(open_at, open_at)
                    if self.t[open_at][1] == "{":
                        self._require_pattern(open_at, close, self.text(3), line)
                    self.is_module = True
                    self.p += 5
                elif self.eat("="):
                    owner = self.module_owner() if parent < 0 else self.init_of(parent)
                    self.scan(owner, expr=True, stops=(",",))
                if not self.eat(","):
                    break
                continue
            if not self.is_id():
                break
            name = self.text()
            self.p += 1
            self.eat("!")
            ptype = self.parse_type() if self.eat(":") else ""
            if self.is_op("="):
                if self.is_id(1, "require") and self.is_op("(", 2) and self.is_str(3) and self._value_ends(5):
                    self.imports.append([self.text(3), "=", name, line])
                    self.is_module = True
                    self.p += 5
                    if not self.eat(","):
                        break
                    continue
                if self._function_value_at(1):
                    self.p += 1
                    idx = self.add_decl("fun", name, parent, line, mods, annos, tparams=[], recv="",
                                        params=[], ret="", ftype=ptype, kw=kw)
                    self._function_value(idx)
                    names.append(name)
                    if not self.eat(","):
                        break
                    continue
                if self.is_id(1, "class"):
                    self.p += 1
                    idx = self.parse_class(parent, mods, annos, name)
                    names.append(name)
                    if not self.eat(","):
                        break
                    continue
                if self.is_op("{", 1) and self._object_has_functions(self.p + 1):
                    self.p += 1
                    self.parse_object_literal(parent, name, line, mods, annos)
                    names.append(name)
                    if not self.eat(","):
                        break
                    continue
            idx = self.add_decl("prop", name, parent, line, mods, annos, recv="", type=ptype,
                                delegate="", has_init=False, kw=kw)
            names.append(name)
            d = self.decls[idx]
            if self.eat("="):
                d["has_init"] = True
                self._init_type(d, 0)
                self.scan(idx, expr=True, stops=(",",))
            d["end"] = self.line(-1)
            if not self.eat(","):
                break
        return names

    def _object_has_functions(self, open_at: int) -> bool:
        close = self.match.get(open_at)
        if close is None:
            return False
        t = self.t
        j = open_at + 1
        while j < close:
            kind, s = t[j][0], t[j][1]
            if kind == OP and s in ("(", "[", "{"):
                cl = self.match.get(j)
                if cl is None:
                    return False
                if s == "(" and cl + 1 < self.n and t[cl + 1][1] in ("{", "=>"):
                    return True
                j = cl + 1
                continue
            if s == "=>" or (kind == ID and s == "function"):
                return True
            j += 1
        return False

    def parse_object_literal(self, parent: int, name: str, line: int, mods: List[str],
                             annos: List[str]) -> int:
        """At '{' of an object literal that is a declaration (a module object)."""
        idx = self.add_decl("object", name, parent, line, mods, annos, tparams=[], params=[],
                            supers=[], inject_ctor=False)
        close = self.match.get(self.p)
        self.p += 1
        end = close if close is not None else self.n
        while self.p < end:
            if self.eat(","):
                continue
            before = self.p
            mline = self.line()
            if self.eat("..."):
                self.scan(self.init_of(idx), until="}", stops=(",",))
                continue
            acc = None
            if self.is_id() and self.text() in ("get", "set", "async") and (
                    self.is_id(1) or self.is_str(1) or self.is_op("[", 1) or self.is_op("*", 1)):
                acc = self.text()
                self.p += 1
            self.eat("*")
            key = self._member_name()
            if key is None:
                self.p = max(self.p, before + 1)
                continue
            if self.is_op("(") or self.is_op("<"):
                kind = "prop" if acc in ("get", "set") else "fun"
                fidx = self.add_decl(kind, key, idx, mline, ["async"] if acc == "async" else [], [],
                                     tparams=[], recv="", params=[], ret="", type="")
                self.parse_function_rest(fidx)
            elif self.eat(":"):
                if self._function_value_at():
                    fidx = self.add_decl("fun", key, idx, mline, [], [], tparams=[], recv="",
                                         params=[], ret="")
                    self._function_value(fidx)
                else:
                    pidx = self.add_decl("prop", key, idx, mline, [], [], recv="", type="",
                                         delegate="", has_init=True)
                    pd = self.decls[pidx]
                    if self.is_id() and (self.p + 1 >= end or self.is_op(",", 1)):
                        pd["alias"] = self.text()
                    self._init_type(pd, 0)
                    self.scan(pidx, until="}", stops=(",",))
                    pd["end"] = self.line(-1)
            else:  # shorthand `a` / `a = default`
                pidx = self.add_decl("prop", key, idx, mline, [], [], recv="", type="",
                                     delegate="", has_init=True, alias=key)
            if self.p == before:
                self.p += 1
        self.p = end + 1
        self.decls[idx]["end"] = self.line(-1)
        return idx

    # ---- code ----

    def _ends_expression(self) -> bool:
        if self.p == 0 or not self.nl():
            return False
        prev = self.t[self.p - 1][1]
        cur = self.t[self.p][1]
        return prev not in _CONT_AFTER and cur not in _CONT_BEFORE

    def scan(self, owner: int, until: Optional[str] = None, stops: Tuple[str, ...] = (),
             expr: bool = False, top: bool = False) -> None:
        depth = 0
        t = self.t
        start = self.p
        while self.p < self.n:
            kind, s, line, _nl = t[self.p]
            if depth == 0:
                if kind == OP and s in stops:
                    return
                if kind == OP and s in (")", "]", "}"):
                    if s == until or expr or stops or until is None:
                        return
                    if until == ")" and s == "}":
                        return
                    self.p += 1
                    continue
                if expr and ((s == ";" and kind == OP) or self._ends_expression()):
                    return
                if top and self.p > start and self._stmt_start() and (
                        self._decl_here() or self._is_cjs_export()):
                    return
            if kind == OP:
                if s in ("(", "[", "{"):
                    depth += 1
                elif s in (")", "]", "}"):
                    depth -= 1
                elif s == "@":
                    self.parse_annotations()
                    continue
                elif s == "<jsx":
                    self._jsx_call(owner)
                    continue
                elif s.startswith("<attr:"):
                    self._attr_ref(owner, s[6:])
                    self.p += 1
                    continue
                elif s == "?." and self.is_op("(", 1) and self.p > 0 and t[self.p - 1][0] == ID:
                    self._record_call(owner, self.p - 1, self.p + 1)
                self.p += 1
                continue
            if kind != ID:
                self.p += 1
                continue
            prev = t[self.p - 1][1] if self.p > 0 else ""
            dotted = self.p > 0 and t[self.p - 1][0] == OP and prev in (".", "?.")
            if not dotted:
                if s == "function":
                    k = 2 if self.is_op("*", 1) else 1
                    if self.is_id(k) and depth == 0 and self._stmt_start_at(self.p):
                        self.parse_function(owner, [], [])
                        continue
                    self.p += 1
                    continue
                if s == "class" and self.is_id(1) and self.text(1) not in ("extends", "implements"):
                    self.parse_class(owner, [], [])
                    continue
                if s in ("const", "let", "var") and (self.is_id(1) or self.is_op("{", 1) or self.is_op("[", 1)):
                    self._local_var(owner)
                    continue
                if s in ("as", "satisfies") and prev not in ("{", ","):
                    if self.is_id(1) and _CAP_RE.match(self.text(1)):
                        self.trefs.append([owner, self.text(1), line, s])
                elif s == "instanceof" and self.is_id(1) and _CAP_RE.match(self.text(1)):
                    self.trefs.append([owner, self.text(1), line, s])
                elif s in ("import", "require") and self.is_op("(", 1) and self.is_str(2):
                    self.imports.append([self.text(2), None, None, line])
                    if s == "require":
                        self.is_module = True
                    self.p += 1
                    continue
                elif s == "this" and owner >= 0 and self.is_op(".", 1) and self.is_id(2) \
                        and self.is_op("=", 3):
                    self._this_field(owner)
                if _CAP_RE.match(s) and self.is_op(".", 1) and prev not in ("@", "new"):
                    self.trefs.append([owner, s, line, "qual"])
                if s in TS_NOT_CALLS:
                    self.p += 1
                    continue
            nxt = t[self.p + 1][1] if self.p + 1 < self.n and t[self.p + 1][0] == OP else ""
            if nxt == "(":
                close = self.match.get(self.p + 1)
                if not (close is not None and close + 1 < self.n and t[close + 1][1] == "{"
                        and t[close + 1][0] == OP):  # `name(…) {` is a method definition
                    self._record_call(owner, self.p, self.p + 1)
            elif nxt == "<":
                close = self._generic_call_close(self.p + 1)
                if close is not None:  # foo<A, B>(…): the type arguments hold no calls,
                    self._record_call(owner, self.p, close + 1)  # and their commas end nothing
                    self.p = close + 1
                    continue
            elif prev == "new" and nxt != "." and _CAP_RE.match(s):
                self._record_call(owner, self.p, self.p + 1)
            self.p += 1

    def _stmt_start_at(self, i: int) -> bool:
        if i == 0:
            return True
        return self.t[i][3] or self.t[i - 1][1] in (";", "{", "}")

    def _is_cjs_export(self) -> bool:
        return (self.is_id(0, "module") and self.is_op(".", 1) and self.is_id(2, "exports")) or \
            (self.is_id(0, "exports") and self.is_op(".", 1) and self.is_id(2) and self.is_op("=", 3))

    def _generic_call_close(self, i: int) -> Optional[int]:
        """At '<' (index i): if this is foo<T, U>(, the index of the closing '>'.
        Brackets must balance inside and no statement keyword may appear,
        so `i < n) { return x > (y)` is not a call of i."""
        depth = 0
        inner = 0
        t = self.t
        for j in range(i, min(i + 60, self.n)):
            kind, s = t[j][0], t[j][1]
            if s == "<":
                depth += 1
            elif s == ">":
                depth -= 1
                if depth == 0:
                    nxt = t[j + 1][1] if j + 1 < self.n else ""
                    return j if nxt == "(" and inner == 0 else None
            elif kind == OP and s in ("(", "[", "{"):
                inner += 1
            elif kind == OP and s in (")", "]", "}"):
                inner -= 1
                if inner < 0:
                    return None
            elif kind == ID and s in TS_NOT_CALLS and s not in ("keyof", "typeof", "readonly",
                                                                 "null", "undefined", "unique",
                                                                 "infer", "is", "extends"):
                return None
            elif kind in (ID, STR, NUM) or s in _GENERIC_OK:
                continue
            else:
                return None
        return None

    def _record_call(self, owner: int, name_at: int, open_at: int) -> None:
        nargs = 0
        if open_at < self.n and self.t[open_at][1] == "(":
            close = self.match.get(open_at)
            if close is not None:
                nargs = self._count_args(open_at, close)
        self.calls.append([owner, self.t[name_at][1], self._receiver(name_at),
                           self.t[name_at][2], nargs, False])

    def _jsx_call(self, owner: int) -> None:
        """At OP '<jsx': <Foo …> / <Foo.Bar …> is a call of Foo / Foo.Bar."""
        line = self.line()
        self.p += 1
        if not (self.is_id() and _CAP_RE.match(self.text())):
            while self.is_id() or (self.is_op(".") and self.is_id(1)):
                self.p += 1
            return
        segs = [self.text()]
        self.p += 1
        while self.is_op(".") and self.is_id(1):
            segs.append(self.text(1))
            self.p += 2
        recv = [[s, 0] for s in segs[:-1]] or None
        self.calls.append([owner, segs[-1], recv, line, -1, False])

    def _attr_ref(self, owner: int, attr: str) -> None:
        """At OP '<attr:name' (before '{'): onClick={handler}, onSave={this.save},
        component={Page} / render={renderRow} -> a reference to the function.
        Other attribute values (label={x}) are data, not references."""
        if not (_HANDLER_ATTR_RE.match(attr) or attr in _REF_ATTRS):
            if not (self.is_id(2) and _CAP_RE.match(self.text(2)) and self.is_op("}", 3)):
                return
        j = self.p + 2
        t = self.t
        segs = []
        while j < self.n and t[j][0] == ID:
            segs.append(t[j][1])
            if j + 2 < self.n and t[j + 1][1] in (".", "?.") and t[j + 2][0] == ID:
                j += 2
                continue
            j += 1
            break
        if not segs or j >= self.n or t[j][1] != "}" or segs[0] in TS_NOT_CALLS - {"this"}:
            return
        if segs == ["this"]:
            return
        recv = [[s, 0] for s in segs[:-1]] or None
        self.calls.append([owner, segs[-1], recv, t[self.p + 2][2], -1, True])

    def _this_field(self, owner: int) -> None:
        """`this.x = …` in a constructor (JS classes): declare field x."""
        d = self.decls[owner]
        cls = d.get("parent", -1)
        if d["kind"] != "init" or cls < 0 or self.decls[cls]["kind"] != "class":
            return
        name = self.text(2)
        seen = self._fields.setdefault(cls, set())
        if name in seen:
            return
        seen.add(name)
        pidx = self.add_decl("prop", name, cls, self.line(), [], [], recv="", type="", delegate="",
                             has_init=True)
        pd = self.decls[pidx]
        self._init_type(pd, 4)
        if not pd.get("itype") and self.is_id(4) and self._value_ends(5):
            for p in d.get("params", ()):
                if p[0] == self.text(4) and p[1]:
                    pd["type"] = p[1]

    def _local_var(self, owner: int) -> None:
        self.p += 1  # const / let / var
        locals_ = self.decls[owner].setdefault("locals", {}) if owner >= 0 else {}
        while self.p < self.n:
            line = self.line()
            if self.is_op("{") or self.is_op("["):
                open_at = self.p
                names = self._pattern_names(open_at)
                self.skip_balanced()
                if owner >= 0 and names:
                    self.decls[owner].setdefault("vars", []).extend(names)
                if self.eat(":"):
                    self.parse_type()
                if self.is_op("=") and self.is_id(1, "require") and self.is_op("(", 2) and self.is_str(3) \
                        and self.t[open_at][1] == "{" and self._value_ends(5):
                    self._require_pattern(open_at, self.match.get(open_at, open_at), self.text(3), line)
                    self.is_module = True
                    self.p += 5
                return  # any other initializer is scanned by the caller
            if not self.is_id():
                return
            name = self.text()
            self.p += 1
            self.eat("!")
            tname = self.parse_type() if self.eat(":") else ""
            if tname:
                locals_[name] = [tname, "decl"]
            elif owner >= 0:
                self.decls[owner].setdefault("vars", []).append(name)
            if not self.is_op("="):
                if self.eat(","):
                    continue
                return
            if self._function_value_at(1):
                self.p += 1
                fidx = self.add_decl("fun", name, owner, line, [], [], tparams=[], recv="",
                                     params=[], ret="", ftype=tname)
                self._function_value(fidx)
                if self.eat(","):
                    continue
                return
            if self.is_id(1, "require") and self.is_op("(", 2) and self.is_str(3):
                self.imports.append([self.text(3), "=", name, line])
                self.is_module = True
                return
            if not tname:
                tmp = {}  # type: dict
                self._init_type(tmp, 1)
                if tmp.get("itype"):
                    locals_[name] = [tmp["itype"], "ctor"]
            return


def signature(d: dict) -> str:
    """One-line signature of a decl, e.g. 'async load(id: string): Promise<User>'."""
    k = d["kind"]
    if d["name"] == "<module>":
        return "top-level code"
    mods = " ".join(m for m in d.get("mods", ()) if m in (
        "private", "protected", "static", "abstract", "override", "readonly", "async", "declare"))
    attrs = " ".join("@" + a for a in d.get("annos", ()))
    pre = " ".join(x for x in (attrs, mods) if x)
    pre = pre + " " if pre else ""
    tp = f"<{', '.join(d['tparams'])}>" if d.get("tparams") else ""

    def param(p):
        name = "{…}" if p[0] == "<destructured>" else p[0]
        return f"{'...' if p[5] else ''}{name}{'?' if p[3] else ''}{': ' + p[1] if p[1] else ''}"

    if k in ("fun", "init"):
        ps = ", ".join(param(p) for p in d.get("params", ()))
        if k == "init":
            s = f"{pre}constructor({ps})"
        elif d.get("ftype") and not d.get("ret"):
            s = f"{pre}{d.get('kw') or 'const'} {d['name']}: {d['ftype']} = ({ps}) => …"
        elif d.get("kw"):  # const f = (…) => …
            s = f"{pre}{d['kw']} {d['name']} = {tp}({ps}){': ' + d['ret'] if d.get('ret') else ''} => …"
        else:
            s = f"{pre}{'function ' if d.get('parent', -1) < 0 else ''}{d['name']}{tp}({ps})"
            if d.get("ret"):
                s += f": {d['ret']}"
    elif k in ("class", "interface", "enum", "namespace"):
        s = f"{pre}{k} {d['name']}{tp}"
        ext = [x[0] for x in d.get("supers", ()) if x[1]]
        impl = [x[0] for x in d.get("supers", ()) if not x[1]]
        if ext:
            s += " extends " + ", ".join(ext)
        if impl:
            s += (" extends " if k == "interface" else " implements ") + ", ".join(impl)
    elif k == "object":
        s = f"const {d['name']} = {{ … }}" if d["name"] != "default" else "export default { … }"
    elif k == "prop":
        s = f"{pre}{d['kw'] + ' ' if d.get('kw') else ''}{d['name']}"
        if d.get("type"):
            s += f": {d['type']}"
        elif d.get("itype"):
            s += f" = new {d['itype']}(…)"
        elif d.get("dtype"):
            s += f" = inject({d['dtype']})"
        elif d.get("alias") and d["alias"] != d["name"]:
            s += f": {d['alias']}"
    elif k == "typealias":
        s = f"type {d['name']}{tp} = {d.get('type', '')}"
    else:
        s = d["name"]
    return s if len(s) <= 180 else s[:177] + "..."


def parse_ts_with_idents(source: str, rel: str = "x.ts") -> Tuple[dict, List[str]]:
    """Facts plus every identifier token in code (not in strings, comments or
    JSX text), for the repo map's references."""
    parser = _TsParser(source, rel)
    facts = parser.run()
    return facts, [t[1].lstrip("#") for t in parser.t if t[0] == ID]


def parse_ts(source: str, rel: str = "x.ts") -> dict:
    """Parse one TypeScript / JavaScript file into facts (see module docstring)."""
    return _TsParser(source, rel).run()
