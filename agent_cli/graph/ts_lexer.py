"""TypeScript / JavaScript tokenizer for the code graph (no tree-sitter).

Same token shape as kotlin_lexer: ``(kind, text, line, nl)``. Specifics:
  * strings '…' and "…"; template literals `…${expr}…` (nested) with the
    code in ${} lexed in place. A STR token's text is the literal's content
    when it is plain and short (import specifiers, require('x')), else "".
  * regex literals (/…/flags where an operand cannot precede) become a STR.
  * `#private` names and `$`-identifiers are IDs; `?.` is one token unless
    it is `? .5`; `>>` is never merged (generics close with single `>`).
  * JSX (in .tsx/.jsx/.js): an element in expression position is emitted as
    OP "<jsx" followed by its tag name tokens (Foo / Foo.Bar; nothing for
    a fragment or a dashed/namespaced tag). Attribute and child `{…}` code
    is lexed in place, and an attribute value `{…}` is preceded by OP
    "<attr:name" (e.g. "<attr:onClick"). Text, string attributes and closing
    tags are dropped. Tokens
    inside JSX never carry a newline flag (so JSX never ends a statement).
"""
from typing import List

from .kotlin_lexer import ID, NUM, OP, STR, Token, _Lexer

_MULTI = (
    "...", "===", "!==", "**=", "??=", "&&=", "||=",
    "=>", "??", "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
    "%=", "**", "|=", "&=", "^=",
)
# Keywords after which an expression (so a regex or JSX) may start.
_EXPR_KEYWORDS = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw",
    "case", "do", "else", "yield", "await", "default",
})
_JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")


def _ident_start(c: str) -> bool:
    return c.isalpha() or c == "_" or c == "$"


def _ident_char(c: str) -> bool:
    return c.isalnum() or c == "_" or c == "$"


class _TsLexer(_Lexer):
    def __init__(self, src: str, jsx: bool):
        super().__init__(src)
        self.jsx = jsx
        self.in_jsx = 0

    def emit(self, kind: str, text: str, line: int) -> None:
        if self.in_jsx:
            self.nl = False
        super().emit(kind, text, line)

    # -- context --

    def _operand_before(self) -> bool:
        """Does the previous token end an operand (so '/' is division and '<'
        is less-than)?"""
        if not self.out:
            return False
        kind, text = self.out[-1][0], self.out[-1][1]
        if kind in (NUM, STR):
            return True
        if kind == ID:
            return text not in _EXPR_KEYWORDS
        return text in (")", "]", "}")

    # -- literals --

    def _quoted(self, q: str) -> None:
        s, n = self.s, self.n
        start = self.i + 1
        j = start
        plain = True
        while j < n:
            c = s[j]
            if c == "\\":
                plain = False
                j += 2
                continue
            if c == q or c == "\n":
                break
            j += 1
        content = s[start:min(j, n)]
        self.emit(STR, content if plain and len(content) <= 200 else "", self.line)
        self.i = j + 1 if j < n and s[j] == q else j

    def _template(self) -> None:
        s, n = self.s, self.n
        line = self.line
        self.i += 1
        start = self.i
        plain = True
        while self.i < n:
            c = s[self.i]
            if c == "\\":
                plain = False
                self.i += 2
                continue
            if c == "`":
                break
            if c == "$" and self.i + 1 < n and s[self.i + 1] == "{":
                plain = False
                self.i += 2
                saved = self.in_jsx
                self.in_jsx = 0
                self._template_expr()
                self.in_jsx = saved
                continue
            if c == "\n":
                self.line += 1
                plain = False
            self.i += 1
        content = s[start:self.i]
        self.i = min(self.i + 1, n)
        self.emit(STR, content if plain and len(content) <= 200 else "", line)

    def _regex(self) -> bool:
        """At '/': consume a regex literal if there is one on this line."""
        s, n = self.s, self.n
        j = self.i + 1
        in_class = False
        while j < n:
            c = s[j]
            if c == "\\":
                j += 2
                continue
            if c == "\n":
                return False
            if in_class:
                if c == "]":
                    in_class = False
            elif c == "[":
                in_class = True
            elif c == "/":
                j += 1
                while j < n and _ident_char(s[j]):
                    j += 1
                self.emit(STR, "", self.line)
                self.i = j
                return True
            j += 1
        return False

    # -- JSX --

    def _jsx_start(self) -> bool:
        """At '<' in expression position: is this a JSX element?"""
        s, n = self.s, self.n
        j = self.i + 1
        if j >= n:
            return False
        if s[j] == ">":
            return True
        if not _ident_start(s[j]):
            return False
        k = j
        while k < n and (_ident_char(s[k]) or s[k] in ".:-"):
            k += 1
        m = k
        while m < n and s[m] in " \t":
            m += 1
        # generic arrow in .tsx: <T,>(x: T) => x / <T extends U>(…)
        if m < n and s[m] == ",":
            return False
        if s.startswith("extends", m) and m + 7 < n and not _ident_char(s[m + 7]):
            return False
        return True

    def _newline(self) -> None:
        self.line += 1
        self.i += 1

    def _jsx_code(self) -> None:
        """At '{' inside JSX: emit '{', lex code to the matching '}', emit '}'."""
        line = self.line
        self.emit(OP, "{", line)
        self.i += 1
        saved = self.in_jsx
        self.in_jsx = 0
        depth = 0
        while self.i < self.n:
            c = self.s[self.i]
            if c == "}" and depth == 0:
                break
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            self._one()
        self.in_jsx = saved
        self.nl = False
        self.emit(OP, "}", self.line)
        self.i = min(self.i + 1, self.n)

    def _jsx_element(self) -> bool:
        """At '<'. Returns False (nothing consumed) if the element never closes."""
        s, n = self.s, self.n
        save = (self.i, self.line, len(self.out), self.nl)
        self.in_jsx += 1
        ok = self._jsx_element_inner()
        self.in_jsx -= 1
        if not ok:
            self.i, self.line, cut, self.nl = save
            del self.out[cut:]
        return ok

    def _jsx_element_inner(self) -> bool:
        s, n = self.s, self.n
        line = self.line
        self.i += 1
        if self.i < n and s[self.i] == ">":  # fragment
            self.emit(OP, "<jsx", line)
            self.i += 1
            return self._jsx_children()
        j = self.i
        while j < n and (_ident_char(s[j]) or s[j] in ".:-"):
            j += 1
        name = s[self.i:j]
        self.i = j
        self.emit(OP, "<jsx", line)
        if name and "-" not in name and ":" not in name:
            for k, part in enumerate(name.split(".")):
                if k:
                    self.emit(OP, ".", line)
                if part:
                    self.emit(ID, part, line)
        attr = ""
        while self.i < n:  # attributes
            c = s[self.i]
            if c == "\n":
                self._newline()
            elif c.isspace():
                self.i += 1
            elif s.startswith("/>", self.i):
                self.i += 2
                return True
            elif c == ">":
                self.i += 1
                return self._jsx_children()
            elif c == "{":
                self._jsx_code()
            elif c == "=":
                self.i += 1
                while self.i < n and s[self.i] in " \t":
                    self.i += 1
                if self.i < n and s[self.i] == "{":
                    self.emit(OP, "<attr:" + attr, self.line)
                    self._jsx_code()
                elif self.i < n and s[self.i] == "<" and self._jsx_start():
                    if not self._jsx_element():
                        return False
            elif c in "\"'":
                end = s.find(c, self.i + 1)
                if end < 0:
                    return False
                self.line += s.count("\n", self.i, end)
                self.i = end + 1
            elif c == "<":
                return False  # not JSX after all
            elif _ident_start(c):
                j = self.i + 1
                while j < n and (_ident_char(s[j]) or s[j] in "-:"):
                    j += 1
                attr = s[self.i:j]
                self.i = j
            else:
                self.i += 1
        return False

    def _jsx_children(self) -> bool:
        s, n = self.s, self.n
        while self.i < n:
            c = s[self.i]
            if c == "<":
                if self.i + 1 < n and s[self.i + 1] == "/":
                    end = s.find(">", self.i)
                    if end < 0:
                        return False
                    self.line += s.count("\n", self.i, end)
                    self.i = end + 1
                    return True
                if self._jsx_start():
                    if not self._jsx_element():
                        return False
                else:
                    self.i += 1
            elif c == "{":
                self._jsx_code()
            elif c == "\n":
                self._newline()
            else:
                self.i += 1
        return False

    # -- main loop --

    def _one(self) -> None:
        s = self.s
        c = s[self.i]
        if c == "\n":
            self.line += 1
            self.nl = True
            self.i += 1
        elif c.isspace():
            self.i += 1
        elif s.startswith("//", self.i) or (self.i == 0 and s.startswith("#!")):
            j = s.find("\n", self.i)
            self.i = self.n if j < 0 else j
        elif s.startswith("/*", self.i):
            j = s.find("*/", self.i + 2)
            self._advance_to(self.n if j < 0 else j + 2)
        elif c in "\"'":
            self._quoted(c)
        elif c == "`":
            self._template()
        elif _ident_start(c) or (c == "#" and self.i + 1 < self.n and _ident_start(s[self.i + 1])):
            j = self.i + 1
            while j < self.n and _ident_char(s[j]):
                j += 1
            self.emit(ID, s[self.i:j], self.line)
            self.i = j
        elif c.isdigit():
            j = self.i + 1
            while j < self.n:
                d = s[j]
                if _ident_char(d):
                    j += 1
                elif d == "." and j + 1 < self.n and s[j + 1].isdigit():
                    j += 1
                elif d in "+-" and s[j - 1] in "eE" and not s[self.i:j].lower().startswith("0x"):
                    j += 1
                else:
                    break
            self.emit(NUM, s[self.i:j], self.line)
            self.i = j
        elif c == "/" and not self._operand_before() and self._regex():
            return
        elif c == "<" and self.jsx and not self._operand_before() and self._jsx_start() \
                and self._jsx_element():
            return
        elif c == "?" and s.startswith("?.", self.i) and not (
                self.i + 2 < self.n and s[self.i + 2].isdigit()):
            self.emit(OP, "?.", self.line)
            self.i += 2
        else:
            for op in _MULTI:
                if s.startswith(op, self.i):
                    self.emit(OP, op, self.line)
                    self.i += len(op)
                    return
            self.emit(OP, c, self.line)
            self.i += 1


def jsx_enabled(rel: str) -> bool:
    """JSX is lexed in .tsx/.jsx and plain .js files (CRA-style), not .ts."""
    low = rel.lower()
    return low.endswith((".tsx",) + _JS_EXTS)


def tokenize_ts(source: str, jsx: bool = True) -> List[Token]:
    """Tokenize TypeScript / JavaScript. Never raises on malformed input."""
    return _TsLexer(source, jsx).run()
