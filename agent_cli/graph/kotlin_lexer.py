"""A small Kotlin tokenizer, written for the code graph (no tree-sitter).

It turns source into a flat token list and drops everything that is not
code: line and (nested) block comments, the text of string, raw-string and
char literals. Code inside string templates (``"${user.name()}"``) is kept:
its tokens are emitted in place so that calls made from templates are still
seen. ``$name`` short templates are dropped (they are references, not calls).

Each token is a tuple ``(kind, text, line, nl)`` where nl is True when a
real newline (outside string literals) separates it from the previous token,
and kind is one of
  ID   identifier or keyword (backticked names come without the backticks)
  NUM  number literal
  STR  a whole string / char literal (text is a placeholder)
  OP   punctuation / operator
Labels (``return@forEach``, ``this@Outer``, ``loop@``) are folded: the label
part is dropped so ``this@Outer.foo()`` reads as ``this.foo()``.
"""
from typing import List, Tuple

Token = Tuple[str, str, int, bool]

ID, NUM, STR, OP = "ID", "NUM", "STR", "OP"

_MULTI_OPS = (
    "?.", "?:", "::", "->", "..", "!!", "==", "!=", "<=", ">=", "&&", "||",
    "+=", "-=", "*=", "/=", "%=", "++", "--",
)
_TRIPLE_OPS = ("===", "!==", "..<")


def _is_ident_start(c: str) -> bool:
    return c.isalpha() or c == "_"


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


class _Lexer:
    def __init__(self, src: str):
        self.s = src
        self.n = len(src)
        self.i = 0
        self.line = 1
        self.out = []  # type: List[Token]
        self.nl = False  # newline seen since the last token

    def emit(self, kind: str, text: str, line: int) -> None:
        self.out.append((kind, text, line, self.nl))
        self.nl = False

    # -- helpers --

    def _advance_to(self, j: int) -> None:
        """Move to index j (through a comment), counting newlines on the way."""
        k = self.s.count("\n", self.i, j)
        self.line += k
        self.nl = self.nl or k > 0
        self.i = j

    def _skip_block_comment(self) -> None:
        """Skip /* ... */ with Kotlin's nesting rules."""
        s, n = self.s, self.n
        depth = 0
        j = self.i
        while j < n:
            if s.startswith("/*", j):
                depth += 1
                j += 2
            elif s.startswith("*/", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        self._advance_to(min(j, n))

    def _template_expr(self) -> None:
        """At the char after '${': lex code until the matching '}'."""
        depth = 0
        while self.i < self.n:
            c = self.s[self.i]
            if c == "}" and depth == 0:
                self.i += 1
                return
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            self._one()

    def _string(self, raw: bool) -> None:
        """At the opening quote(s). Emits template tokens, then one STR token."""
        s, n = self.s, self.n
        start_line = self.line
        self.i += 3 if raw else 1
        while self.i < n:
            c = s[self.i]
            if raw:
                if s.startswith('"""', self.i):
                    # A run of 4+ quotes: the extra ones belong to the content,
                    # the last three close the literal.
                    j = self.i + 3
                    while j < n and s[j] == '"':
                        j += 1
                    self.i = j
                    break
            else:
                if c == "\\":
                    self.i += 2
                    continue
                if c == '"':
                    self.i += 1
                    break
                if c == "\n":  # unterminated: stop at end of line
                    break
            if c == "$" and self.i + 1 < n and s[self.i + 1] == "{":
                self.i += 2
                self._template_expr()
                continue
            if c == "\n":
                self.line += 1
            self.i += 1
        self.emit(STR, '""', start_line)

    def _char(self) -> None:
        s, n = self.s, self.n
        j = self.i + 1
        while j < n and s[j] != "'" and s[j] != "\n":
            j += 2 if s[j] == "\\" else 1
        self.emit(STR, "''", self.line)
        self._advance_to(min(j + 1, n))

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
        elif s.startswith("//", self.i):
            j = s.find("\n", self.i)
            self.i = self.n if j < 0 else j
        elif s.startswith("/*", self.i):
            self._skip_block_comment()
        elif s.startswith('"""', self.i):
            self._string(raw=True)
        elif c == '"':
            self._string(raw=False)
        elif c == "'":
            self._char()
        elif c == "`":
            j = s.find("`", self.i + 1)
            if j < 0 or "\n" in s[self.i:j]:
                self.i += 1
                return
            self.emit(ID, s[self.i + 1:j], self.line)
            self.i = j + 1
        elif _is_ident_start(c):
            j = self.i + 1
            while j < self.n and _is_ident_char(s[j]):
                j += 1
            self.emit(ID, s[self.i:j], self.line)
            self.i = j
            # Label: ident immediately followed by @ident (return@x, this@X) or
            # a label definition (loop@ for ...). Drop the label part.
            if j < self.n and s[j] == "@":
                k = j + 1
                while k < self.n and _is_ident_char(s[k]):
                    k += 1
                self.i = k
        elif c.isdigit():
            j = self.i + 1
            while j < self.n:
                d = s[j]
                if _is_ident_char(d):
                    j += 1
                elif d == "." and j + 1 < self.n and s[j + 1].isdigit():
                    j += 1  # 1.5 but not 1..10
                elif d in "+-" and s[j - 1] in "eE" and not s[self.i:j].lower().startswith("0x"):
                    j += 1
                else:
                    break
            self.emit(NUM, s[self.i:j], self.line)
            self.i = j
        else:
            for op in _TRIPLE_OPS:
                if s.startswith(op, self.i):
                    self.emit(OP, op, self.line)
                    self.i += 3
                    return
            two = s[self.i:self.i + 2]
            if two in _MULTI_OPS:
                self.emit(OP, two, self.line)
                self.i += 2
                return
            self.emit(OP, c, self.line)
            self.i += 1

    def run(self) -> List[Token]:
        while self.i < self.n:
            self._one()
        return self.out


def tokenize(source: str) -> List[Token]:
    """Tokenize Kotlin source. Never raises on malformed input."""
    return _Lexer(source).run()
