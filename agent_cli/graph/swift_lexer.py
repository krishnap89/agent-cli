"""Swift tokenizer for the code graph (no tree-sitter).

Same token shape as kotlin_lexer: ``(kind, text, line, nl)``. Differences:
  * strings: "…", multi-line \"\"\"…\"\"\", raw #"…"# / ##"…"## (incl. multi-line);
    interpolation \\(expr) (\\#(expr) in raw strings) is lexed as code in place.
    A STR token's text is the literal's content when it has no interpolation
    and fits on one line (used for "logo" in UIImage(named: "logo")), else "".
  * compiler directives (#if/#elseif/#else/#endif/#warning/#error/
    #sourceLocation) are dropped with the rest of their line; code in every
    branch is kept. Other #words (#selector, #available, #Preview) become
    OP "#" followed by the word.
  * $0 / $name are identifiers; `?.` is one token when written without a
    space; nested /* */ comments are skipped.
"""
from typing import List

from .kotlin_lexer import ID, NUM, OP, STR, Token, _Lexer, _is_ident_char, _is_ident_start

_DIRECTIVES = frozenset({"if", "elseif", "else", "endif", "warning", "error", "sourceLocation"})
_MULTI = (
    "...", "..<", "===", "!==",
    "?.", "->", "??", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=",
)


class _SwiftLexer(_Lexer):
    def _interpolation(self) -> None:
        """At the char after '(' of \\( … ): lex code to the matching ')'."""
        depth = 0
        while self.i < self.n:
            c = self.s[self.i]
            if c == ")" and depth == 0:
                self.i += 1
                return
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            self._one()

    def _swift_string(self, hashes: int) -> None:
        """At the first '"' (after any '#'s)."""
        s, n = self.s, self.n
        start_line = self.line
        multi = s.startswith('"""', self.i)
        self.i += 3 if multi else 1
        close = ('"""' if multi else '"') + "#" * hashes
        esc = "\\" + "#" * hashes
        content_start = self.i
        plain = True
        while self.i < n:
            if s.startswith(close, self.i):
                content = s[content_start:self.i]
                self.i += len(close)
                break
            c = s[self.i]
            if c == "\n":
                if not multi:
                    content = s[content_start:self.i]
                    break
                self.line += 1
                plain = False
                self.i += 1
                continue
            if s.startswith(esc, self.i):
                j = self.i + len(esc)
                if j < n and s[j] == "(":
                    plain = False
                    self.i = j + 1
                    self._interpolation()
                    continue
                self.i = j + 1  # escaped char
                continue
            self.i += 1
        else:
            content = s[content_start:self.i]
        text = content if plain and not multi and len(content) <= 200 else ""
        self.emit(STR, text, start_line)

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
        elif c == '"':
            self._swift_string(0)
        elif c == "#":
            j = self.i
            while j < self.n and s[j] == "#":
                j += 1
            if j < self.n and s[j] == '"':  # raw string #"…"#
                hashes = j - self.i
                self.i = j
                self._swift_string(hashes)
                return
            k = j
            while k < self.n and _is_ident_char(s[k]):
                k += 1
            word = s[j:k]
            if word in _DIRECTIVES:
                e = s.find("\n", k)
                self.i = self.n if e < 0 else e
                return
            self.emit(OP, "#", self.line)
            self.i += 1
        elif c == "`":
            j = s.find("`", self.i + 1)
            if j < 0 or "\n" in s[self.i:j]:
                self.i += 1
                return
            self.emit(ID, s[self.i + 1:j], self.line)
            self.i = j + 1
        elif _is_ident_start(c) or (c == "$" and self.i + 1 < self.n and _is_ident_char(s[self.i + 1])):
            j = self.i + 1
            while j < self.n and _is_ident_char(s[j]):
                j += 1
            self.emit(ID, s[self.i:j], self.line)
            self.i = j
        elif c.isdigit():
            j = self.i + 1
            while j < self.n:
                d = s[j]
                if _is_ident_char(d):
                    j += 1
                elif d == "." and j + 1 < self.n and s[j + 1].isdigit():
                    j += 1
                elif d in "+-" and s[j - 1] in "eEpP" and not s[self.i:j].lower().startswith("0b"):
                    j += 1
                else:
                    break
            self.emit(NUM, s[self.i:j], self.line)
            self.i = j
        else:
            for op in _MULTI:
                if s.startswith(op, self.i):
                    self.emit(OP, op, self.line)
                    self.i += len(op)
                    return
            self.emit(OP, c, self.line)
            self.i += 1


def tokenize_swift(source: str) -> List[Token]:
    """Tokenize Swift source. Never raises on malformed input."""
    return _SwiftLexer(source).run()
