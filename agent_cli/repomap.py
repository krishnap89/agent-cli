"""Repo map: a compact, ranked outline of a codebase for small-context models.

How it works
1. List source files (git ls-files, so .gitignore is respected).
2. For each file, extract definitions (classes, functions, methods + line
   numbers + signatures) and the set of identifiers the file uses.
   Python uses the ast module; other languages use simple regex patterns.
   Results are cached in .agent-cache/ and only changed files are re-parsed.
3. Rank: a symbol matters if many OTHER files use it. A file's score is the
   sum of its symbols' scores. If a query is given, files and symbols whose
   names or paths match the query words are boosted heavily.
4. Render the top files and their top symbols until the character budget is
   used up, e.g.

       billing/tax.py
         12 class TaxCalculator
         20   def calculate(self, order, region) -> Decimal
         45 def load_rates(path)
"""
import ast
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CACHE_VERSION = 1
MAX_FILE_BYTES = 500_000
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", "out", "vendor", ".idea", ".vscode", ".agent-cache",
    ".next", ".nuxt", "coverage", ".tox", ".mypy_cache", ".pytest_cache", "bin", "obj",
}
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "how", "what", "where",
    "does", "work", "works", "code", "file", "files", "function", "class", "explain",
    "show", "find", "fix", "add", "use", "used", "make", "into", "why", "when", "are",
    "can", "you", "our", "all", "get", "set", "new", "not", "bug", "please",
}

# ---- Regex patterns for non-Python languages --------------------------------
# Each pattern's group(1) is the symbol name; the whole line is the signature.
_JS = [
    ("class", r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
    ("def", r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("),
    ("def", r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
    ("type", r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"),
    ("method", r"^\s+(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+)*([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?::\s*[^{=]+)?\{\s*$"),
]
_JAVA_LIKE = [
    ("class", r"^\s*(?:public|private|protected|internal|abstract|final|static|sealed|partial|data|open|\s)*(?:class|interface|enum|record|struct|object)\s+([A-Za-z_]\w*)"),
    ("method", r"^\s*(?:public|private|protected|internal|static|final|abstract|override|virtual|async|synchronized|suspend|fun|\s)+[\w<>\[\],.? ]*?\b([A-Za-z_]\w*)\s*\([^;]*$"),
]
LANG_PATTERNS = {
    ".js": _JS, ".jsx": _JS, ".ts": _JS, ".tsx": _JS, ".mjs": _JS, ".cjs": _JS, ".vue": _JS,
    ".java": _JAVA_LIKE, ".cs": _JAVA_LIKE, ".kt": _JAVA_LIKE, ".scala": _JAVA_LIKE,
    ".go": [
        ("def", r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\[(]"),
        ("type", r"^type\s+([A-Za-z_]\w*)\s+"),
    ],
    ".rs": [
        ("def", r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)"),
        ("type", r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|union)\s+([A-Za-z_]\w*)"),
        ("class", r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:]+\s+for\s+)?([A-Za-z_]\w*)"),
    ],
    ".rb": [
        ("class", r"^\s*(?:class|module)\s+([A-Z]\w*)"),
        ("def", r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!]?)"),
    ],
    ".php": [
        ("class", r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+([A-Za-z_]\w*)"),
        ("def", r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|abstract\s+|final\s+)*function\s+([A-Za-z_]\w*)"),
    ],
    ".c": [("def", r"^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;]*$"), ("type", r"^\s*(?:typedef\s+)?struct\s+([A-Za-z_]\w*)")],
    ".h": [("def", r"^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;]*\)\s*;"), ("type", r"^\s*(?:typedef\s+)?struct\s+([A-Za-z_]\w*)")],
    ".cpp": [("class", r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)"), ("def", r"^[A-Za-z_][\w\s\*&:<>,]*?\b([A-Za-z_][\w:]*)\s*\([^;]*$")],
    ".hpp": [("class", r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)")],
    ".swift": [
        ("class", r"^\s*(?:public\s+|private\s+|open\s+|final\s+)*(?:class|struct|enum|protocol|extension)\s+([A-Za-z_]\w*)"),
        ("def", r"^\s*(?:public\s+|private\s+|open\s+|static\s+|override\s+)*func\s+([A-Za-z_]\w*)"),
    ],
}
LANG_PATTERNS = {ext: [(k, re.compile(p)) for k, p in pats] for ext, pats in LANG_PATTERNS.items()}
NOT_METHODS = {"if", "for", "while", "switch", "catch", "return", "function", "else", "new", "sizeof", "elif"}


# ---- Symbol extraction --------------------------------------------------------

def _python_defs(source: str) -> List[list]:
    """[line, depth, kind, name, signature] for classes/functions, via ast."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    defs = []

    def visit(node, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in child.bases)
                sig = f"class {child.name}({bases})" if bases else f"class {child.name}"
                defs.append([child.lineno, depth, "class", child.name, sig])
                visit(child, depth + 1)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                sig = f"{prefix} {child.name}({ast.unparse(child.args)})"
                if child.returns is not None:
                    sig += f" -> {ast.unparse(child.returns)}"
                defs.append([child.lineno, depth, "def", child.name, sig])
                # don't descend into function bodies (nested helpers are noise)

    visit(tree, 0)
    return defs


def _regex_defs(source: str, ext: str) -> List[list]:
    patterns = LANG_PATTERNS.get(ext)
    if not patterns:
        return []
    defs = []
    for lineno, line in enumerate(source.splitlines(), 1):
        if len(line) > 300:
            continue
        for kind, rx in patterns:
            m = rx.match(line)
            if m and m.group(1) not in NOT_METHODS:
                indent = len(line) - len(line.lstrip())
                sig = line.strip().rstrip("{").strip()
                defs.append([lineno, min(indent // 2, 3), kind, m.group(1), sig[:150]])
                break
    return defs


def extract(path: Path) -> Optional[dict]:
    """Return {"defs": [...], "idents": [...]} for one file, or None to skip it."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        source = path.read_text(errors="replace")
    except OSError:
        return None
    if "\0" in source[:1000]:
        return None  # binary
    lines = source.splitlines()
    if lines and sum(len(l) for l in lines[:50]) / min(len(lines), 50) > 300:
        return None  # minified / generated
    ext = path.suffix.lower()
    defs = _python_defs(source) if ext == ".py" else _regex_defs(source, ext)
    idents = sorted(set(IDENT_RE.findall(source)))
    return {"defs": defs, "idents": idents}


# ---- The map ------------------------------------------------------------------

def _split_words(text: str) -> List[str]:
    """'calculateTaxRate get_user' -> ['calculate', 'tax', 'rate', 'user'] (+ originals)."""
    out = []
    for tok in re.findall(r"[A-Za-z0-9_]+", text):
        out.append(tok.lower())
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok).replace("_", " ").split()
        out += [p.lower() for p in parts]
    stems = []
    for w in out:  # crude singular forms: "interceptors" -> "interceptor"
        if w.endswith("ies") and len(w) > 4:
            stems.append(w[:-3] + "y")
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 4:
            stems.append(w[:-1])
    return [w for w in dict.fromkeys(out + stems) if len(w) >= 3 and w not in STOPWORDS]


class RepoMap:
    def __init__(self, root: Path, cache_dir: Optional[Path] = None):
        self.root = root.resolve()
        self.cache_path = (cache_dir or self.root / ".agent-cache") / "repomap.json"
        self.files: Dict[str, dict] = {}
        self._ref_files: Dict[str, int] = {}

    # -- indexing --

    def list_files(self) -> List[str]:
        exts = set(LANG_PATTERNS) | {".py"}
        try:
            out = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.root, capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return [f for f in out.stdout.splitlines()
                        if Path(f).suffix.lower() in exts
                        and not any(part in SKIP_DIRS for part in Path(f).parts)]
        except (OSError, subprocess.SubprocessError):
            pass
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if Path(name).suffix.lower() in exts:
                    found.append(os.path.relpath(os.path.join(dirpath, name), self.root))
        return found

    def build(self, progress=None) -> Tuple[int, int]:
        """Index the repo, reusing the cache for unchanged files. Returns (total, reparsed)."""
        cache = {}
        try:
            data = json.loads(self.cache_path.read_text())
            if data.get("version") == CACHE_VERSION:
                cache = data["files"]
        except (OSError, ValueError, KeyError):
            pass

        files, reparsed = {}, 0
        paths = self.list_files()
        for i, rel in enumerate(paths):
            full = self.root / rel
            try:
                st = full.stat()
            except OSError:
                continue
            key = [st.st_mtime, st.st_size]
            entry = cache.get(rel)
            if entry and entry.get("key") == key:
                files[rel] = entry
                continue
            info = extract(full)
            reparsed += 1
            if info is not None:
                info["key"] = key
                files[rel] = info
            if progress and reparsed % 200 == 0:
                progress(i + 1, len(paths))

        self.files = files
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            ignore = self.cache_path.parent / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n")  # keep the cache out of git status
            self.cache_path.write_text(json.dumps({"version": CACHE_VERSION, "files": files}))
        except OSError:
            pass

        # How many files mention each identifier.
        counts = Counter()
        for info in files.values():
            counts.update(info["idents"])
        self._ref_files = counts
        return len(files), reparsed

    # -- ranking --

    def _symbol_score(self, name: str, n_definers: int) -> float:
        # Files that use the name, minus the definer(s); log-damped so one
        # hugely popular helper doesn't dominate; split across duplicate names.
        refs = max(self._ref_files.get(name, 0) - n_definers, 0)
        if name.startswith("__") or name in ("main", "init", "setup", "test"):
            refs = min(refs, 1)
        return math.log1p(refs) / max(n_definers, 1)

    def render(self, query: str = "", budget: int = 4000, focus: str = "",
               exclude: Optional[set] = None) -> str:
        if not self.files:
            return "(repo map is empty: no supported source files found)"

        definers = defaultdict(int)
        for info in self.files.values():
            for d in info["defs"]:
                definers[d[3]] += 1

        words = _split_words(query)
        focus = focus.strip("/")
        file_scores, sym_scores = {}, {}
        for rel, info in self.files.items():
            if (focus and not rel.startswith(focus)) or (exclude and rel in exclude):
                continue
            rel_l = rel.lower()
            path_hit = sum(1 for w in words if w in rel_l)
            base = match = 0.0
            for d in info["defs"]:
                name = d[3]
                s = self._symbol_score(name, definers[name])
                base += s
                hits = sum(1 for w in words if w in name.lower())
                if hits:
                    s = s * 3 + 5 * hits
                    match += 5 * hits
                sym_scores[(rel, d[0])] = s
            if words:
                # With a query, matching files always outrank popular non-matching ones.
                match += 15 * path_hit
                total = match * 10 + base * 0.1 if match else base * 0.02
            else:
                total = base
            is_test = bool(re.search(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_|_test\.|\.spec\.|\.test\.", rel_l))
            if is_test and "test" not in words:
                total *= 0.2
            file_scores[rel] = total + 0.01 * min(len(info["defs"]), 20)

        ranked = [r for r in sorted(file_scores, key=lambda r: -file_scores[r]) if file_scores[r] > 0]
        out, used, shown = [], 0, 0
        for rel in ranked:
            defs = self.files[rel]["defs"]
            # Files matching the query get more detail; others a short summary,
            # so the map covers many files instead of one.
            matched = words and (any(w in rel.lower() for w in words) or any(
                sym_scores.get((rel, d[0]), 0) >= 5 for d in defs))
            if matched:
                n_hits = sum(1 for d in defs if any(w in d[3].lower() for w in words))
                limit = min(25, n_hits + 4)
            else:
                limit = 8
            best = sorted(defs, key=lambda d: -sym_scores.get((rel, d[0]), 0))[:limit]
            keep = {d[0] for d in best}
            block = [rel]
            for d in defs:
                # keep the best symbols, plus the classes that contain them
                if d[0] in keep or (d[2] == "class" and any(k > d[0] for k in keep)):
                    sig = d[4] if len(d[4]) <= 110 else d[4][:107] + "..."
                    block.append(f"{d[0]:>6} {'  ' * d[1]}{sig}")
            if len(defs) > len(keep):
                block.append(f"{'':>6} ... {len(defs) - len(keep)} more (use outline)")
            text = "\n".join(block)
            if used + len(text) > budget:
                if shown == 0:  # show at least part of the top file, cut at a line
                    out.append(text[:budget].rsplit("\n", 1)[0])
                    shown = 1
                if budget - used > 200:
                    continue  # a smaller file further down may still fit
                break
            out.append(text)
            used += len(text) + 1
            shown += 1

        remaining = len(ranked) - shown
        if remaining > 0:
            out.append(f"\n({remaining} more files not shown; call repo_map with a query or path to focus)")
        return "\n".join(out)

    def outline(self, rel: str) -> str:
        info = self.files.get(rel)
        if info is None:
            full = self.root / rel
            info = extract(full) if full.is_file() else None
        if not info or not info["defs"]:
            return f"No symbols found in {rel}."
        return "\n".join(f"{d[0]:>6} {'  ' * d[1]}{d[4]}" for d in info["defs"])
