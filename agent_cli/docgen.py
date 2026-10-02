"""KDoc generation for Kotlin declarations.

Implements /doc: scan Kotlin files for undocumented declarations, detect the
project's existing KDoc style, generate KDoc via the model with SEARCH/REPLACE
blocks, and verify that only comments changed (no code modifications).
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Kotlin declaration scanner
# ---------------------------------------------------------------------------

# Matches a KDoc block: /** ... */
_KDOC_RE = re.compile(r'/\*\*.*?\*/', re.DOTALL)

# Visibility keywords
_VISIBILITY = {"public", "internal", "protected", "private"}

# Modifiers that can appear before a declaration
_MODIFIERS = {
    "abstract", "open", "final", "sealed", "data", "inner", "value",
    "inline", "external", "actual", "expect", "suspend", "tailrec",
    "operator", "infix", "override", "lateinit", "const", "companion",
    "annotation", "enum", "fun",  # "fun interface"
}

# Annotation pattern
_ANNOTATION_RE = re.compile(r'^\s*@\w+')

# Declaration patterns
_CLASS_RE = re.compile(
    r'^\s*(?:(?:' + '|'.join(_VISIBILITY | _MODIFIERS) + r')\s+)*'
    r'(?:class|interface|object|enum\s+class)\s+([A-Za-z_]\w*)',
)
_FUN_RE = re.compile(
    r'^\s*(?:(?:' + '|'.join(_VISIBILITY | _MODIFIERS) + r')\s+)*'
    r'fun\s+(?:<[^>]+>\s*)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*[\(<]',
)
_PROP_RE = re.compile(
    r'^\s*(?:(?:' + '|'.join(_VISIBILITY | _MODIFIERS) + r')\s+)*'
    r'(?:val|var)\s+([A-Za-z_]\w*)',
)


class Declaration:
    """A Kotlin declaration found by the scanner."""
    __slots__ = (
        "kind", "name", "start_line", "decl_line", "end_line",
        "visibility", "has_kdoc", "is_override", "signature",
        "annotations",
    )

    def __init__(
        self,
        kind: str,           # "class", "interface", "object", "enum", "fun", "val", "var"
        name: str,
        start_line: int,     # first line (annotation or KDoc above)
        decl_line: int,      # the actual declaration keyword line
        end_line: int,       # last line of the body
        visibility: str,     # "public", "internal", "protected", "private"
        has_kdoc: bool,
        is_override: bool,
        signature: str,      # the declaration line(s) up to the body
        annotations: List[str],
    ):
        self.kind = kind
        self.name = name
        self.start_line = start_line
        self.decl_line = decl_line
        self.end_line = end_line
        self.visibility = visibility
        self.has_kdoc = has_kdoc
        self.is_override = is_override
        self.signature = signature
        self.annotations = annotations


def scan_declarations(source: str) -> List[Declaration]:
    """Find all declarations in a Kotlin source file.

    Handles classes, interfaces, objects, enums, functions (including
    extension functions, generic functions, multi-line signatures,
    expression bodies), and properties. Ignores content inside strings
    and comments.
    """
    lines = source.splitlines()
    cleaned = _strip_strings_and_comments(source)
    cleaned_lines = cleaned.splitlines()

    # Pad to same length
    while len(cleaned_lines) < len(lines):
        cleaned_lines.append("")

    decls = []  # type: List[Declaration]
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]

        # Try class/interface/object/enum
        m = _CLASS_RE.match(line)
        if m:
            kind = _class_kind(line)
            name = m.group(1)
            decl = _build_declaration(
                lines, cleaned_lines, i, kind, name,
            )
            if decl is not None:
                decls.append(decl)
            i += 1
            continue

        # Try function
        m = _FUN_RE.match(line)
        if m:
            name = m.group(1)
            decl = _build_declaration(
                lines, cleaned_lines, i, "fun", name,
            )
            if decl is not None:
                decls.append(decl)
            i += 1
            continue

        # Try property
        m = _PROP_RE.match(line)
        if m:
            name = m.group(1)
            kind = "val" if "val " in line else "var"
            decl = _build_declaration(
                lines, cleaned_lines, i, kind, name,
            )
            if decl is not None:
                decls.append(decl)
            i += 1
            continue

        i += 1

    return decls


def _class_kind(line: str) -> str:
    if re.search(r'\benum\s+class\b', line):
        return "enum"
    if re.search(r'\binterface\b', line):
        return "interface"
    if re.search(r'\bobject\b', line):
        return "object"
    return "class"


def _build_declaration(
    lines: List[str],
    cleaned_lines: List[str],
    decl_idx: int,
    kind: str,
    name: str,
) -> Optional[Declaration]:
    """Build a Declaration from context around the declaration line."""
    decl_line_num = decl_idx + 1  # 1-indexed
    line = cleaned_lines[decl_idx]

    # Determine visibility
    visibility = "public"  # default in Kotlin
    for v in _VISIBILITY:
        if re.search(r'\b' + v + r'\b', line):
            visibility = v
            break

    is_override = "override " in line or "override\t" in line

    # Look backwards for annotations and KDoc
    annotations = []  # type: List[str]
    start_idx = decl_idx
    j = decl_idx - 1
    while j >= 0:
        prev = lines[j].strip()
        if not prev:
            j -= 1
            continue
        if _ANNOTATION_RE.match(prev):
            annotations.insert(0, prev)
            start_idx = j
            j -= 1
            continue
        break

    # Check for KDoc above (above annotations if any)
    has_kdoc = False
    check_from = start_idx - 1
    while check_from >= 0 and not lines[check_from].strip():
        check_from -= 1
    if check_from >= 0:
        # Look for */ ending
        if lines[check_from].strip().endswith("*/"):
            # Find the matching /**
            k = check_from
            while k >= 0:
                if "/**" in lines[k]:
                    has_kdoc = True
                    start_idx = k
                    break
                k -= 1

    start_line = start_idx + 1  # 1-indexed

    # Build signature (may span multiple lines for multi-line params)
    sig_lines = [lines[decl_idx]]
    if kind == "fun":
        # Check if params span multiple lines
        open_parens = line.count("(") - line.count(")")
        si = decl_idx + 1
        while open_parens > 0 and si < len(cleaned_lines):
            sig_lines.append(lines[si])
            open_parens += cleaned_lines[si].count("(") - cleaned_lines[si].count(")")
            si += 1
        # Include return type line if next line has ":"
        if si < len(cleaned_lines) and cleaned_lines[si].strip().startswith(":"):
            sig_lines.append(lines[si])

    signature = "\n".join(sig_lines)

    # Find end of body
    end_idx = _find_body_end(cleaned_lines, decl_idx)
    end_line = end_idx + 1  # 1-indexed

    return Declaration(
        kind=kind,
        name=name,
        start_line=start_line,
        decl_line=decl_line_num,
        end_line=end_line,
        visibility=visibility,
        has_kdoc=has_kdoc,
        is_override=is_override,
        signature=signature,
        annotations=annotations,
    )


def _find_body_end(cleaned_lines: List[str], start: int) -> int:
    """Find the last line of a declaration body (matching braces)."""
    brace_count = 0
    found_open = False
    for i in range(start, len(cleaned_lines)):
        line = cleaned_lines[i]
        brace_count += line.count("{") - line.count("}")
        if "{" in line:
            found_open = True
        if found_open and brace_count <= 0:
            return i
        # Expression body (= ...) without braces: single line or until next decl
        if not found_open and i == start:
            if "=" in line and "{" not in line:
                return i
    return max(start, len(cleaned_lines) - 1)


def _strip_strings_and_comments(source: str) -> str:
    """Replace string contents and comment bodies with spaces, preserving line structure.

    This ensures the scanner doesn't match declarations inside strings or comments.
    Handles: "...", raw strings \"\"\"...\"\"\", char literals '...', // line comments,
    /* block comments */.
    """
    result = []  # type: List[str]
    i = 0
    n = len(source)
    while i < n:
        c = source[i]

        # Raw string """..."""
        if source[i:i+3] == '"""':
            result.append('"""')
            i += 3
            while i < n and source[i:i+3] != '"""':
                if source[i] == '\n':
                    result.append('\n')
                else:
                    result.append(' ')
                i += 1
            if i < n:
                result.append('"""')
                i += 3
            continue

        # Regular string "..."
        if c == '"':
            result.append('"')
            i += 1
            while i < n and source[i] != '"' and source[i] != '\n':
                if source[i] == '\\' and i + 1 < n:
                    result.append('  ')
                    i += 2
                else:
                    result.append(' ')
                    i += 1
            if i < n and source[i] == '"':
                result.append('"')
                i += 1
            continue

        # Char literal '...'
        if c == "'":
            result.append("'")
            i += 1
            while i < n and source[i] != "'" and source[i] != '\n':
                if source[i] == '\\' and i + 1 < n:
                    result.append('  ')
                    i += 2
                else:
                    result.append(' ')
                    i += 1
            if i < n and source[i] == "'":
                result.append("'")
                i += 1
            continue

        # Line comment //
        if source[i:i+2] == '//' and (i < 2 or source[i-1:i+1] != '*/'):
            while i < n and source[i] != '\n':
                result.append(' ')
                i += 1
            continue

        # Block comment /* ... */ (but preserve KDoc /** ... */ structure for detection)
        if source[i:i+2] == '/*':
            is_kdoc = source[i:i+3] == '/**'
            if is_kdoc:
                # Keep KDoc markers but blank content
                result.append('/**')
                i += 3
                while i < n and source[i:i+2] != '*/':
                    if source[i] == '\n':
                        result.append('\n')
                    else:
                        result.append(' ')
                    i += 1
                if i < n:
                    result.append('*/')
                    i += 2
            else:
                while i < n and source[i:i+2] != '*/':
                    if source[i] == '\n':
                        result.append('\n')
                    else:
                        result.append(' ')
                    i += 1
                if i < n:
                    result.append('  ')  # */
                    i += 2
            continue

        result.append(c)
        i += 1

    return ''.join(result)


# ---------------------------------------------------------------------------
# Filter declarations for documentation
# ---------------------------------------------------------------------------

def filter_declarations(
    decls: List[Declaration],
    rel_path: str,
    visibility_filter: str = "internal",  # "public" or "internal"
    include_properties: bool = False,
    update_mode: bool = False,
) -> List[Declaration]:
    """Filter declarations to those that should be documented.

    Skips: private, overrides, already-documented (unless update_mode),
    generated code, test files.
    """
    # Skip test files
    if "/src/test/" in rel_path or "/src/androidTest/" in rel_path:
        return []
    # Skip generated code
    if "/build/" in rel_path or "Generated" in rel_path:
        return []

    allowed_vis = {"public", "internal"} if visibility_filter == "internal" else {"public"}

    result = []  # type: List[Declaration]
    for d in decls:
        if d.visibility not in allowed_vis:
            continue
        if d.is_override:
            continue
        if d.kind in ("val", "var") and not include_properties:
            continue
        if d.has_kdoc and not update_mode:
            continue
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def detect_kdoc_style(
    root: Path,
    module_path: str = "",
    max_samples: int = 20,
) -> Dict[str, Any]:
    """Sample existing KDoc blocks and derive style rules.

    Returns {"uses_tags": bool, "summary_style": "third-person"|"imperative",
             "max_width": int, "examples": List[str]}.
    """
    samples = _collect_kdoc_samples(root, module_path, max_samples)

    if not samples:
        return {
            "uses_tags": True,
            "summary_style": "third-person",
            "max_width": 100,
            "examples": [],
        }

    uses_tags = any(
        re.search(r'@(?:param|return|throws|property)\b', s)
        for s in samples
    )
    tag_count = sum(1 for s in samples if re.search(r'@(?:param|return|throws|property)\b', s))
    uses_tags = tag_count > len(samples) * 0.3

    # Summary style: check first content line after /**
    imperative_count = 0
    third_person_count = 0
    for s in samples:
        first_line = _kdoc_first_line(s)
        if first_line:
            # Third person typically starts with a verb ending in 's'
            first_word = first_line.split()[0] if first_line.split() else ""
            if first_word.endswith("s") and first_word[0].isupper():
                third_person_count += 1
            elif first_word[0].isupper():
                imperative_count += 1

    summary_style = "third-person" if third_person_count >= imperative_count else "imperative"

    # Max line width
    widths = []  # type: List[int]
    for s in samples:
        for line in s.splitlines():
            widths.append(len(line))
    max_width = max(widths) if widths else 100
    max_width = min(max(max_width, 80), 120)

    # Pick 2-3 representative examples
    examples = samples[:3]

    return {
        "uses_tags": uses_tags,
        "summary_style": summary_style,
        "max_width": max_width,
        "examples": examples,
    }


def _collect_kdoc_samples(root: Path, module_path: str, max_samples: int) -> List[str]:
    """Collect existing KDoc blocks from the module (or project)."""
    search_dirs = []  # type: List[Path]
    if module_path:
        src = root / module_path / "src" / "main"
        if src.is_dir():
            search_dirs.append(src)
    if not search_dirs:
        search_dirs.append(root)

    samples = []  # type: List[str]
    for search_dir in search_dirs:
        for dirpath, dirnames, filenames in os.walk(search_dir):
            dirnames[:] = [d for d in dirnames if d not in {
                ".git", ".gradle", "build", ".idea", "node_modules", "test", "androidTest",
            }]
            for name in filenames:
                if not name.endswith(".kt") and not name.endswith(".kts"):
                    continue
                try:
                    text = Path(dirpath, name).read_text(errors="replace")
                except OSError:
                    continue
                for m in _KDOC_RE.finditer(text):
                    block = m.group(0)
                    if len(block) > 30:  # skip trivial ones
                        samples.append(block)
                        if len(samples) >= max_samples:
                            return samples
    return samples


def _kdoc_first_line(kdoc: str) -> str:
    """Extract the first content line from a KDoc block."""
    for line in kdoc.splitlines():
        stripped = line.strip().lstrip("/*").strip()
        if stripped and not stripped.startswith("@"):
            return stripped
    return ""


def build_style_prompt(style: Dict[str, Any]) -> str:
    """Build the style section of the generation prompt."""
    parts = []  # type: List[str]
    if style["uses_tags"]:
        parts.append("Use @param for every parameter, @return for non-Unit functions, "
                      "@throws only for exceptions this code throws.")
    else:
        parts.append("Use prose descriptions, not @param/@return tags.")

    if style["summary_style"] == "third-person":
        parts.append('Start the summary with a third-person verb (e.g. "Returns ...", "Creates ...").')
    else:
        parts.append('Start the summary with an imperative verb (e.g. "Return ...", "Create ...").')

    parts.append(f"Keep lines under {style['max_width']} characters.")

    if style["examples"]:
        parts.append("Follow this KDoc style:\n" + "\n\n".join(style["examples"][:2]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# --update mode: detect outdated KDoc
# ---------------------------------------------------------------------------

def find_outdated_kdoc(
    source: str,
    decls: List[Declaration],
    uses_tags: bool,
) -> List[Declaration]:
    """Find declarations whose existing KDoc is outdated.

    Checks: @param names not in signature, missing @param, @return on Unit
    functions or missing @return on non-Unit.
    """
    if not uses_tags:
        return []

    lines = source.splitlines()
    outdated = []  # type: List[Declaration]

    for d in decls:
        if not d.has_kdoc:
            continue
        if d.kind not in ("fun",):
            continue

        # Extract KDoc text
        kdoc_text = _extract_kdoc_above(lines, d.decl_line - 1)
        if not kdoc_text:
            continue

        # Extract @param names from KDoc
        kdoc_params = set(re.findall(r'@param\s+(\w+)', kdoc_text))

        # Extract parameter names from signature
        sig_params = set(re.findall(r'(\w+)\s*:', d.signature))
        # Remove common non-param words
        sig_params -= {"return", "override", "fun", "suspend", "private",
                       "public", "internal", "protected", "open", "abstract"}

        # Check for mismatches
        extra_params = kdoc_params - sig_params
        missing_params = sig_params - kdoc_params

        # Check @return
        has_return_tag = "@return" in kdoc_text
        returns_unit = _returns_unit(d.signature)

        is_outdated = False
        if extra_params:
            is_outdated = True
        if missing_params and uses_tags:
            is_outdated = True
        if has_return_tag and returns_unit:
            is_outdated = True
        if not has_return_tag and not returns_unit and uses_tags:
            is_outdated = True

        if is_outdated:
            outdated.append(d)

    return outdated


def _extract_kdoc_above(lines: List[str], decl_idx: int) -> str:
    """Extract the KDoc block above a declaration line (0-indexed)."""
    # Go backwards to find */
    j = decl_idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    # Skip annotations
    while j >= 0 and _ANNOTATION_RE.match(lines[j].strip()):
        j -= 1
    while j >= 0 and not lines[j].strip():
        j -= 1

    if j < 0 or not lines[j].strip().endswith("*/"):
        return ""

    end = j
    while j >= 0:
        if "/**" in lines[j]:
            return "\n".join(lines[j:end + 1])
        j -= 1
    return ""


def _returns_unit(signature: str) -> bool:
    """Check if a function signature returns Unit (no explicit return type, or : Unit)."""
    # Look for ": Type" after the closing paren
    after_paren = signature.rsplit(")", 1)
    if len(after_paren) < 2:
        return True
    rest = after_paren[1].strip().rstrip("{").strip()
    if not rest or rest.startswith("="):
        return True
    if rest.startswith(":"):
        ret_type = rest[1:].strip().split()[0] if rest[1:].strip() else ""
        return ret_type == "Unit"
    return True


# ---------------------------------------------------------------------------
# Code-change guardrail: comment stripper
# ---------------------------------------------------------------------------

def strip_comments_and_whitespace(source: str) -> str:
    """Strip all comments and whitespace from Kotlin source.

    Used to compare before/after and ensure only comments changed.
    Handles KDoc, block comments, line comments, strings (preserved).
    """
    result = []  # type: List[str]
    i = 0
    n = len(source)
    while i < n:
        c = source[i]

        # Raw string
        if source[i:i+3] == '"""':
            result.append('"""')
            i += 3
            while i < n and source[i:i+3] != '"""':
                result.append(source[i])
                i += 1
            if i < n:
                result.append('"""')
                i += 3
            continue

        # Regular string
        if c == '"':
            result.append('"')
            i += 1
            while i < n and source[i] != '"' and source[i] != '\n':
                if source[i] == '\\' and i + 1 < n:
                    result.append(source[i:i+2])
                    i += 2
                else:
                    result.append(source[i])
                    i += 1
            if i < n and source[i] == '"':
                result.append('"')
                i += 1
            continue

        # Char literal
        if c == "'":
            result.append("'")
            i += 1
            while i < n and source[i] != "'" and source[i] != '\n':
                if source[i] == '\\' and i + 1 < n:
                    result.append(source[i:i+2])
                    i += 2
                else:
                    result.append(source[i])
                    i += 1
            if i < n and source[i] == "'":
                result.append("'")
                i += 1
            continue

        # Line comment
        if source[i:i+2] == '//':
            while i < n and source[i] != '\n':
                i += 1
            continue

        # Block comment (including KDoc)
        if source[i:i+2] == '/*':
            while i < n and source[i:i+2] != '*/':
                i += 1
            if i < n:
                i += 2
            continue

        # Skip whitespace
        if c in (' ', '\t', '\n', '\r'):
            i += 1
            continue

        result.append(c)
        i += 1

    return ''.join(result)


def verify_code_unchanged(old_source: str, new_source: str) -> Optional[int]:
    """Compare old and new source with comments and whitespace removed.

    Returns None if code is identical, or the approximate line number
    where the first difference was found.
    """
    old_stripped = strip_comments_and_whitespace(old_source)
    new_stripped = strip_comments_and_whitespace(new_source)

    if old_stripped == new_stripped:
        return None

    # Find approximate line of first difference
    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()
    for i, (ol, nl) in enumerate(zip(old_lines, new_lines)):
        ol_s = strip_comments_and_whitespace(ol)
        nl_s = strip_comments_and_whitespace(nl)
        if ol_s != nl_s:
            return i + 1
    return len(old_lines) + 1


def verify_kdoc_balanced(source: str) -> Optional[int]:
    """Check that every /** has a matching */. Returns the line of the
    first unmatched /** or None if balanced."""
    lines = source.splitlines()
    in_kdoc = False
    start_line = 0
    for i, line in enumerate(lines):
        if "/**" in line and not in_kdoc:
            in_kdoc = True
            start_line = i + 1
        if "*/" in line and in_kdoc:
            in_kdoc = False
    if in_kdoc:
        return start_line
    return None


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

DOC_SYSTEM = """You add KDoc to Kotlin declarations. You use SEARCH/REPLACE blocks.

Rules:
- Add KDoc only. Do not change, reformat or reorder any code.
- Put each KDoc directly above the declaration and above its annotations.
- Describe what the code actually does, based on its body. Do not guess.
- Be concise: a one-sentence summary; add detail only if it is not obvious.
- Mention side effects, threading (suspend, main thread), and nullability where relevant.
- @throws only for exceptions this code throws. No TODOs, no "This function...".
- For @Composable functions, describe what is shown and what each callback does.
"""

DOC_PROMPT = """Add KDoc to these declarations in {file_path}.

{style_rules}

Declarations to document (line numbers for reference):
{decl_list}

The full file contents follow. Read the function bodies to describe them accurately.

{file_content}
"""

UPDATE_PROMPT = """Update the outdated KDoc blocks in {file_path}.
Keep the original wording where still correct. Only fix:
- @param names that don't match the current signature
- Missing @param tags for new parameters
- @return on Unit functions (remove it) or missing @return on non-Unit functions

{style_rules}

Declarations with outdated KDoc:
{decl_list}

{file_content}
"""


def build_doc_messages(
    file_path: str,
    source: str,
    decls: List[Declaration],
    style: Dict[str, Any],
    notes_text: str = "",
    update_mode: bool = False,
) -> List[dict]:
    """Build messages for one KDoc generation request."""
    style_rules = build_style_prompt(style)
    if notes_text:
        style_rules += "\n\nProject notes (follow these):\n" + notes_text

    decl_list = "\n".join(
        f"- line {d.decl_line}: {d.kind} {d.name}"
        + (f" ({', '.join(d.annotations)})" if d.annotations else "")
        for d in decls
    )

    file_content = f"```kotlin\n{source}\n```"

    if update_mode:
        prompt = UPDATE_PROMPT.format(
            file_path=file_path,
            style_rules=style_rules,
            decl_list=decl_list,
            file_content=file_content,
        )
    else:
        prompt = DOC_PROMPT.format(
            file_path=file_path,
            style_rules=style_rules,
            decl_list=decl_list,
            file_content=file_content,
        )

    return [
        {"role": "system", "content": DOC_SYSTEM},
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Resolve target (reuses testgen's pattern but simplified)
# ---------------------------------------------------------------------------

def resolve_doc_target(
    root: Path,
    target: str,
    repo_map_files: Dict[str, Any],
) -> List[str]:
    """Resolve a /doc target to a list of repo-relative .kt file paths.

    target can be: a class name, FQN, file path, or folder/module path.
    Raises ValueError on failure.
    """
    # File path
    if target.endswith(".kt") or target.endswith(".kts"):
        p = root / target
        if p.is_file():
            return [target]
        raise ValueError(f"File not found: {target}")

    # Check if it's a directory/module
    d = root / target
    if d.is_dir():
        kt_files = []  # type: List[str]
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [dn for dn in dirnames if dn not in {
                ".git", ".gradle", "build", ".idea", "node_modules",
            }]
            for name in filenames:
                if name.endswith(".kt") or name.endswith(".kts"):
                    p = Path(dirpath) / name
                    rel = p.relative_to(root).as_posix()
                    if "/src/test/" not in rel and "/src/androidTest/" not in rel:
                        if "/build/" not in rel and "Generated" not in name:
                            kt_files.append(rel)
        if kt_files:
            return sorted(kt_files)
        raise ValueError(f"No Kotlin files found in {target}.")

    # Class name lookup in repo map
    matches = []  # type: List[str]
    is_fqn = "." in target
    for rel, info in repo_map_files.items():
        if not rel.endswith(".kt"):
            continue
        if "/src/test/" in rel or "/src/androidTest/" in rel:
            continue
        for d_entry in info.get("defs", []):
            if d_entry[2] not in ("class", "type"):
                continue
            name = d_entry[3]
            if is_fqn:
                # Would need to check package — simplified: match class name part
                if target.endswith("." + name):
                    matches.append(rel)
            elif name == target:
                matches.append(rel)

    if matches:
        return list(dict.fromkeys(matches))  # deduplicate preserving order

    raise ValueError(f"Target not found: {target}. Use a file path, folder, or class name.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_doc_report(
    documented: int,
    files_count: int,
    skipped_existing: int,
    skipped_override: int,
    updated: int,
    code_verified: bool,
) -> str:
    """Format the final /doc report."""
    parts = [f"Documented {documented} declaration{'s' if documented != 1 else ''} "
             f"in {files_count} file{'s' if files_count != 1 else ''}"]

    skip_parts = []  # type: List[str]
    if skipped_existing:
        skip_parts.append(f"{skipped_existing} already documented")
    if skipped_override:
        skip_parts.append(f"{skipped_override} override{'s' if skipped_override != 1 else ''}")
    if skip_parts:
        parts.append(f" ({', '.join(skip_parts)} skipped)")

    if updated:
        parts.append(f"\nUpdated {updated} outdated KDoc block{'s' if updated != 1 else ''}")

    if code_verified:
        parts.append("\nCode unchanged: verified")

    parts.append("\n/undo removes all changes from /doc")
    return "".join(parts)
