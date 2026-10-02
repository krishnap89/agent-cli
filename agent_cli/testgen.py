"""Generate JVM unit tests for a Kotlin class.

Implements /test-gen: resolve target, detect test libraries, classify the class,
collect context, ask for a plan, write tests, compile, run, report.
Uses the Gradle module mapping and compile checks from android.py.
"""
import difflib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .android import (
    compile_tasks,
    format_kotlin_errors,
    is_build_config_file,
    map_file_to_module,
    parse_kotlin_errors,
    run_gradle,
)
from .config import Config


# ---------------------------------------------------------------------------
# Step 1: Resolve the target
# ---------------------------------------------------------------------------

def resolve_target(
    root: Path,
    target: str,
    method: str,
    repo_map_files: Dict[str, Any],
    modules: Dict[str, Dict],
) -> Tuple[str, str, Optional[str], str]:
    """Resolve a target to (rel_path, class_name, method_or_None, module_name).

    Raises ValueError on failure with a user-friendly message.
    """
    # File path?
    if target.endswith(".kt") or "/" in target:
        rel = target
        path = root / rel
        if not path.is_file():
            raise ValueError(f"File not found: {rel}")
        if "/src/test/" in rel or "/src/androidTest/" in rel:
            raise ValueError("That is already a test file.")
        if not _is_main_source(rel):
            raise ValueError(f"{rel} is not in a main source set (src/main/ or a flavor).")
        class_name = _class_name_from_file(path)
        mod = map_file_to_module(rel, modules)
        if mod is None:
            raise ValueError(f"Cannot determine the Gradle module for {rel}.")
        if method:
            _validate_method(path, method)
        return rel, class_name, method or None, mod

    # Class name (simple or fully qualified)
    matches = _find_class_in_map(target, repo_map_files, root)
    if not matches:
        close = _close_names(target, repo_map_files)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Class not found: {target}.{hint}")
    if len(matches) > 1:
        lines = [f"  {i+1}. {m[0]} (module {m[2]})" for i, m in enumerate(matches)]
        raise ValueError("Multiple matches found:\n" + "\n".join(lines) +
                         "\nSpecify the file path to disambiguate.")
    rel, class_name, mod = matches[0]
    if method:
        _validate_method(root / rel, method)
    return rel, class_name, method or None, mod


def _is_main_source(rel: str) -> bool:
    parts = rel.split("/")
    if "src" in parts:
        idx = parts.index("src")
        if idx + 1 < len(parts):
            src_set = parts[idx + 1]
            return src_set not in ("test", "androidTest")
    return True


def _class_name_from_file(path: Path) -> str:
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        m = re.match(r"^\s*(?:data\s+|sealed\s+|abstract\s+|open\s+|internal\s+|private\s+)*"
                     r"(?:class|object|interface)\s+([A-Za-z_]\w*)", line)
        if m:
            return m.group(1)
    return path.stem


def _validate_method(path: Path, method: str) -> None:
    text = path.read_text(errors="replace")
    pattern = re.compile(r"^\s*(?:.*\s)?fun\s+" + re.escape(method) + r"\s*[\(<]", re.MULTILINE)
    if not pattern.search(text):
        funs = re.findall(r"^\s*(?:.*\s)?fun\s+([A-Za-z_]\w*)", text, re.MULTILINE)
        hint = f" Available: {', '.join(funs[:10])}" if funs else ""
        raise ValueError(f"Method '{method}' not found in the class.{hint}")


def _find_class_in_map(
    target: str,
    repo_map_files: Dict[str, Any],
    root: Path,
) -> List[Tuple[str, str, str]]:
    """Find classes matching the target name. Returns [(rel_path, class_name, module)]."""
    is_fqn = "." in target
    results = []  # type: List[Tuple[str, str, str]]
    for rel, info in repo_map_files.items():
        if not rel.endswith(".kt"):
            continue
        if "/src/test/" in rel or "/src/androidTest/" in rel:
            continue
        for d in info.get("defs", []):
            if d[2] not in ("class", "type"):
                continue
            name = d[3]
            if is_fqn:
                pkg = _package_from_file(root / rel)
                fqn = f"{pkg}.{name}" if pkg else name
                if fqn == target:
                    results.append((rel, name, ""))
            else:
                if name == target:
                    results.append((rel, name, ""))
    return results


def _close_names(target: str, repo_map_files: Dict[str, Any]) -> List[str]:
    all_names = []  # type: List[str]
    for rel, info in repo_map_files.items():
        if not rel.endswith(".kt"):
            continue
        for d in info.get("defs", []):
            if d[2] in ("class", "type"):
                all_names.append(d[3])
    return difflib.get_close_matches(target, all_names, n=5, cutoff=0.5)


def _package_from_file(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        m = re.match(r"^\s*package\s+([\w.]+)", line)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Step 2: Detect the test setup
# ---------------------------------------------------------------------------

LIBRARY_MAP = {
    "junit:junit": "junit4",
    "org.junit.jupiter:junit-jupiter": "junit5",
    "org.junit.jupiter:junit-jupiter-api": "junit5",
    "org.junit.jupiter:junit-jupiter-engine": "junit5",
    "io.mockk:mockk": "mockk",
    "org.mockito.kotlin:mockito-kotlin": "mockito-kotlin",
    "org.mockito:mockito-core": "mockito",
    "org.jetbrains.kotlinx:kotlinx-coroutines-test": "coroutines-test",
    "app.cash.turbine:turbine": "turbine",
    "com.google.truth:truth": "truth",
    "com.willowtreeapps.assertk:assertk": "assertk",
    "com.willowtreeapps.assertk:assertk-jvm": "assertk",
    "org.jetbrains.kotlin:kotlin-test": "kotlin-test",
    "org.jetbrains.kotlin:kotlin-test-junit": "kotlin-test",
    "org.robolectric:robolectric": "robolectric",
    "androidx.arch.core:core-testing": "arch-testing",
}


def detect_test_setup(
    root: Path,
    module_name: str,
    modules: Dict[str, Dict],
) -> Dict[str, Any]:
    """Detect test libraries from build files and version catalog.

    Returns {"libs": set_of_capability_names, "has_junit5": bool,
             "dsl": "groovy"|"kotlin", "catalog": dict_or_None}.
    """
    mod_info = modules.get(module_name, {})
    mod_path = mod_info.get("path", ".")
    build_file = mod_info.get("build_file", "build.gradle.kts")
    dsl = "kotlin" if build_file.endswith(".kts") else "groovy"

    catalog = _parse_version_catalog(root)
    build_text = _read_file(root / mod_path / build_file)
    root_build = _read_build_root(root)

    libs = set()  # type: Set[str]

    # Direct dependencies in build file
    for coord, cap in LIBRARY_MAP.items():
        if coord in build_text or coord in root_build:
            libs.add(cap)

    # Version catalog aliases
    if catalog:
        _resolve_catalog_deps(build_text, catalog, libs)

    # kotlin("test") shorthand
    if 'kotlin("test")' in build_text or "kotlin-test" in build_text:
        libs.add("kotlin-test")

    # useJUnitPlatform()
    if "useJUnitPlatform()" in build_text or "useJUnitPlatform()" in root_build:
        libs.add("junit5")

    # Ensure at least one framework detected from standard patterns
    if "junit4" not in libs and "junit5" not in libs and "kotlin-test" not in libs:
        # Check for common patterns
        if "testImplementation" in build_text:
            if "junit" in build_text.lower():
                libs.add("junit4")

    return {
        "libs": libs,
        "has_junit5": "junit5" in libs,
        "dsl": dsl,
        "catalog": catalog,
    }


def _read_file(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _read_build_root(root: Path) -> str:
    for name in ("build.gradle.kts", "build.gradle"):
        text = _read_file(root / name)
        if text:
            return text
    return ""


def _parse_version_catalog(root: Path) -> Optional[Dict[str, str]]:
    """Parse gradle/libs.versions.toml [libraries] section into {alias: "group:artifact"}."""
    toml_path = root / "gradle" / "libs.versions.toml"
    text = _read_file(toml_path)
    if not text:
        return None

    catalog = {}  # type: Dict[str, str]
    in_libraries = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_libraries = stripped.startswith("[libraries")
            continue
        if not in_libraries:
            continue
        m = re.match(r'^([\w-]+)\s*=\s*(.+)$', stripped)
        if not m:
            continue
        alias = m.group(1).strip()
        value = m.group(2).strip()
        coord = _parse_catalog_library(value)
        if coord:
            catalog[alias] = coord
    return catalog if catalog else None


def _parse_catalog_library(value: str) -> Optional[str]:
    """Parse a TOML library value to 'group:artifact'."""
    # "group:artifact:version" form
    if value.startswith('"') and value.count(":") >= 2:
        parts = value.strip('"').split(":")
        return f"{parts[0]}:{parts[1]}"

    # { module = "group:artifact", ... } form
    m = re.search(r'module\s*=\s*"([^"]+)"', value)
    if m:
        parts = m.group(1).split(":")
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}"

    # { group = "g", name = "a", ... } form
    g = re.search(r'group\s*=\s*"([^"]+)"', value)
    n = re.search(r'name\s*=\s*"([^"]+)"', value)
    if g and n:
        return f"{g.group(1)}:{n.group(1)}"

    return None


def _resolve_catalog_deps(build_text: str, catalog: Dict[str, str], libs: Set[str]) -> None:
    """Resolve version catalog aliases used in testImplementation."""
    # Find libs.xxx references in testImplementation lines
    for line in build_text.splitlines():
        stripped = line.strip()
        if "testImplementation" not in stripped:
            continue
        # Match libs.xxx.yyy patterns
        for m in re.finditer(r'libs\.([\w.]+)', stripped):
            alias_dotted = m.group(1)
            alias_dashed = alias_dotted.replace(".", "-")
            coord = catalog.get(alias_dashed) or catalog.get(alias_dotted)
            if coord:
                for lib_coord, cap in LIBRARY_MAP.items():
                    if coord.startswith(lib_coord.rsplit(":", 1)[0] + ":") or coord == lib_coord:
                        libs.add(cap)
                        break
                # Also try prefix matching
                for lib_coord, cap in LIBRARY_MAP.items():
                    g, a = lib_coord.split(":", 1)
                    cg = coord.split(":")[0] if ":" in coord else ""
                    ca = coord.split(":")[1] if ":" in coord else ""
                    if g == cg and (a == ca or ca.startswith(a)):
                        libs.add(cap)


def check_required_libs(
    libs: Set[str],
    has_suspend: bool,
    has_deps: bool,
    dsl: str,
    catalog: Optional[Dict[str, str]],
) -> List[str]:
    """Check if required test libraries are present. Returns list of missing-dependency messages."""
    missing = []  # type: List[str]
    has_framework = bool(libs & {"junit4", "junit5", "kotlin-test"})
    if not has_framework:
        if dsl == "kotlin":
            missing.append('Missing test framework. Add: testImplementation("junit:junit:4.13.2")')
        else:
            missing.append("Missing test framework. Add: testImplementation 'junit:junit:4.13.2'")

    has_mock = bool(libs & {"mockk", "mockito-kotlin", "mockito"})
    if has_deps and not has_mock:
        if dsl == "kotlin":
            missing.append('Missing mocking library. Add: testImplementation("io.mockk:mockk:1.13.9")')
        else:
            missing.append("Missing mocking library. Add: testImplementation 'io.mockk:mockk:1.13.9'")

    if has_suspend and "coroutines-test" not in libs:
        if dsl == "kotlin":
            missing.append('Missing coroutines test. Add: testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")')
        else:
            missing.append("Missing coroutines test. Add: testImplementation 'org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3'")

    return missing


# ---------------------------------------------------------------------------
# Step 3: Classify the class
# ---------------------------------------------------------------------------

CLASS_KINDS = {
    "viewmodel": "ViewModel",
    "repository": "Repository",
    "usecase": "UseCase",
    "plain": "Plain",
    "dao": "DAO",
    "ui": "UI",
}


def classify_class(path: Path, class_name: str) -> str:
    """Classify a Kotlin class into a kind for prompt customisation."""
    text = path.read_text(errors="replace")

    # DAO
    if re.search(r"@Dao\b", text):
        return "dao"

    # UI components
    if re.search(r":\s*(Activity|Fragment|Service|BroadcastReceiver)\b", text):
        return "ui"
    if re.search(r"@Composable\b", text):
        return "ui"

    # ViewModel
    if re.search(r":\s*(ViewModel|AndroidViewModel)\b", text):
        return "viewmodel"

    # Repository / DataSource
    if class_name.endswith(("Repository", "DataSource")):
        return "repository"

    # UseCase / Interactor
    if class_name.endswith(("UseCase", "Interactor")):
        return "usecase"

    return "plain"


KIND_RULES = {
    "viewmodel": ("Set the main dispatcher with {dispatcher_setup}; "
                  "assert on exposed StateFlow/LiveData state; use advanceUntilIdle()."),
    "repository": ("Mock the data sources; test success, empty and error results; "
                   "use runTest for suspend functions."),
    "usecase": ("Mock the repository; test the main path, edge cases and error cases; "
                "use runTest for suspend functions."),
    "plain": "Test public functions with representative inputs and edge cases.",
}


# ---------------------------------------------------------------------------
# Step 2b: Find style examples and helpers
# ---------------------------------------------------------------------------

def find_test_examples(
    root: Path,
    module_path: str,
    class_name: str,
    kind: str,
    max_examples: int = 2,
    max_chars: int = 3000,
) -> List[Tuple[str, str]]:
    """Find existing test files to use as style examples.

    Returns [(rel_path, truncated_content)].
    """
    test_dir = root / module_path / "src" / "test"
    if not test_dir.is_dir():
        # Fall back to other modules
        for d in root.iterdir():
            td = d / "src" / "test"
            if td.is_dir():
                test_dir = td
                break
        else:
            return []

    test_files = []  # type: List[Tuple[str, int, Path]]
    for dirpath, _, filenames in os.walk(test_dir):
        for name in filenames:
            if not name.endswith("Test.kt") and not name.endswith("Tests.kt"):
                continue
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError:
                continue
            # Score: prefer matching kind, then same package, then shorter
            score = 0
            if kind == "viewmodel" and "ViewModel" in name:
                score += 100
            elif kind == "repository" and ("Repository" in name or "Repo" in name):
                score += 100
            elif kind == "usecase" and ("UseCase" in name or "Interactor" in name):
                score += 100
            score -= min(size, 10000) // 100  # prefer shorter
            test_files.append((name, score, p))

    test_files.sort(key=lambda x: -x[1])
    results = []  # type: List[Tuple[str, str]]
    for name, _, p in test_files[:max_examples]:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)..."
        results.append((rel, text))
    return results


def find_test_helpers(root: Path, module_path: str) -> List[Dict[str, str]]:
    """Find test helper classes: main-dispatcher rules, fakes, fixtures."""
    helpers = []  # type: List[Dict[str, str]]
    test_dir = root / module_path / "src" / "test"
    if not test_dir.is_dir():
        return helpers

    patterns = [
        re.compile(r"(?:Fake|Mock)\w+"),
        re.compile(r"\w+Fixtures?"),
        re.compile(r"\w+TestData"),
        re.compile(r"\w+Rule"),
        re.compile(r"\w+Extension"),
    ]
    dispatcher_pattern = re.compile(r"Dispatchers\.setMain|MainDispatcherRule|MainCoroutineRule", re.MULTILINE)

    for dirpath, _, filenames in os.walk(test_dir):
        for name in filenames:
            if not name.endswith(".kt"):
                continue
            p = Path(dirpath) / name
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            pkg = _package_from_file(p)

            # Check for dispatcher rule
            if dispatcher_pattern.search(text):
                class_name_match = re.search(r"class\s+(\w+)", text)
                if class_name_match:
                    helpers.append({
                        "name": class_name_match.group(1),
                        "package": pkg,
                        "path": rel,
                        "kind": "dispatcher-rule",
                        "outline": _outline(text),
                    })

            # Check for fakes/fixtures
            for pat in patterns:
                m = re.search(r"class\s+(" + pat.pattern + r")", text)
                if m:
                    helpers.append({
                        "name": m.group(1),
                        "package": pkg,
                        "path": rel,
                        "kind": "helper",
                        "outline": _outline(text),
                    })
                    break

    return helpers


def _outline(text: str, max_lines: int = 20) -> str:
    """Extract class/fun signatures from Kotlin source."""
    lines = []  # type: List[str]
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"(class|object|interface|fun|val|var|override\s+fun)\s", stripped):
            lines.append(stripped[:120])
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3b: Analyse dependencies
# ---------------------------------------------------------------------------

def has_suspend_funs(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return bool(re.search(r"\bsuspend\s+fun\b", text))


def has_flow_or_viewmodelscope(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return bool(re.search(r"\b(viewModelScope|Flow<|StateFlow<|SharedFlow<|CoroutineDispatcher)\b", text))


def has_constructor_deps(path: Path) -> bool:
    """Check if the primary constructor has parameters that look like dependencies."""
    text = path.read_text(errors="replace")
    m = re.search(r"class\s+\w+\s*\(([^)]+)\)", text)
    if not m:
        return False
    params = m.group(1)
    # Has parameters that are likely interfaces/classes (capitalized types)
    return bool(re.search(r":\s*[A-Z]\w+", params))


def extract_constructor_types(path: Path) -> List[str]:
    """Extract type names from the primary constructor."""
    text = path.read_text(errors="replace")
    m = re.search(r"class\s+\w+\s*\(([^)]*)\)", text, re.DOTALL)
    if not m:
        return []
    params = m.group(1)
    types = re.findall(r":\s*([A-Z]\w+)", params)
    return list(dict.fromkeys(types))


# ---------------------------------------------------------------------------
# Step 4: Collect context
# ---------------------------------------------------------------------------

def collect_context(
    root: Path,
    target_path: str,
    class_name: str,
    method: Optional[str],
    kind: str,
    setup: Dict[str, Any],
    examples: List[Tuple[str, str]],
    helpers: List[Dict[str, str]],
    notes_text: str,
    existing_test_path: Optional[str],
    repo_map_files: Dict[str, Any],
    budget: int,
) -> str:
    """Build the context string for the model, within budget."""
    parts = []  # type: List[str]

    # Target class source
    target_text = (root / target_path).read_text(errors="replace")
    if method:
        target_text = _extract_method_context(target_text, class_name, method)
    if len(target_text) > int(budget * 0.4):
        target_text = target_text[:int(budget * 0.4)] + "\n...(truncated)..."
    parts.append(f"# Target class: {target_path}\n```kotlin\n{target_text}\n```")

    # Constructor dependency outlines
    dep_types = extract_constructor_types(root / target_path)
    for dtype in dep_types[:6]:
        outline_text = _find_type_outline(dtype, repo_map_files, root)
        if outline_text:
            parts.append(f"# Dependency: {dtype}\n```kotlin\n{outline_text}\n```")

    # Detected libraries
    libs_list = ", ".join(sorted(setup["libs"])) if setup["libs"] else "(none)"
    parts.append(f"# Detected test libraries: {libs_list}")

    # Kind-specific rules
    rule = KIND_RULES.get(kind, KIND_RULES["plain"])
    parts.append(f"# Class kind: {kind}\nRule: {rule}")

    # Style examples
    for ex_path, ex_text in examples:
        parts.append(f"# Style example: {ex_path}\n```kotlin\n{ex_text}\n```")

    # Helpers
    if helpers:
        helper_text = "\n".join(
            f"- {h['name']} ({h['package']}, {h['kind']}): {h['path']}\n  {h['outline'][:200]}"
            for h in helpers[:4]
        )
        parts.append(f"# Existing test helpers:\n{helper_text}")

    # Existing test file
    if existing_test_path:
        try:
            existing = (root / existing_test_path).read_text(errors="replace")
            parts.append(f"# Existing test file: {existing_test_path}\n```kotlin\n{existing}\n```")
        except OSError:
            pass

    # Notes
    if notes_text:
        parts.append(f"# Project notes:\n{notes_text}")

    result = "\n\n".join(parts)
    if len(result) > budget:
        result = result[:budget] + "\n...(truncated)..."
    return result


def _extract_method_context(text: str, class_name: str, method: str) -> str:
    """Extract class header + specific method from source."""
    lines = text.splitlines()
    # Find class declaration
    class_start = 0
    for i, line in enumerate(lines):
        if re.search(r"class\s+" + re.escape(class_name), line):
            class_start = i
            break

    # Find method
    method_start = method_end = None
    for i, line in enumerate(lines):
        if re.search(r"fun\s+" + re.escape(method) + r"\s*[\(<]", line):
            method_start = i
            break
    if method_start is not None:
        brace_count = 0
        for i in range(method_start, len(lines)):
            brace_count += lines[i].count("{") - lines[i].count("}")
            if brace_count <= 0 and i > method_start:
                method_end = i + 1
                break
        if method_end is None:
            method_end = min(method_start + 30, len(lines))

    # Build: class header (first ~10 lines) + method
    header = "\n".join(lines[class_start:min(class_start + 10, len(lines))])
    if method_start is not None:
        body = "\n".join(lines[method_start:method_end])
        return f"{header}\n\n    // ... (other members omitted)\n\n{body}\n}}"
    return text


def _find_type_outline(
    type_name: str,
    repo_map_files: Dict[str, Any],
    root: Path,
) -> Optional[str]:
    """Find a type in the repo map and return its outline or full source if small."""
    for rel, info in repo_map_files.items():
        for d in info.get("defs", []):
            if d[3] == type_name and d[2] in ("class", "type"):
                try:
                    text = (root / rel).read_text(errors="replace")
                except OSError:
                    continue
                # Small data classes / enums / sealed classes: include in full
                if len(text) < 1500 and re.search(
                    r"(data\s+class|enum\s+class|sealed\s+class|sealed\s+interface)", text
                ):
                    return text
                # Otherwise just signatures
                return _outline(text)
    return None


# ---------------------------------------------------------------------------
# Step 5-6: Build prompts
# ---------------------------------------------------------------------------

PLAN_PROMPT = """List the test cases for {class_name}{method_part}.
Group by method. One line per test case: scenario -> expected outcome.
Cover: main path, edge cases (empty, null, zero, boundaries), error paths.
Exclude private functions.{existing_note}
Only list the test cases, no code. Number them."""

WRITE_PROMPT = """You write JUnit unit tests for Kotlin Android code.
Use ONLY these libraries: {libs}. Do not import anything else.
Follow the style of the example tests exactly: structure, naming, mocking, assertions.
Reuse these existing helpers instead of writing your own: {helpers}.
Only edit {test_path}. Never change production code.
One behaviour per test. Implement exactly the approved test cases.
Use runTest for suspend functions.

Test cases to implement:
{plan}
"""


def build_plan_prompt(class_name: str, method: Optional[str], has_existing: bool) -> str:
    method_part = f".{method}" if method else ""
    existing_note = "\nOnly list cases NOT already covered." if has_existing else ""
    return PLAN_PROMPT.format(
        class_name=class_name,
        method_part=method_part,
        existing_note=existing_note,
    )


def build_write_prompt(
    libs: Set[str],
    helpers: List[Dict[str, str]],
    test_path: str,
    plan: str,
) -> str:
    libs_str = ", ".join(sorted(libs)) if libs else "(standard library only)"
    helpers_str = ", ".join(h["name"] for h in helpers) if helpers else "(none)"
    return WRITE_PROMPT.format(
        libs=libs_str,
        helpers=helpers_str,
        test_path=test_path,
        plan=plan,
    )


# ---------------------------------------------------------------------------
# Step 6b: Determine test file path
# ---------------------------------------------------------------------------

def test_file_path(
    root: Path,
    module_path: str,
    target_path: str,
    class_name: str,
) -> str:
    """Determine the test file path for a class."""
    # Determine package from target
    pkg = _package_from_file(root / target_path)
    pkg_path = pkg.replace(".", "/") if pkg else ""

    # Determine java/ or kotlin/ for test sources
    test_base = root / module_path / "src" / "test"
    if (test_base / "kotlin").is_dir():
        lang_dir = "kotlin"
    elif (test_base / "java").is_dir():
        lang_dir = "java"
    else:
        # Mirror main source set
        main_base = root / module_path / "src" / "main"
        if (main_base / "kotlin").is_dir():
            lang_dir = "kotlin"
        else:
            lang_dir = "java"

    parts = [module_path, "src", "test", lang_dir]
    if pkg_path:
        parts.append(pkg_path)
    parts.append(f"{class_name}Test.kt")
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Step 7: Compile check
# ---------------------------------------------------------------------------

def compile_test_task(
    module_name: str,
    modules: Dict[str, Dict],
    variant: str = "Debug",
) -> str:
    """Get the compile task for a test source set."""
    mod_info = modules.get(module_name, {})
    is_android = mod_info.get("android", False)
    prefix = module_name if module_name != ":" else ""
    if is_android:
        return f"{prefix}:compile{variant}UnitTestKotlin"
    else:
        return f"{prefix}:compileTestKotlin"


def run_test_task(
    module_name: str,
    modules: Dict[str, Dict],
    test_fqcn: str,
    variant: str = "Debug",
) -> str:
    """Get the Gradle task to run a specific test class."""
    mod_info = modules.get(module_name, {})
    is_android = mod_info.get("android", False)
    prefix = module_name if module_name != ":" else ""
    if is_android:
        return f"{prefix}:test{variant}UnitTest"
    else:
        return f"{prefix}:test"


# ---------------------------------------------------------------------------
# Step 8: Parse JUnit XML report
# ---------------------------------------------------------------------------

def parse_junit_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse a JUnit XML report. Returns [{name, passed, failure_message, failure_trace}]."""
    results = []  # type: List[Dict[str, Any]]
    try:
        tree = ET.parse(str(xml_path))
    except (ET.ParseError, OSError):
        return results

    for tc in tree.iter("testcase"):
        name = tc.get("name", "")
        failure = tc.find("failure")
        error = tc.find("error")
        elem = failure if failure is not None else error
        if elem is not None:
            msg = elem.get("message", "")
            trace = (elem.text or "")[:2000]
            results.append({"name": name, "passed": False,
                            "failure_message": msg, "failure_trace": trace})
        else:
            results.append({"name": name, "passed": True,
                            "failure_message": "", "failure_trace": ""})
    return results


def find_junit_xml(
    root: Path,
    module_path: str,
    variant: str,
    test_fqcn: str,
) -> Optional[Path]:
    """Find the JUnit XML report for a test class."""
    # Common paths for Gradle test results
    candidates = [
        root / module_path / "build" / "test-results" / f"test{variant}UnitTest" / f"TEST-{test_fqcn}.xml",
        root / module_path / "build" / "test-results" / "test" / f"TEST-{test_fqcn}.xml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# Step 9: Report
# ---------------------------------------------------------------------------

def format_report(
    test_path: str,
    total_tests: int,
    passed: int,
    ignored: int,
    suspected_bugs: List[str],
    fix_rounds: int,
) -> str:
    """Format the final report."""
    lines = [f"Created {test_path} ({total_tests} tests)"]
    compile_note = f" ({fix_rounds} fix round{'s' if fix_rounds != 1 else ''})" if fix_rounds else ""
    lines.append(f"Compile: passed{compile_note}")

    failed = total_tests - passed - ignored
    parts = []  # type: List[str]
    if passed:
        parts.append(f"{passed} passed")
    if ignored:
        parts.append(f"{ignored} ignored as suspected bug")
    if failed:
        parts.append(f"{failed} failed")
    lines.append(f"Tests:   {', '.join(parts)}")

    for bug in suspected_bugs:
        lines.append(f"Suspected bug: {bug}")

    lines.append("/undo removes the generated tests.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse CODE_BUG / TEST_BUG classifications from model output
# ---------------------------------------------------------------------------

_BUG_LINE_RE = re.compile(r"^(\w+)\s*:\s*(TEST_BUG|CODE_BUG)(?:\s*-\s*(.+))?$", re.MULTILINE)


def parse_bug_classifications(text: str) -> Dict[str, Tuple[str, str]]:
    """Parse TEST_BUG/CODE_BUG lines from model output.

    Returns {test_name: (kind, reason)}.
    """
    result = {}  # type: Dict[str, Tuple[str, str]]
    for m in _BUG_LINE_RE.finditer(text):
        name = m.group(1)
        kind = m.group(2)
        reason = m.group(3) or ""
        result[name] = (kind, reason.strip())
    return result
