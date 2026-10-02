"""Android project detection, module mapping, Gradle compile checks, and ktlint.

Provides Kotlin/Android-specific checks for Feature 1b:
- Detect Android projects (gradlew + settings.gradle + android plugin)
- Map changed files to Gradle modules
- Run compile checks via Gradle
- Run ktlint (optional, syntax-only by default)
- Parse Kotlin compiler error output
"""
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def detect_android(root: Path, mode: str = "auto") -> Optional[Dict]:
    """Detect an Android project.

    mode: "auto" (detect), "on" (force), "off" (disable).
    Returns {"modules": {":app": {"path": "app", "android": True}, ...}} or None.
    """
    if mode == "off":
        return None
    if mode == "on" or _is_android_project(root):
        modules = _discover_modules(root)
        return {"modules": modules} if modules else None
    return None


def _is_android_project(root: Path) -> bool:
    gradlew = root / "gradlew"
    if not gradlew.exists():
        return False
    settings = None
    for name in ("settings.gradle", "settings.gradle.kts"):
        p = root / name
        if p.is_file():
            settings = p
            break
    if settings is None:
        return False
    return _has_android_plugin(root)


def _has_android_plugin(root: Path) -> bool:
    """Check if any build.gradle(.kts) applies an Android plugin, including via version catalog aliases."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {
            ".git", ".gradle", "build", ".idea", "node_modules",
        }]
        for name in filenames:
            if name in ("build.gradle", "build.gradle.kts"):
                try:
                    text = Path(dirpath, name).read_text(errors="replace")
                except OSError:
                    continue
                if _text_has_android_plugin(text):
                    return True
    toml = root / "gradle" / "libs.versions.toml"
    if toml.is_file():
        try:
            text = toml.read_text(errors="replace")
        except OSError:
            return False
        if _toml_has_android_plugin(text):
            return True
    return False


def _text_has_android_plugin(text: str) -> bool:
    return bool(re.search(
        r"""(?:com\.android\.(?:application|library)|"""
        r"""id\s*\(\s*["']com\.android\.(?:application|library)["']\)|"""
        r"""alias\s*\(\s*libs\.plugins\.[^)]*android[^)]*\))""",
        text,
    ))


def _toml_has_android_plugin(text: str) -> bool:
    in_plugins = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_plugins = stripped.startswith("[plugins")
            continue
        if in_plugins:
            if re.search(r"""id\s*=\s*["']com\.android\.(?:application|library)["']""", stripped):
                return True
    return False


def _discover_modules(root: Path) -> Dict[str, Dict]:
    """Find all Gradle modules by locating build.gradle(.kts) files."""
    modules = {}  # type: Dict[str, Dict]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {
            ".git", ".gradle", "build", ".idea", "node_modules",
        }]
        for name in filenames:
            if name in ("build.gradle", "build.gradle.kts"):
                mod_path = Path(dirpath)
                rel = mod_path.relative_to(root).as_posix()
                if rel == ".":
                    mod_name = ":"
                else:
                    mod_name = ":" + rel.replace("/", ":")
                try:
                    text = Path(dirpath, name).read_text(errors="replace")
                except OSError:
                    text = ""
                is_android = _text_has_android_plugin(text)
                modules[mod_name] = {
                    "path": rel,
                    "android": is_android,
                    "build_file": name,
                }
    return modules


def map_file_to_module(rel_path: str, modules: Dict[str, Dict]) -> Optional[str]:
    """Find the nearest module for a file path. Returns the module name or None."""
    parts = Path(rel_path).parts
    best_name = None  # type: Optional[str]
    best_depth = -1
    for mod_name, mod_info in modules.items():
        mod_path = mod_info["path"]
        if mod_path == ".":
            if best_depth < 0:
                best_name = mod_name
                best_depth = 0
            continue
        mod_parts = Path(mod_path).parts
        if parts[:len(mod_parts)] == mod_parts and len(mod_parts) > best_depth:
            best_name = mod_name
            best_depth = len(mod_parts)
    return best_name


def is_build_config_file(rel_path: str) -> bool:
    """Check if a file is a build configuration file."""
    name = Path(rel_path).name
    return name in (
        "build.gradle", "build.gradle.kts",
        "settings.gradle", "settings.gradle.kts",
        "libs.versions.toml", "gradle.properties",
    )


def compile_tasks(
    changed_files: List[str],
    modules: Dict[str, Dict],
    variant: str = "Debug",
) -> Tuple[List[str], bool]:
    """Determine Gradle tasks for the changed files.

    Returns (tasks, is_build_config_change).
    """
    build_config = any(is_build_config_file(f) for f in changed_files)
    if build_config:
        return ["help"], True

    affected = {}  # type: Dict[str, Set[str]]
    for rel in changed_files:
        mod = map_file_to_module(rel, modules)
        if mod is None:
            continue
        ext = Path(rel).suffix.lower()
        affected.setdefault(mod, set()).add(ext)

    tasks = []  # type: List[str]
    for mod_name, exts in sorted(affected.items()):
        mod_info = modules.get(mod_name, {})
        is_android = mod_info.get("android", False)
        prefix = mod_name if mod_name != ":" else ""

        has_code = bool(exts & {".kt", ".kts", ".java"})
        has_xml_only = exts <= {".xml"}

        if has_code:
            if is_android:
                tasks.append(f"{prefix}:compile{variant}Kotlin")
            else:
                tasks.append(f"{prefix}:compileKotlin")
        elif has_xml_only and is_android:
            tasks.append(f"{prefix}:process{variant}Resources")

    return tasks, False


def run_gradle(
    root: Path,
    tasks: List[str],
    gradle_args: str = "--offline --console=plain -q",
    timeout: int = 600,
) -> Tuple[bool, str]:
    """Run ./gradlew with the given tasks. Returns (passed, output)."""
    gradlew = root / "gradlew"
    cmd_parts = [str(gradlew)] + tasks + gradle_args.split()
    try:
        r = subprocess.run(
            cmd_parts, cwd=root,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Gradle timed out after {timeout}s"
    output = (r.stdout + r.stderr).strip()
    return r.returncode == 0, output


def is_offline_dependency_error(output: str) -> bool:
    """Check if the error is about offline mode + missing dependencies."""
    lower = output.lower()
    has_offline = "offline" in lower
    has_resolve = "no cached version" in lower or "could not resolve" in lower
    return has_offline and has_resolve


_E_LINE_NEW = re.compile(
    r"^e:\s+file:///(.+?):(\d+):(\d+)\s+(.+)$"
)
_E_LINE_OLD = re.compile(
    r"^e:\s+(/\S+?):\s+\((\d+),\s*(\d+)\):\s+(.+)$"
)
_W_LINE = re.compile(r"^w:\s+")


def parse_kotlin_errors(
    output: str,
    root: Path,
    changed_files: Set[str],
    max_errors: int = 20,
) -> Tuple[List[Dict], bool]:
    """Parse Kotlin compiler errors from Gradle output.

    Returns (errors, all_pre_existing).
    Each error is {"file": rel, "line": int, "col": int, "message": str}.
    all_pre_existing is True if every error is in a file not in changed_files.
    """
    errors = []  # type: List[Dict]
    root_str = str(root.resolve())
    # On macOS, /var/folders is symlinked via /private; try both
    root_strs = [root_str]
    try:
        real = str(root.resolve().absolute())
        if real != root_str:
            root_strs.append(real)
        # Also try the non-resolved path
        plain = str(root)
        if plain not in root_strs:
            root_strs.append(plain)
    except OSError:
        pass

    for line in output.splitlines():
        if _W_LINE.match(line):
            continue
        m = _E_LINE_NEW.match(line) or _E_LINE_OLD.match(line)
        if not m:
            continue
        abs_path = m.group(1)
        lineno = int(m.group(2))
        col = int(m.group(3))
        message = m.group(4).strip()

        rel = abs_path
        for rs in root_strs:
            if abs_path.startswith(rs + "/") or abs_path.startswith(rs + os.sep):
                rel = abs_path[len(rs):].lstrip("/").lstrip(os.sep)
                break

        dup = any(
            e["file"] == rel and e["line"] == lineno and e["message"] == message
            for e in errors
        )
        if not dup:
            errors.append({
                "file": rel,
                "line": lineno,
                "col": col,
                "message": message,
            })

    if not errors:
        return errors, False

    all_pre_existing = all(e["file"] not in changed_files for e in errors)

    if len(errors) > max_errors:
        omitted = len(errors) - max_errors
        errors = errors[:max_errors]
        errors.append({
            "file": "",
            "line": 0,
            "col": 0,
            "message": f"... {omitted} more errors omitted",
        })

    return errors, all_pre_existing


def format_kotlin_errors(
    errors: List[Dict],
    root: Path,
) -> str:
    """Format parsed Kotlin errors with code context for the model."""
    parts = []  # type: List[str]
    for e in errors:
        if not e["file"]:
            parts.append(e["message"])
            continue
        header = f"{e['file']}:{e['line']}: {e['message']}"
        context = _get_context_lines(root / e["file"], e["line"])
        if context:
            parts.append(f"{header}\n{context}")
        else:
            parts.append(header)
    return "\n\n".join(parts)


def _get_context_lines(path: Path, lineno: int, context: int = 2) -> str:
    """Get a few lines around the error with line numbers."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    start = max(lineno - context - 1, 0)
    end = min(lineno + context, len(lines))
    result = []  # type: List[str]
    for i in range(start, end):
        marker = " >> " if i + 1 == lineno else "    "
        result.append(f"{marker}{i + 1:>4} {lines[i]}")
    return "\n".join(result)


def run_ktlint(
    root: Path,
    files: List[str],
    mode: str = "syntax",
) -> List[str]:
    """Run ktlint on the given files.

    mode: "syntax" = only parse failures, "full" = all violations, "off" = skip.
    Returns list of error strings.
    """
    if mode == "off":
        return []
    if not shutil.which("ktlint"):
        return []

    kt_files = [f for f in files if Path(f).suffix.lower() in (".kt", ".kts")]
    if not kt_files:
        return []

    cmd = ["ktlint", "--relative"] + kt_files
    try:
        r = subprocess.run(
            cmd, cwd=root,
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if r.returncode == 0:
        return []

    output = (r.stdout + r.stderr).strip()
    if not output:
        return []

    if mode == "syntax":
        errors = []  # type: List[str]
        for line in output.splitlines():
            if _is_ktlint_parse_error(line):
                errors.append(line)
        return errors

    return output.splitlines()


def _is_ktlint_parse_error(line: str) -> bool:
    """Check if a ktlint output line is a parse/syntax failure (not a style violation)."""
    lower = line.lower()
    return any(kw in lower for kw in (
        "syntax error",
        "parsing error",
        "unexpected",
        "expecting",
        "cannot be parsed",
        "not a valid kotlin file",
    ))


def module_display_name(mod_name: str) -> str:
    """Format module name for display. ':' -> '(root)', ':app' -> ':app'."""
    return "(root)" if mod_name == ":" else mod_name


def format_detected_modules(modules: Dict[str, Dict]) -> str:
    """Format the detected modules for the startup message."""
    names = sorted(modules.keys())
    display = [module_display_name(n) for n in names if n != ":"]
    if not display:
        display = ["(root)"]
    return ", ".join(display)
