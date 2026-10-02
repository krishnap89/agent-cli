"""Tests for Feature 1b: Kotlin/Android checks.

Uses fake gradlew scripts to test without a real Gradle installation.
"""
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import patch

from agent_cli.android import (
    compile_tasks,
    detect_android,
    format_detected_modules,
    format_kotlin_errors,
    is_build_config_file,
    is_offline_dependency_error,
    map_file_to_module,
    module_display_name,
    parse_kotlin_errors,
    run_gradle,
)
from agent_cli.checks import builtin_lint, ext_lint_cmd
from agent_cli.config import Config
from agent_cli.coder import Coder
from tests.fake_server import FakeServer


def _write_fake_gradlew(root, output="", exit_code=0, record_args=True):
    """Write a fake gradlew shell script that records its arguments and prints canned output."""
    # type: (Path, str, int, bool) -> Path
    gradlew = root / "gradlew"
    lines = ["#!/bin/sh"]
    if record_args:
        lines.append('echo "$@" >> "$(dirname "$0")/.gradlew_args"')
    if output:
        # Write the output to a file so we can include multi-line content
        out_file = root / ".gradlew_output"
        out_file.write_text(output)
        lines.append(f'cat "$(dirname "$0")/.gradlew_output"')
    lines.append(f"exit {exit_code}")
    gradlew.write_text("\n".join(lines) + "\n")
    gradlew.chmod(gradlew.stat().st_mode | stat.S_IEXEC)
    return gradlew


def _setup_android_project(root, groovy=True, modules=None):
    """Create a minimal Android project structure.

    modules: dict of {module_path: is_android}, e.g. {"app": True, "core/data": False}
    """
    # type: (Path, bool, Optional[Dict[str, bool]]) -> None
    ext = ".gradle" if groovy else ".gradle.kts"
    settings_name = "settings" + ext
    (root / settings_name).write_text("include ':app'\n")
    _write_fake_gradlew(root)

    if modules is None:
        modules = {"app": True}

    for mod_path, is_android in modules.items():
        mod_dir = root / mod_path
        mod_dir.mkdir(parents=True, exist_ok=True)
        build_file = mod_dir / ("build" + ext)
        if is_android:
            build_file.write_text(
                'plugins {\n    id "com.android.application"\n}\n'
                if groovy else
                'plugins {\n    id("com.android.application")\n}\n'
            )
        else:
            build_file.write_text(
                'plugins {\n    id "org.jetbrains.kotlin.jvm"\n}\n'
            )


def _setup_version_catalog_project(root):
    """Create an Android project using version catalog plugin aliases."""
    # type: (Path) -> None
    (root / "settings.gradle.kts").write_text("include(\":app\")\n")
    _write_fake_gradlew(root)
    app = root / "app"
    app.mkdir()
    (app / "build.gradle.kts").write_text(
        "plugins {\n    alias(libs.plugins.android.application)\n}\n"
    )
    gradle_dir = root / "gradle"
    gradle_dir.mkdir()
    (gradle_dir / "libs.versions.toml").write_text(
        "[plugins]\n"
        'android-application = { id = "com.android.application", version = "8.2.0" }\n'
    )


class TestAndroidDetection(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_groovy_dsl(self):
        """Android detection works with Groovy DSL build files."""
        _setup_android_project(self.root, groovy=True)
        result = detect_android(self.root, "auto")
        self.assertIsNotNone(result)
        self.assertIn(":app", result["modules"])
        self.assertTrue(result["modules"][":app"]["android"])

    def test_kotlin_dsl(self):
        """Android detection works with Kotlin DSL build files."""
        _setup_android_project(self.root, groovy=False)
        result = detect_android(self.root, "auto")
        self.assertIsNotNone(result)
        self.assertIn(":app", result["modules"])

    def test_version_catalog_aliases(self):
        """Android detection works with version catalog plugin aliases."""
        _setup_version_catalog_project(self.root)
        result = detect_android(self.root, "auto")
        self.assertIsNotNone(result)
        self.assertIn(":app", result["modules"])

    def test_non_android_project(self):
        """Non-Android projects are not detected."""
        (self.root / "build.gradle").write_text(
            'plugins { id "org.jetbrains.kotlin.jvm" }\n'
        )
        (self.root / "settings.gradle").write_text("rootProject.name = 'test'\n")
        _write_fake_gradlew(self.root)
        result = detect_android(self.root, "auto")
        self.assertIsNone(result)

    def test_no_gradlew(self):
        """No gradlew → not detected."""
        (self.root / "settings.gradle").write_text("")
        (self.root / "build.gradle").write_text('id "com.android.application"')
        result = detect_android(self.root, "auto")
        self.assertIsNone(result)

    def test_mode_off(self):
        _setup_android_project(self.root)
        result = detect_android(self.root, "off")
        self.assertIsNone(result)

    def test_mode_on_forces(self):
        """mode=on detects even without android plugin."""
        (self.root / "settings.gradle").write_text("")
        (self.root / "build.gradle").write_text('id "org.jetbrains.kotlin.jvm"')
        _write_fake_gradlew(self.root)
        result = detect_android(self.root, "on")
        self.assertIsNotNone(result)

    def test_multi_module(self):
        """Multiple modules are discovered."""
        _setup_android_project(self.root, modules={
            "app": True,
            "core/data": False,
            "feature/login": True,
        })
        result = detect_android(self.root, "auto")
        self.assertIsNotNone(result)
        mods = result["modules"]
        self.assertIn(":app", mods)
        self.assertIn(":core:data", mods)
        self.assertIn(":feature:login", mods)
        self.assertTrue(mods[":app"]["android"])
        self.assertFalse(mods[":core:data"]["android"])
        self.assertTrue(mods[":feature:login"]["android"])


class TestModuleMapping(unittest.TestCase):
    def setUp(self):
        self.modules = {
            ":": {"path": ".", "android": False},
            ":app": {"path": "app", "android": True},
            ":core:data": {"path": "core/data", "android": False},
            ":feature:login": {"path": "feature/login", "android": True},
        }

    def test_app_file(self):
        mod = map_file_to_module("app/src/main/kotlin/com/example/App.kt", self.modules)
        self.assertEqual(mod, ":app")

    def test_feature_file(self):
        mod = map_file_to_module("feature/login/src/main/kotlin/Login.kt", self.modules)
        self.assertEqual(mod, ":feature:login")

    def test_root_file(self):
        mod = map_file_to_module("build.gradle", self.modules)
        self.assertEqual(mod, ":")

    def test_no_match(self):
        mod = map_file_to_module("totally/unknown/file.kt", {":app": {"path": "app", "android": True}})
        self.assertIsNone(mod)


class TestCompileTasks(unittest.TestCase):
    def setUp(self):
        self.modules = {
            ":app": {"path": "app", "android": True},
            ":core:data": {"path": "core/data", "android": False},
            ":feature:login": {"path": "feature/login", "android": True},
        }

    def test_kotlin_file_android_module(self):
        """Android module → compileDebugKotlin."""
        tasks, is_bc = compile_tasks(
            ["app/src/main/kotlin/App.kt"], self.modules, "Debug"
        )
        self.assertEqual(tasks, [":app:compileDebugKotlin"])
        self.assertFalse(is_bc)

    def test_kotlin_file_plain_module(self):
        """Non-Android Kotlin module → compileKotlin."""
        tasks, _ = compile_tasks(
            ["core/data/src/main/kotlin/Repo.kt"], self.modules, "Debug"
        )
        self.assertEqual(tasks, [":core:data:compileKotlin"])

    def test_xml_only_android(self):
        """XML-only change in Android module → processDebugResources."""
        tasks, _ = compile_tasks(
            ["app/src/main/res/layout/activity.xml"], self.modules, "Debug"
        )
        self.assertEqual(tasks, [":app:processDebugResources"])

    def test_variant_release(self):
        """AGENT_ANDROID_VARIANT=Release is respected."""
        tasks, _ = compile_tasks(
            ["app/src/main/kotlin/App.kt"], self.modules, "Release"
        )
        self.assertEqual(tasks, [":app:compileReleaseKotlin"])

    def test_two_modules_one_call(self):
        """Two changed files in two modules produce one task list."""
        tasks, _ = compile_tasks(
            ["app/src/main/kotlin/App.kt", "feature/login/src/main/kotlin/Login.kt"],
            self.modules, "Debug",
        )
        self.assertEqual(len(tasks), 2)
        self.assertIn(":app:compileDebugKotlin", tasks)
        self.assertIn(":feature:login:compileDebugKotlin", tasks)

    def test_build_config_change(self):
        """build.gradle change → help task."""
        tasks, is_bc = compile_tasks(
            ["app/build.gradle.kts"], self.modules, "Debug"
        )
        self.assertEqual(tasks, ["help"])
        self.assertTrue(is_bc)

    def test_settings_gradle(self):
        tasks, is_bc = compile_tasks(
            ["settings.gradle.kts"], self.modules, "Debug"
        )
        self.assertTrue(is_bc)

    def test_libs_versions_toml(self):
        tasks, is_bc = compile_tasks(
            ["gradle/libs.versions.toml"], self.modules, "Debug"
        )
        self.assertTrue(is_bc)


class TestParseKotlinErrors(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_new_format(self):
        """Kotlin 1.8+ error format: e: file:///abs/path:line:col message."""
        abs_path = str(self.root / "app/src/main/kotlin/Foo.kt")
        output = f"e: file:///{abs_path}:12:5 Unresolved reference: bar"
        errors, pre = parse_kotlin_errors(output, self.root, {"app/src/main/kotlin/Foo.kt"})
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "app/src/main/kotlin/Foo.kt")
        self.assertEqual(errors[0]["line"], 12)
        self.assertEqual(errors[0]["message"], "Unresolved reference: bar")
        self.assertFalse(pre)

    def test_old_format(self):
        """Older Kotlin error format: e: /abs/path: (line, col): message."""
        abs_path = str(self.root / "app/Foo.kt")
        output = f"e: {abs_path}: (12, 5): Unresolved reference: bar"
        errors, pre = parse_kotlin_errors(output, self.root, {"app/Foo.kt"})
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["line"], 12)

    def test_warnings_ignored(self):
        output = "w: /some/path/Foo.kt: (1, 1): Some warning"
        errors, _ = parse_kotlin_errors(output, self.root, set())
        self.assertEqual(len(errors), 0)

    def test_cap_at_20(self):
        lines = []
        for i in range(25):
            abs_p = str(self.root / f"Foo{i}.kt")
            lines.append(f"e: file:///{abs_p}:{i+1}:1 Error {i}")
        output = "\n".join(lines)
        changed = {f"Foo{i}.kt" for i in range(25)}
        errors, _ = parse_kotlin_errors(output, self.root, changed)
        self.assertEqual(len(errors), 21)  # 20 + 1 "omitted" entry
        self.assertIn("omitted", errors[-1]["message"])

    def test_pre_existing_errors(self):
        """Errors only in unchanged files are flagged as pre-existing."""
        abs_path = str(self.root / "other/File.kt")
        output = f"e: file:///{abs_path}:5:1 Some error"
        errors, pre = parse_kotlin_errors(output, self.root, {"app/Changed.kt"})
        self.assertTrue(pre)

    def test_dedup(self):
        abs_path = str(self.root / "Foo.kt")
        output = (
            f"e: file:///{abs_path}:12:5 Unresolved reference: bar\n"
            f"e: file:///{abs_path}:12:5 Unresolved reference: bar\n"
        )
        errors, _ = parse_kotlin_errors(output, self.root, {"Foo.kt"})
        self.assertEqual(len(errors), 1)


class TestRunGradle(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_success(self):
        _write_fake_gradlew(self.root, output="BUILD SUCCESSFUL", exit_code=0)
        passed, output = run_gradle(self.root, [":app:compileDebugKotlin"])
        self.assertTrue(passed)
        self.assertIn("BUILD SUCCESSFUL", output)
        # Check recorded args
        args = (self.root / ".gradlew_args").read_text().strip()
        self.assertIn(":app:compileDebugKotlin", args)

    def test_failure(self):
        _write_fake_gradlew(self.root, output="FAILURE", exit_code=1)
        passed, output = run_gradle(self.root, [":app:compileDebugKotlin"])
        self.assertFalse(passed)

    def test_two_tasks_one_call(self):
        """Two tasks in one call appear in the same gradlew invocation."""
        _write_fake_gradlew(self.root, output="ok", exit_code=0)
        run_gradle(self.root, [":app:compileDebugKotlin", ":feature:login:compileDebugKotlin"])
        args = (self.root / ".gradlew_args").read_text().strip()
        self.assertIn(":app:compileDebugKotlin", args)
        self.assertIn(":feature:login:compileDebugKotlin", args)


class TestOfflineDependencyError(unittest.TestCase):
    def test_detected(self):
        output = (
            "FAILURE: Build failed with an exception.\n"
            "Could not resolve all dependencies for configuration ':app:debugCompileClasspath'.\n"
            "No cached version of com.google.android.material:material:1.9.0 available for offline mode."
        )
        self.assertTrue(is_offline_dependency_error(output))

    def test_not_detected(self):
        output = "e: file:///Foo.kt:1:1 Unresolved reference: bar"
        self.assertFalse(is_offline_dependency_error(output))


class TestBuildConfigFile(unittest.TestCase):
    def test_build_gradle(self):
        self.assertTrue(is_build_config_file("app/build.gradle"))
        self.assertTrue(is_build_config_file("build.gradle.kts"))

    def test_settings(self):
        self.assertTrue(is_build_config_file("settings.gradle"))
        self.assertTrue(is_build_config_file("settings.gradle.kts"))

    def test_versions_toml(self):
        self.assertTrue(is_build_config_file("gradle/libs.versions.toml"))

    def test_normal_file(self):
        self.assertFalse(is_build_config_file("app/src/main/kotlin/Foo.kt"))


class TestXmlLint(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_valid_xml(self):
        (self.root / "layout.xml").write_text(
            '<?xml version="1.0"?>\n<LinearLayout/>\n'
        )
        errs = builtin_lint(self.root, ["layout.xml"])
        self.assertEqual(errs, [])

    def test_malformed_xml(self):
        (self.root / "bad.xml").write_text("<LinearLayout><unclosed>\n")
        errs = builtin_lint(self.root, ["bad.xml"])
        self.assertEqual(len(errs), 1)
        self.assertIn("XMLParseError", errs[0])
        self.assertIn("bad.xml", errs[0])


class TestPerExtLintCmd(unittest.TestCase):
    def test_lookup(self):
        with patch.dict(os.environ, {"AGENT_LINT_CMD_KT": "ktlint {files}"}):
            self.assertEqual(ext_lint_cmd(".kt"), "ktlint {files}")

    def test_no_env(self):
        self.assertIsNone(ext_lint_cmd(".py"))

    def test_empty_env(self):
        with patch.dict(os.environ, {"AGENT_LINT_CMD_PY": ""}):
            self.assertIsNone(ext_lint_cmd(".py"))


class TestCoderCompileIntegration(unittest.TestCase):
    """Integration test: unresolved reference → fix request → fake model fixes → compile passes."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(
            f"cd {self.root} && git init -q && "
            f"git config user.email test@test && git config user.name test"
        )

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def _make_coder(self, script, gradlew_output="", gradlew_exit=0):
        # type: (list, str, int) -> Coder
        _setup_android_project(self.root)
        _write_fake_gradlew(self.root, output=gradlew_output, exit_code=gradlew_exit)

        self.server = FakeServer(script)
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
        )
        coder = Coder(cfg, mode="code")
        return coder

    def test_compile_error_triggers_fix(self):
        """Unresolved reference error → fix request → fake model fixes → compile passes."""
        _setup_android_project(self.root)

        kt_file = self.root / "app" / "src" / "main" / "kotlin" / "Foo.kt"
        kt_file.parent.mkdir(parents=True, exist_ok=True)
        kt_file.write_text("fun main() { println(bar) }\n")

        abs_path = str(kt_file.resolve())
        error_output = f"e: file:///{abs_path}:1:22 Unresolved reference: bar"

        # Stateful gradlew: first call fails with error, second succeeds
        gradlew = self.root / "gradlew"
        gradlew.write_text(
            "#!/bin/sh\n"
            'COUNTER_FILE="$(dirname "$0")/.gradlew_count"\n'
            'if [ -f "$COUNTER_FILE" ]; then\n'
            '  COUNT=$(cat "$COUNTER_FILE")\n'
            "else\n"
            "  COUNT=0\n"
            "fi\n"
            'COUNT=$((COUNT + 1))\n'
            'echo "$COUNT" > "$COUNTER_FILE"\n'
            'echo "$@" >> "$(dirname "$0")/.gradlew_args"\n'
            'if [ "$COUNT" -eq 1 ]; then\n'
            f'  echo "{error_output}"\n'
            "  exit 1\n"
            "fi\n"
            'echo "BUILD SUCCESSFUL"\n'
            "exit 0\n"
        )
        gradlew.chmod(gradlew.stat().st_mode | stat.S_IEXEC)

        bad_edit = """I'll update Foo.kt.

app/src/main/kotlin/Foo.kt
```kotlin
<<<<<<< SEARCH
fun main() { println(bar) }
=======
fun main() { println(baz) }
>>>>>>> REPLACE
```"""
        fix_edit = """Let me fix that.

app/src/main/kotlin/Foo.kt
```kotlin
<<<<<<< SEARCH
fun main() { println(baz) }
=======
fun main() { val bar = "hello"; println(bar) }
>>>>>>> REPLACE
```"""

        self.server = FakeServer([bad_edit, fix_edit])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
        )
        coder = Coder(cfg, mode="code")
        coder.add(["app/src/main/kotlin/Foo.kt"])
        coder.send("break it then fix it")

        # Should have made 2 model requests
        self.assertEqual(len(self.server.requests), 2)
        # The fix feedback should mention compile
        last_msgs = self.server.requests[1]["messages"]
        feedback = [m for m in last_msgs
                    if m["role"] == "user" and "compile" in m.get("content", "").lower()]
        self.assertTrue(len(feedback) > 0)

    def test_pre_existing_errors_not_sent(self):
        """Errors only in unchanged files are not sent to the model."""
        _setup_android_project(self.root)

        kt_file = self.root / "app" / "src" / "main" / "kotlin" / "Foo.kt"
        kt_file.parent.mkdir(parents=True, exist_ok=True)
        kt_file.write_text("fun main() {}\n")

        # Error in a different file
        other_abs = str((self.root / "app" / "src" / "main" / "kotlin" / "Other.kt").resolve())
        error_output = f"e: file:///{other_abs}:5:1 Some pre-existing error"

        _write_fake_gradlew(self.root, output=error_output, exit_code=1)

        edit = """app/src/main/kotlin/Foo.kt
```kotlin
<<<<<<< SEARCH
fun main() {}
=======
fun main() { println("hello") }
>>>>>>> REPLACE
```"""

        self.server = FakeServer([edit])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
        )
        coder = Coder(cfg, mode="code")
        coder.add(["app/src/main/kotlin/Foo.kt"])
        coder.send("edit it")

        # Only 1 request: no fix attempt since errors are pre-existing
        self.assertEqual(len(self.server.requests), 1)

    def test_offline_dependency_not_sent(self):
        """Offline dependency failure prints hint, not sent to model."""
        _setup_android_project(self.root)

        kt_file = self.root / "app" / "src" / "main" / "kotlin" / "Foo.kt"
        kt_file.parent.mkdir(parents=True, exist_ok=True)
        kt_file.write_text("fun main() {}\n")

        error_output = (
            "FAILURE\n"
            "No cached version of com.example:lib:1.0 available for offline mode.\n"
            "Could not resolve com.example:lib:1.0."
        )
        _write_fake_gradlew(self.root, output=error_output, exit_code=1)

        edit = """app/src/main/kotlin/Foo.kt
```kotlin
<<<<<<< SEARCH
fun main() {}
=======
fun main() { println("test") }
>>>>>>> REPLACE
```"""

        self.server = FakeServer([edit])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
        )
        coder = Coder(cfg, mode="code")
        coder.add(["app/src/main/kotlin/Foo.kt"])
        coder.send("edit it")

        # Only 1 request: offline error not sent to model
        self.assertEqual(len(self.server.requests), 1)

    def test_per_ext_lint_overrides_kotlin(self):
        """AGENT_LINT_CMD_KT overrides the Kotlin check for .kt files only."""
        _setup_android_project(self.root)

        kt_file = self.root / "app" / "src" / "main" / "kotlin" / "Foo.kt"
        kt_file.parent.mkdir(parents=True, exist_ok=True)
        kt_file.write_text("fun main() {}\n")

        edit = """app/src/main/kotlin/Foo.kt
```kotlin
<<<<<<< SEARCH
fun main() {}
=======
fun main() { println("test") }
>>>>>>> REPLACE
```"""
        self.server = FakeServer([edit])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
        )
        with patch.dict(os.environ, {"AGENT_LINT_CMD_KT": "echo ok"}):
            coder = Coder(cfg, mode="code")
            coder.add(["app/src/main/kotlin/Foo.kt"])
            coder.send("edit it")

        # The per-ext command ran (echo ok → exit 0), so no lint errors
        self.assertEqual(len(self.server.requests), 1)


class TestFormatDetectedModules(unittest.TestCase):
    def test_single(self):
        mods = {":app": {"path": "app", "android": True}}
        self.assertEqual(format_detected_modules(mods), ":app")

    def test_multiple(self):
        mods = {
            ":app": {"path": "app", "android": True},
            ":core:data": {"path": "core/data", "android": False},
        }
        result = format_detected_modules(mods)
        self.assertIn(":app", result)
        self.assertIn(":core:data", result)

    def test_root_only(self):
        mods = {":": {"path": ".", "android": False}}
        self.assertEqual(format_detected_modules(mods), "(root)")


class TestModuleDisplayName(unittest.TestCase):
    def test_root(self):
        self.assertEqual(module_display_name(":"), "(root)")

    def test_normal(self):
        self.assertEqual(module_display_name(":app"), ":app")


if __name__ == "__main__":
    unittest.main()
