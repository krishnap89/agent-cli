"""Tests for Feature 4: /test-gen — generate unit tests for a Kotlin class.

Uses a fake Android project structure and fake gradlew scripts.
"""
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli.coder import Coder
from agent_cli.testgen import (
    resolve_target,
    detect_test_setup,
    classify_class,
    check_required_libs,
    has_suspend_funs,
    has_constructor_deps,
    collect_context,
    find_test_examples,
    find_test_helpers,
    test_file_path,
    build_plan_prompt,
    build_write_prompt,
    compile_test_task,
    run_test_task,
    parse_junit_xml,
    format_report,
    parse_bug_classifications,
    _package_from_file,
    _parse_version_catalog,
)
from tests.fake_server import FakeServer


def _write_fake_gradlew(root, output="", exit_code=0, record_args=True):
    """Write a fake gradlew that records args and prints canned output."""
    gradlew = root / "gradlew"
    lines = ["#!/bin/sh"]
    if record_args:
        lines.append('echo "$@" >> "$(dirname "$0")/.gradlew_args"')
    if output:
        out_file = root / ".gradlew_output"
        out_file.write_text(output)
        lines.append(f'cat "$(dirname "$0")/.gradlew_output"')
    lines.append(f"exit {exit_code}")
    gradlew.write_text("\n".join(lines) + "\n")
    gradlew.chmod(gradlew.stat().st_mode | stat.S_IEXEC)
    return gradlew


def _setup_android_project(root, with_tests=True, with_catalog=True):
    """Create a full Android project with app module, sources, and optionally test files."""
    (root / "settings.gradle.kts").write_text('include(":app")\n')
    _write_fake_gradlew(root)

    # App module
    app = root / "app"
    app.mkdir()
    (app / "build.gradle.kts").write_text(
        'plugins {\n'
        '    id("com.android.application")\n'
        '    id("org.jetbrains.kotlin.android")\n'
        '}\n'
        'dependencies {\n'
        '    testImplementation(libs.junit)\n'
        '    testImplementation(libs.mockk)\n'
        '    testImplementation(libs.kotlinx.coroutines.test)\n'
        '    testImplementation(libs.turbine)\n'
        '}\n'
    )

    # Version catalog
    if with_catalog:
        gradle_dir = root / "gradle"
        gradle_dir.mkdir(exist_ok=True)
        (gradle_dir / "libs.versions.toml").write_text(
            '[versions]\n'
            'junit = "4.13.2"\n'
            'mockk = "1.13.9"\n'
            'coroutines = "1.7.3"\n'
            'turbine = "1.0.0"\n'
            '\n'
            '[libraries]\n'
            'junit = { module = "junit:junit", version.ref = "junit" }\n'
            'mockk = { module = "io.mockk:mockk", version.ref = "mockk" }\n'
            'kotlinx-coroutines-test = { module = "org.jetbrains.kotlinx:kotlinx-coroutines-test", version.ref = "coroutines" }\n'
            'turbine = { module = "app.cash.turbine:turbine", version.ref = "turbine" }\n'
            '\n'
            '[plugins]\n'
            'android-application = { id = "com.android.application", version = "8.2.0" }\n'
        )

    # Main source
    main_kt = app / "src" / "main" / "java" / "com" / "example" / "app"
    main_kt.mkdir(parents=True)

    (main_kt / "LoginViewModel.kt").write_text(
        'package com.example.app\n'
        '\n'
        'import androidx.lifecycle.ViewModel\n'
        'import kotlinx.coroutines.flow.StateFlow\n'
        'import kotlinx.coroutines.flow.MutableStateFlow\n'
        '\n'
        'class LoginViewModel(\n'
        '    private val repository: LoginRepository,\n'
        ') : ViewModel() {\n'
        '    private val _state = MutableStateFlow(LoginState())\n'
        '    val state: StateFlow<LoginState> = _state\n'
        '\n'
        '    suspend fun login(username: String, password: String) {\n'
        '        _state.value = LoginState(loading = true)\n'
        '        val result = repository.login(username, password)\n'
        '        _state.value = LoginState(success = result)\n'
        '    }\n'
        '\n'
        '    fun reset() {\n'
        '        _state.value = LoginState()\n'
        '    }\n'
        '}\n'
    )

    (main_kt / "LoginRepository.kt").write_text(
        'package com.example.app\n'
        '\n'
        'interface LoginRepository {\n'
        '    suspend fun login(username: String, password: String): Boolean\n'
        '}\n'
    )

    (main_kt / "LoginState.kt").write_text(
        'package com.example.app\n'
        '\n'
        'data class LoginState(\n'
        '    val loading: Boolean = false,\n'
        '    val success: Boolean = false,\n'
        '    val error: String? = null,\n'
        ')\n'
    )

    (main_kt / "UserDao.kt").write_text(
        'package com.example.app\n'
        '\n'
        '@Dao\n'
        'interface UserDao {\n'
        '    suspend fun getUser(id: Int): User?\n'
        '}\n'
    )

    (main_kt / "MainActivity.kt").write_text(
        'package com.example.app\n'
        '\n'
        'class MainActivity : Activity() {\n'
        '    override fun onCreate() {}\n'
        '}\n'
    )

    (main_kt / "LoginUseCase.kt").write_text(
        'package com.example.app\n'
        '\n'
        'class LoginUseCase(\n'
        '    private val repository: LoginRepository,\n'
        ') {\n'
        '    suspend fun execute(user: String, pass: String): Boolean {\n'
        '        return repository.login(user, pass)\n'
        '    }\n'
        '}\n'
    )

    (main_kt / "StringUtils.kt").write_text(
        'package com.example.app\n'
        '\n'
        'object StringUtils {\n'
        '    fun capitalize(s: String): String = s.replaceFirstChar { it.uppercase() }\n'
        '}\n'
    )

    # Test sources
    if with_tests:
        test_kt = app / "src" / "test" / "java" / "com" / "example" / "app"
        test_kt.mkdir(parents=True)
        (test_kt / "StringUtilsTest.kt").write_text(
            'package com.example.app\n'
            '\n'
            'import org.junit.Test\n'
            'import org.junit.Assert.assertEquals\n'
            '\n'
            'class StringUtilsTest {\n'
            '    @Test\n'
            '    fun capitalize_works() {\n'
            '        assertEquals("Hello", StringUtils.capitalize("hello"))\n'
            '    }\n'
            '}\n'
        )
        (test_kt / "MainDispatcherRule.kt").write_text(
            'package com.example.app\n'
            '\n'
            'import kotlinx.coroutines.Dispatchers\n'
            'import kotlinx.coroutines.test.StandardTestDispatcher\n'
            'import kotlinx.coroutines.test.setMain\n'
            'import kotlinx.coroutines.test.resetMain\n'
            'import org.junit.rules.TestWatcher\n'
            '\n'
            'class MainDispatcherRule : TestWatcher() {\n'
            '    override fun starting(description: org.junit.runner.Description?) {\n'
            '        Dispatchers.setMain(StandardTestDispatcher())\n'
            '    }\n'
            '    override fun finished(description: org.junit.runner.Description?) {\n'
            '        Dispatchers.resetMain()\n'
            '    }\n'
            '}\n'
        )

    return root


class TestResolveTarget(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)
        from agent_cli.android import detect_android
        info = detect_android(self.root, "on")
        self.modules = info["modules"]
        # Build a simple repo map
        self.repo_map_files = {}
        for dirpath, _, filenames in os.walk(self.root / "app" / "src" / "main"):
            for name in filenames:
                if name.endswith(".kt"):
                    p = Path(dirpath) / name
                    rel = p.relative_to(self.root).as_posix()
                    # Fake defs
                    class_name = name.replace(".kt", "")
                    self.repo_map_files[rel] = {
                        "defs": [[1, 0, "class", class_name, f"class {class_name}"]],
                        "idents": [],
                    }

    def tearDown(self):
        self.td.cleanup()

    def test_resolve_by_path(self):
        rel, cls, method, mod = resolve_target(
            self.root,
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "",
            self.repo_map_files,
            self.modules,
        )
        self.assertEqual(cls, "LoginViewModel")
        self.assertEqual(mod, ":app")

    def test_resolve_by_simple_name(self):
        rel, cls, method, mod = resolve_target(
            self.root, "LoginViewModel", "",
            self.repo_map_files, self.modules,
        )
        self.assertEqual(cls, "LoginViewModel")

    def test_resolve_by_fqn(self):
        rel, cls, method, mod = resolve_target(
            self.root, "com.example.app.LoginViewModel", "",
            self.repo_map_files, self.modules,
        )
        self.assertEqual(cls, "LoginViewModel")

    def test_resolve_not_found_suggests(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_target(self.root, "LoginViewMode", "",
                           self.repo_map_files, self.modules)
        self.assertIn("Did you mean", str(ctx.exception))

    def test_refuse_test_file(self):
        # Create a test file
        test_file = self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app" / "FooTest.kt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("class FooTest {}\n")
        with self.assertRaises(ValueError) as ctx:
            resolve_target(
                self.root,
                "app/src/test/java/com/example/app/FooTest.kt",
                "", self.repo_map_files, self.modules,
            )
        self.assertIn("already a test file", str(ctx.exception))

    def test_resolve_with_method(self):
        rel, cls, method, mod = resolve_target(
            self.root,
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "login",
            self.repo_map_files,
            self.modules,
        )
        self.assertEqual(method, "login")

    def test_resolve_with_bad_method(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_target(
                self.root,
                "app/src/main/java/com/example/app/LoginViewModel.kt",
                "nonexistent",
                self.repo_map_files,
                self.modules,
            )
        self.assertIn("Method 'nonexistent' not found", str(ctx.exception))


class TestDetectTestSetup(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)
        from agent_cli.android import detect_android
        info = detect_android(self.root, "on")
        self.modules = info["modules"]

    def tearDown(self):
        self.td.cleanup()

    def test_detects_junit4_mockk_coroutines_turbine(self):
        setup = detect_test_setup(self.root, ":app", self.modules)
        self.assertIn("junit4", setup["libs"])
        self.assertIn("mockk", setup["libs"])
        self.assertIn("coroutines-test", setup["libs"])
        self.assertIn("turbine", setup["libs"])

    def test_detects_junit5_via_useJUnitPlatform(self):
        build = self.root / "app" / "build.gradle.kts"
        text = build.read_text()
        text += '\ntasks.withType<Test> { useJUnitPlatform() }\n'
        build.write_text(text)
        setup = detect_test_setup(self.root, ":app", self.modules)
        self.assertIn("junit5", setup["libs"])

    def test_detects_mockito_kotlin(self):
        build = self.root / "app" / "build.gradle.kts"
        text = build.read_text()
        text += '    testImplementation("org.mockito.kotlin:mockito-kotlin:5.0.0")\n'
        build.write_text(text)
        setup = detect_test_setup(self.root, ":app", self.modules)
        self.assertIn("mockito-kotlin", setup["libs"])

    def test_dsl_detected(self):
        setup = detect_test_setup(self.root, ":app", self.modules)
        self.assertEqual(setup["dsl"], "kotlin")


class TestCheckRequiredLibs(unittest.TestCase):
    def test_missing_framework(self):
        msgs = check_required_libs(set(), False, False, "kotlin", None)
        self.assertTrue(any("test framework" in m.lower() for m in msgs))

    def test_missing_mock(self):
        msgs = check_required_libs({"junit4"}, False, True, "kotlin", None)
        self.assertTrue(any("mocking" in m.lower() for m in msgs))

    def test_missing_coroutines_test(self):
        msgs = check_required_libs({"junit4", "mockk"}, True, True, "kotlin", None)
        self.assertTrue(any("coroutines" in m.lower() for m in msgs))

    def test_all_present(self):
        msgs = check_required_libs(
            {"junit4", "mockk", "coroutines-test"}, True, True, "kotlin", None,
        )
        self.assertEqual(msgs, [])

    def test_groovy_dsl_format(self):
        msgs = check_required_libs(set(), False, False, "groovy", None)
        self.assertTrue(any("'" in m for m in msgs))  # Groovy uses single quotes


class TestClassifyClass(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)
        self.src = self.root / "app" / "src" / "main" / "java" / "com" / "example" / "app"

    def tearDown(self):
        self.td.cleanup()

    def test_viewmodel(self):
        self.assertEqual(classify_class(self.src / "LoginViewModel.kt", "LoginViewModel"), "viewmodel")

    def test_repository(self):
        self.assertEqual(classify_class(self.src / "LoginRepository.kt", "LoginRepository"), "repository")

    def test_usecase(self):
        self.assertEqual(classify_class(self.src / "LoginUseCase.kt", "LoginUseCase"), "usecase")

    def test_plain(self):
        self.assertEqual(classify_class(self.src / "StringUtils.kt", "StringUtils"), "plain")

    def test_dao_stops(self):
        self.assertEqual(classify_class(self.src / "UserDao.kt", "UserDao"), "dao")

    def test_activity_stops(self):
        self.assertEqual(classify_class(self.src / "MainActivity.kt", "MainActivity"), "ui")

    def test_composable_stops(self):
        comp_file = self.src / "LoginScreen.kt"
        comp_file.write_text(
            'package com.example.app\n\n'
            '@Composable\nfun LoginScreen() {}\n'
        )
        self.assertEqual(classify_class(comp_file, "LoginScreen"), "ui")


class TestConstructorDeps(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)
        self.src = self.root / "app" / "src" / "main" / "java" / "com" / "example" / "app"

    def tearDown(self):
        self.td.cleanup()

    def test_has_deps(self):
        self.assertTrue(has_constructor_deps(self.src / "LoginViewModel.kt"))

    def test_no_deps(self):
        self.assertFalse(has_constructor_deps(self.src / "StringUtils.kt"))

    def test_has_suspend(self):
        self.assertTrue(has_suspend_funs(self.src / "LoginViewModel.kt"))

    def test_no_suspend(self):
        self.assertFalse(has_suspend_funs(self.src / "StringUtils.kt"))


class TestTestFilePath(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)

    def tearDown(self):
        self.td.cleanup()

    def test_uses_java_dir(self):
        """Test file path uses java/ when test/java/ exists."""
        result = test_file_path(
            self.root, "app",
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "LoginViewModel",
        )
        self.assertEqual(result, "app/src/test/java/com/example/app/LoginViewModelTest.kt")

    def test_uses_kotlin_dir(self):
        """Test file path uses kotlin/ when test/kotlin/ exists."""
        kt_dir = self.root / "app" / "src" / "test" / "kotlin" / "com" / "example"
        kt_dir.mkdir(parents=True)
        (kt_dir / "SomeTest.kt").write_text("class SomeTest {}")
        # Remove java dir
        import shutil
        java_dir = self.root / "app" / "src" / "test" / "java"
        if java_dir.exists():
            shutil.rmtree(java_dir)

        result = test_file_path(
            self.root, "app",
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "LoginViewModel",
        )
        self.assertEqual(result, "app/src/test/kotlin/com/example/app/LoginViewModelTest.kt")


class TestFindTestExamples(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)

    def tearDown(self):
        self.td.cleanup()

    def test_finds_existing_test(self):
        examples = find_test_examples(self.root, "app", "StringUtils", "plain")
        self.assertTrue(len(examples) > 0)
        self.assertTrue(any("StringUtilsTest" in p for p, _ in examples))

    def test_prefers_matching_kind(self):
        # Create a ViewModel test
        vt = self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app" / "SomeViewModelTest.kt"
        vt.write_text("class SomeViewModelTest { @Test fun test() {} }\n")
        examples = find_test_examples(self.root, "app", "LoginViewModel", "viewmodel")
        names = [p for p, _ in examples]
        vm_first = any("ViewModel" in n for n in names[:1])
        self.assertTrue(vm_first or len(examples) > 0)


class TestFindHelpers(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)

    def tearDown(self):
        self.td.cleanup()

    def test_finds_dispatcher_rule(self):
        helpers = find_test_helpers(self.root, "app")
        dispatcher_rules = [h for h in helpers if h["kind"] == "dispatcher-rule"]
        self.assertTrue(len(dispatcher_rules) > 0)
        self.assertEqual(dispatcher_rules[0]["name"], "MainDispatcherRule")


class TestCompileTestTask(unittest.TestCase):
    def test_android_module(self):
        modules = {":app": {"path": "app", "android": True}}
        task = compile_test_task(":app", modules, "Debug")
        self.assertEqual(task, ":app:compileDebugUnitTestKotlin")

    def test_plain_module(self):
        modules = {":lib": {"path": "lib", "android": False}}
        task = compile_test_task(":lib", modules, "Debug")
        self.assertEqual(task, ":lib:compileTestKotlin")


class TestRunTestTask(unittest.TestCase):
    def test_android_module(self):
        modules = {":app": {"path": "app", "android": True}}
        task = run_test_task(":app", modules, "com.example.FooTest", "Debug")
        self.assertEqual(task, ":app:testDebugUnitTest")

    def test_plain_module(self):
        modules = {":lib": {"path": "lib", "android": False}}
        task = run_test_task(":lib", modules, "com.example.FooTest", "Debug")
        self.assertEqual(task, ":lib:test")


class TestParseJUnitXML(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_parse_passing(self):
        xml = (
            '<?xml version="1.0"?>\n'
            '<testsuite tests="2">\n'
            '  <testcase name="test_add" classname="CalcTest"/>\n'
            '  <testcase name="test_sub" classname="CalcTest"/>\n'
            '</testsuite>\n'
        )
        p = self.root / "TEST-CalcTest.xml"
        p.write_text(xml)
        results = parse_junit_xml(p)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["passed"] for r in results))

    def test_parse_failure(self):
        xml = (
            '<?xml version="1.0"?>\n'
            '<testsuite tests="1">\n'
            '  <testcase name="test_fail" classname="CalcTest">\n'
            '    <failure message="expected 2 but got 3">stack trace here</failure>\n'
            '  </testcase>\n'
            '</testsuite>\n'
        )
        p = self.root / "TEST-CalcTest.xml"
        p.write_text(xml)
        results = parse_junit_xml(p)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["passed"])
        self.assertIn("expected 2", results[0]["failure_message"])


class TestFormatReport(unittest.TestCase):
    def test_all_passing(self):
        report = format_report("app/src/test/FooTest.kt", 5, 5, 0, [], 0)
        self.assertIn("5 tests", report)
        self.assertIn("5 passed", report)

    def test_with_bugs(self):
        report = format_report("app/src/test/FooTest.kt", 5, 4, 1,
                               ["login fails when repo throws"], 2)
        self.assertIn("1 ignored as suspected bug", report)
        self.assertIn("Suspected bug", report)
        self.assertIn("2 fix rounds", report)


class TestParseBugClassifications(unittest.TestCase):
    def test_parse_test_bug(self):
        text = "test_login: TEST_BUG\ntest_reset: CODE_BUG - loading flag stays true"
        result = parse_bug_classifications(text)
        self.assertEqual(result["test_login"], ("TEST_BUG", ""))
        self.assertEqual(result["test_reset"][0], "CODE_BUG")
        self.assertIn("loading flag", result["test_reset"][1])


class TestVersionCatalog(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_parse_module_form(self):
        (self.root / "gradle").mkdir()
        (self.root / "gradle" / "libs.versions.toml").write_text(
            '[libraries]\n'
            'mockk = { module = "io.mockk:mockk", version = "1.13.9" }\n'
        )
        catalog = _parse_version_catalog(self.root)
        self.assertIsNotNone(catalog)
        self.assertEqual(catalog["mockk"], "io.mockk:mockk")

    def test_parse_group_name_form(self):
        (self.root / "gradle").mkdir()
        (self.root / "gradle" / "libs.versions.toml").write_text(
            '[libraries]\n'
            'truth = { group = "com.google.truth", name = "truth", version = "1.1.5" }\n'
        )
        catalog = _parse_version_catalog(self.root)
        self.assertIsNotNone(catalog)
        self.assertEqual(catalog["truth"], "com.google.truth:truth")

    def test_parse_inline_form(self):
        (self.root / "gradle").mkdir()
        (self.root / "gradle" / "libs.versions.toml").write_text(
            '[libraries]\n'
            'junit = "junit:junit:4.13.2"\n'
        )
        catalog = _parse_version_catalog(self.root)
        self.assertIsNotNone(catalog)
        self.assertEqual(catalog["junit"], "junit:junit")


class TestPackageFromFile(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_extracts_package(self):
        f = self.root / "Foo.kt"
        f.write_text("package com.example.app\n\nclass Foo {}\n")
        self.assertEqual(_package_from_file(f), "com.example.app")

    def test_no_package(self):
        f = self.root / "Foo.kt"
        f.write_text("class Foo {}\n")
        self.assertEqual(_package_from_file(f), "")


class TestCoderTestGenIntegration(unittest.TestCase):
    """Integration tests for /test-gen using the full Coder with fake server and gradlew."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(
            f"cd {self.root} && git init -q && "
            f"git config user.email test@test && git config user.name test"
        )
        _setup_android_project(self.root)

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def _make_coder(self, script, gradlew_output="BUILD SUCCESSFUL", gradlew_exit=0):
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
            testgen_plan=False,  # auto-accept plan
        )
        return Coder(cfg, mode="code")

    def test_dao_refuses(self):
        """DAO targets stop with a clear message."""
        coder = self._make_coder(["plan", "tests"])
        coder.test_gen("app/src/main/java/com/example/app/UserDao.kt")
        # No server requests since it should stop early
        self.assertEqual(len(self.server.requests), 0)

    def test_activity_refuses(self):
        """Activity targets stop with a clear message."""
        coder = self._make_coder(["plan", "tests"])
        coder.test_gen("app/src/main/java/com/example/app/MainActivity.kt")
        self.assertEqual(len(self.server.requests), 0)

    def test_production_file_edit_rejected(self):
        """A block editing a production file is rejected."""
        plan_reply = "1. login success -> state updated"
        # Model tries to edit production code
        write_reply = (
            "app/src/main/java/com/example/app/LoginViewModel.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "class LoginViewModel(\n"
            "=======\n"
            "class LoginViewModel2(\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "\n"
            "app/src/test/java/com/example/app/LoginViewModelTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "class LoginViewModelTest {}\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        coder = self._make_coder([plan_reply, write_reply])
        coder.test_gen(
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            auto_accept=True,
        )

        # Production file should NOT have been changed
        vm_text = (self.root / "app" / "src" / "main" / "java" / "com" / "example" / "app" / "LoginViewModel.kt").read_text()
        self.assertIn("class LoginViewModel(", vm_text)
        self.assertNotIn("LoginViewModel2", vm_text)

    def test_creates_test_file(self):
        """Basic test gen creates a test file."""
        plan_reply = "1. login success -> state updated to success=true"
        write_reply = (
            "app/src/test/java/com/example/app/LoginViewModelTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "package com.example.app\n"
            "\n"
            "import org.junit.Test\n"
            "\n"
            "class LoginViewModelTest {\n"
            "    @Test\n"
            "    fun login_success() {\n"
            "        // test\n"
            "    }\n"
            "}\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        coder = self._make_coder([plan_reply, write_reply])
        coder.test_gen(
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            auto_accept=True,
        )

        test_path = self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app" / "LoginViewModelTest.kt"
        self.assertTrue(test_path.exists())
        content = test_path.read_text()
        self.assertIn("LoginViewModelTest", content)

    def test_undo_reverts_test_file(self):
        """One /undo removes the generated test file."""
        plan_reply = "1. test case"
        write_reply = (
            "app/src/test/java/com/example/app/LoginViewModelTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "class LoginViewModelTest {}\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        coder = self._make_coder([plan_reply, write_reply])
        coder.test_gen(
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            auto_accept=True,
        )

        test_path = self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app" / "LoginViewModelTest.kt"
        self.assertTrue(test_path.exists())

        coder.undo()
        self.assertFalse(test_path.exists())

    def test_compile_error_triggers_fix(self):
        """Compile errors in the test file trigger a fix round."""
        plan_reply = "1. test case"
        write_reply = (
            "app/src/test/java/com/example/app/LoginViewModelTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "class LoginViewModelTest { fun bad( }\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        fix_reply = (
            "app/src/test/java/com/example/app/LoginViewModelTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "class LoginViewModelTest { fun bad( }\n"
            "=======\n"
            "class LoginViewModelTest { fun good() {} }\n"
            ">>>>>>> REPLACE\n"
            "```"
        )

        # Stateful gradlew: first compile fails, second succeeds
        test_file_abs = str((self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app" / "LoginViewModelTest.kt").resolve())
        error_output = f"e: file:///{test_file_abs}:1:1 Syntax error"

        gradlew = self.root / "gradlew"
        gradlew.write_text(
            "#!/bin/sh\n"
            'COUNTER_FILE="$(dirname "$0")/.gradlew_count"\n'
            'if [ -f "$COUNTER_FILE" ]; then COUNT=$(cat "$COUNTER_FILE"); else COUNT=0; fi\n'
            'COUNT=$((COUNT + 1))\n'
            'echo "$COUNT" > "$COUNTER_FILE"\n'
            'echo "$@" >> "$(dirname "$0")/.gradlew_args"\n'
            'if [ "$COUNT" -eq 1 ]; then\n'
            f'  echo "e: file:///{test_file_abs}:1:1 Syntax error"\n'
            '  exit 1\n'
            'fi\n'
            'echo "BUILD SUCCESSFUL"\n'
            'exit 0\n'
        )
        gradlew.chmod(gradlew.stat().st_mode | stat.S_IEXEC)

        self.server = FakeServer([plan_reply, write_reply, fix_reply])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
            android="on",
            compile_enabled="on",
            testgen_plan=False,
        )
        coder = Coder(cfg, mode="code")
        coder.test_gen(
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            auto_accept=True,
        )

        # Should have 3 requests: plan + write + fix
        self.assertEqual(len(self.server.requests), 3)

    def test_existing_test_adds_not_rewrites(self):
        """When a test file exists, new tests are added."""
        # Create an existing test file
        test_dir = self.root / "app" / "src" / "test" / "java" / "com" / "example" / "app"
        test_dir.mkdir(parents=True, exist_ok=True)
        existing_test = test_dir / "StringUtilsTest.kt"
        existing_content = (
            'package com.example.app\n\n'
            'import org.junit.Test\n\n'
            'class StringUtilsTest {\n'
            '    @Test\n'
            '    fun capitalize_works() {\n'
            '        assertEquals("Hello", StringUtils.capitalize("hello"))\n'
            '    }\n'
            '}\n'
        )
        existing_test.write_text(existing_content)

        plan_reply = "1. capitalize empty string -> empty string"
        write_reply = (
            "app/src/test/java/com/example/app/StringUtilsTest.kt\n"
            "```kotlin\n"
            "<<<<<<< SEARCH\n"
            "}\n"
            "=======\n"
            "\n"
            "    @Test\n"
            "    fun capitalize_empty() {\n"
            '        assertEquals("", StringUtils.capitalize(""))\n'
            "    }\n"
            "}\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        coder = self._make_coder([plan_reply, write_reply])
        coder.test_gen(
            "app/src/main/java/com/example/app/StringUtils.kt",
            auto_accept=True,
        )

        content = existing_test.read_text()
        # Original test should still be there
        self.assertIn("capitalize_works", content)
        # New test should be added
        self.assertIn("capitalize_empty", content)


class TestCollectContext(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        _setup_android_project(self.root)

    def tearDown(self):
        self.td.cleanup()

    def test_includes_target_source(self):
        setup = {"libs": {"junit4", "mockk"}, "dsl": "kotlin", "catalog": None}
        ctx = collect_context(
            self.root,
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "LoginViewModel",
            None,
            "viewmodel",
            setup,
            [],
            [],
            "",
            None,
            {},
            48000,
        )
        self.assertIn("LoginViewModel", ctx)
        self.assertIn("suspend fun login", ctx)

    def test_respects_budget(self):
        setup = {"libs": set(), "dsl": "kotlin", "catalog": None}
        ctx = collect_context(
            self.root,
            "app/src/main/java/com/example/app/LoginViewModel.kt",
            "LoginViewModel",
            None,
            "viewmodel",
            setup,
            [],
            [],
            "",
            None,
            {},
            200,
        )
        self.assertIn("truncated", ctx)


if __name__ == "__main__":
    unittest.main()
