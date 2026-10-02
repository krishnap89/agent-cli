"""Tests for Feature 5: /doc command (KDoc generation)."""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli.docgen import (
    scan_declarations, filter_declarations, detect_kdoc_style,
    build_style_prompt, build_doc_messages, find_outdated_kdoc,
    verify_code_unchanged, verify_kdoc_balanced, resolve_doc_target,
    strip_comments_and_whitespace, format_doc_report, Declaration,
    _strip_strings_and_comments,
)


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------

SAMPLE_KT = """\
package com.example

import android.os.Bundle

/**
 * Already documented class.
 */
class Documented {
    fun alreadyDoc() {}
}

@Suppress("unused")
class UserRepository(
    private val api: Api,
    private val dao: UserDao,
) {

    suspend fun getUser(id: String): User {
        return api.fetchUser(id) ?: dao.getUser(id)
    }

    fun deleteUser(id: String) {
        dao.delete(id)
    }

    private fun internalHelper() {}

    override fun toString(): String = "UserRepository"

    val userName: String = "test"
}

interface Authenticator {
    fun login(email: String, password: String): Boolean
}

object Analytics {
    fun track(event: String) {}
}

enum class Status {
    ACTIVE, INACTIVE
}

fun topLevelFun(x: Int): Int = x + 1

internal fun internalFun() {}

private fun privateFun() {}
"""


class TestScanDeclarations(unittest.TestCase):

    def test_finds_classes(self):
        decls = scan_declarations(SAMPLE_KT)
        names = [d.name for d in decls]
        self.assertIn("Documented", names)
        self.assertIn("UserRepository", names)

    def test_finds_interface(self):
        decls = scan_declarations(SAMPLE_KT)
        ifaces = [d for d in decls if d.kind == "interface"]
        self.assertEqual(len(ifaces), 1)
        self.assertEqual(ifaces[0].name, "Authenticator")

    def test_finds_object(self):
        decls = scan_declarations(SAMPLE_KT)
        objs = [d for d in decls if d.kind == "object"]
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0].name, "Analytics")

    def test_finds_enum(self):
        decls = scan_declarations(SAMPLE_KT)
        enums = [d for d in decls if d.kind == "enum"]
        self.assertEqual(len(enums), 1)
        self.assertEqual(enums[0].name, "Status")

    def test_finds_functions(self):
        decls = scan_declarations(SAMPLE_KT)
        funs = [d for d in decls if d.kind == "fun"]
        fun_names = [d.name for d in funs]
        self.assertIn("getUser", fun_names)
        self.assertIn("deleteUser", fun_names)
        self.assertIn("topLevelFun", fun_names)
        self.assertIn("login", fun_names)
        self.assertIn("track", fun_names)

    def test_finds_properties(self):
        decls = scan_declarations(SAMPLE_KT)
        props = [d for d in decls if d.kind in ("val", "var")]
        prop_names = [d.name for d in props]
        self.assertIn("userName", prop_names)

    def test_detects_existing_kdoc(self):
        decls = scan_declarations(SAMPLE_KT)
        documented = [d for d in decls if d.name == "Documented"]
        self.assertEqual(len(documented), 1)
        self.assertTrue(documented[0].has_kdoc)

    def test_detects_no_kdoc(self):
        decls = scan_declarations(SAMPLE_KT)
        repo = [d for d in decls if d.name == "UserRepository"]
        self.assertEqual(len(repo), 1)
        self.assertFalse(repo[0].has_kdoc)

    def test_detects_visibility(self):
        decls = scan_declarations(SAMPLE_KT)
        by_name = {d.name: d for d in decls}
        self.assertEqual(by_name["internalHelper"].visibility, "private")
        self.assertEqual(by_name["internalFun"].visibility, "internal")
        self.assertEqual(by_name["privateFun"].visibility, "private")
        self.assertEqual(by_name["UserRepository"].visibility, "public")

    def test_detects_override(self):
        decls = scan_declarations(SAMPLE_KT)
        to_string = [d for d in decls if d.name == "toString"]
        self.assertEqual(len(to_string), 1)
        self.assertTrue(to_string[0].is_override)

    def test_detects_annotations(self):
        decls = scan_declarations(SAMPLE_KT)
        repo = [d for d in decls if d.name == "UserRepository"]
        self.assertEqual(len(repo), 1)
        self.assertTrue(any("Suppress" in a for a in repo[0].annotations))

    def test_multiline_signature(self):
        src = """\
fun processData(
    input: String,
    config: Config,
    callback: (Result) -> Unit,
): Boolean {
    return true
}
"""
        decls = scan_declarations(src)
        self.assertEqual(len(decls), 1)
        self.assertIn("input", decls[0].signature)
        self.assertIn("callback", decls[0].signature)

    def test_expression_body(self):
        src = "fun double(x: Int): Int = x * 2\n"
        decls = scan_declarations(src)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0].name, "double")

    def test_generic_function(self):
        src = "fun <T> asList(vararg items: T): List<T> = items.toList()\n"
        decls = scan_declarations(src)
        self.assertTrue(any("asList" in d.name for d in decls))

    def test_extension_function(self):
        src = "fun String.isEmail(): Boolean = contains(\"@\")\n"
        decls = scan_declarations(src)
        names = [d.name for d in decls]
        self.assertTrue(any("isEmail" in n for n in names))

    def test_ignores_declarations_in_strings(self):
        src = '''\
val message = """
class NotAClass {
    fun notAFun() {}
}
"""

class RealClass {
    fun realFun() {}
}
'''
        decls = scan_declarations(src)
        names = [d.name for d in decls]
        self.assertIn("RealClass", names)
        self.assertIn("realFun", names)
        self.assertNotIn("NotAClass", names)
        self.assertNotIn("notAFun", names)

    def test_ignores_declarations_in_comments(self):
        src = """\
// class CommentedOut {}
/* fun alsoCommented() {} */
class ActualClass {}
"""
        decls = scan_declarations(src)
        names = [d.name for d in decls]
        self.assertIn("ActualClass", names)
        self.assertNotIn("CommentedOut", names)
        self.assertNotIn("alsoCommented", names)

    def test_companion_object(self):
        src = """\
class Foo {
    companion object {
        fun create(): Foo = Foo()
    }
}
"""
        decls = scan_declarations(src)
        names = [d.name for d in decls]
        self.assertIn("Foo", names)
        self.assertIn("create", names)

    def test_data_class(self):
        src = "data class User(val name: String, val age: Int)\n"
        decls = scan_declarations(src)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0].kind, "class")
        self.assertEqual(decls[0].name, "User")

    def test_sealed_class(self):
        src = """\
sealed class Result {
    data class Success(val data: String) : Result()
    data class Error(val message: String) : Result()
}
"""
        decls = scan_declarations(src)
        names = [d.name for d in decls]
        self.assertIn("Result", names)
        self.assertIn("Success", names)
        self.assertIn("Error", names)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestFilterDeclarations(unittest.TestCase):

    def _make_decl(self, name="Foo", kind="class", visibility="public",
                   has_kdoc=False, is_override=False):
        return Declaration(
            kind=kind, name=name, start_line=1, decl_line=1, end_line=5,
            visibility=visibility, has_kdoc=has_kdoc, is_override=is_override,
            signature=f"{kind} {name}", annotations=[],
        )

    def test_skips_private(self):
        d = self._make_decl(visibility="private")
        result = filter_declarations([d], "src/main/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_skips_override(self):
        d = self._make_decl(is_override=True)
        result = filter_declarations([d], "src/main/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_skips_existing_kdoc(self):
        d = self._make_decl(has_kdoc=True)
        result = filter_declarations([d], "src/main/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_includes_existing_kdoc_in_update_mode(self):
        d = self._make_decl(has_kdoc=True)
        result = filter_declarations([d], "src/main/Foo.kt", update_mode=True)
        self.assertEqual(len(result), 1)

    def test_skips_test_files(self):
        d = self._make_decl()
        result = filter_declarations([d], "app/src/test/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_skips_generated_code(self):
        d = self._make_decl()
        result = filter_declarations([d], "app/build/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_visibility_public_filter(self):
        d_pub = self._make_decl(name="Pub", visibility="public")
        d_int = self._make_decl(name="Int", visibility="internal")
        result = filter_declarations([d_pub, d_int], "src/main/Foo.kt",
                                     visibility_filter="public")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Pub")

    def test_skips_properties_by_default(self):
        d = self._make_decl(kind="val")
        result = filter_declarations([d], "src/main/Foo.kt")
        self.assertEqual(len(result), 0)

    def test_includes_properties_when_enabled(self):
        d = self._make_decl(kind="val")
        result = filter_declarations([d], "src/main/Foo.kt", include_properties=True)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# String/comment stripping
# ---------------------------------------------------------------------------

class TestStripStringsAndComments(unittest.TestCase):

    def test_strips_line_comments(self):
        src = "val x = 1 // comment\nval y = 2\n"
        cleaned = _strip_strings_and_comments(src)
        self.assertNotIn("comment", cleaned)
        self.assertIn("val x = 1", cleaned)

    def test_strips_block_comments(self):
        src = "val x = /* hidden */ 1\n"
        cleaned = _strip_strings_and_comments(src)
        self.assertNotIn("hidden", cleaned)

    def test_preserves_strings(self):
        src = 'val x = "class Foo"\n'
        cleaned = _strip_strings_and_comments(src)
        # String content is replaced with spaces but quotes are preserved
        self.assertIn('"', cleaned)

    def test_preserves_raw_strings(self):
        src = 'val x = """class Foo"""\n'
        cleaned = _strip_strings_and_comments(src)
        self.assertIn('"""', cleaned)

    def test_preserves_line_structure(self):
        src = "line1\n// comment\nline3\n"
        cleaned = _strip_strings_and_comments(src)
        self.assertEqual(cleaned.count("\n"), src.count("\n"))


# ---------------------------------------------------------------------------
# Code-change guardrail
# ---------------------------------------------------------------------------

class TestVerifyCodeUnchanged(unittest.TestCase):

    def test_identical_code(self):
        old = "fun foo() { return 1 }\n"
        new = "/** Docs. */\nfun foo() { return 1 }\n"
        self.assertIsNone(verify_code_unchanged(old, new))

    def test_code_changed(self):
        old = "fun foo() { return 1 }\n"
        new = "/** Docs. */\nfun foo() { return 2 }\n"
        result = verify_code_unchanged(old, new)
        self.assertIsNotNone(result)

    def test_whitespace_only_change(self):
        old = "fun foo()  { return 1 }\n"
        new = "fun foo() { return 1 }\n"
        self.assertIsNone(verify_code_unchanged(old, new))

    def test_comment_change_only(self):
        old = "// old comment\nfun foo() {}\n"
        new = "/** new KDoc. */\nfun foo() {}\n"
        self.assertIsNone(verify_code_unchanged(old, new))

    def test_string_content_matters(self):
        old = 'val x = "hello"\n'
        new = '/** Doc. */\nval x = "world"\n'
        result = verify_code_unchanged(old, new)
        self.assertIsNotNone(result)


class TestVerifyKdocBalanced(unittest.TestCase):

    def test_balanced(self):
        src = "/** Doc. */\nfun foo() {}\n"
        self.assertIsNone(verify_kdoc_balanced(src))

    def test_unbalanced(self):
        src = "/** Doc.\nfun foo() {}\n"
        result = verify_kdoc_balanced(src)
        self.assertIsNotNone(result)

    def test_multiple_balanced(self):
        src = "/** A. */\nclass A {}\n/** B. */\nclass B {}\n"
        self.assertIsNone(verify_kdoc_balanced(src))


class TestStripCommentsAndWhitespace(unittest.TestCase):

    def test_preserves_code(self):
        src = "fun foo() { return 1 }\n"
        result = strip_comments_and_whitespace(src)
        self.assertIn("funfoo", result)
        self.assertIn("return1", result)

    def test_removes_kdoc(self):
        src = "/** Docs. */\nfun foo() {}\n"
        result = strip_comments_and_whitespace(src)
        self.assertNotIn("Docs", result)
        self.assertIn("funfoo", result)

    def test_preserves_string_content(self):
        src = 'val x = "hello world"\n'
        result = strip_comments_and_whitespace(src)
        self.assertIn('"hello world"', result)

    def test_removes_line_comment(self):
        src = "val x = 1 // comment\n"
        result = strip_comments_and_whitespace(src)
        self.assertNotIn("comment", result)
        self.assertIn("valx=1", result)


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

class TestDetectKdocStyle(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_default_style_no_files(self):
        style = detect_kdoc_style(self.root)
        self.assertTrue(style["uses_tags"])
        self.assertEqual(style["summary_style"], "third-person")
        self.assertEqual(style["max_width"], 100)

    def test_detects_tags(self):
        src_dir = self.root / "src" / "main"
        src_dir.mkdir(parents=True)
        (src_dir / "Foo.kt").write_text("""\
/**
 * Returns the user by ID.
 *
 * @param id The user ID.
 * @return The user object.
 */
fun getUser(id: String): User = api.fetch(id)

/**
 * Deletes the user.
 *
 * @param id The user ID.
 */
fun deleteUser(id: String) {}
""")
        style = detect_kdoc_style(self.root)
        self.assertTrue(style["uses_tags"])

    def test_detects_prose_style(self):
        src_dir = self.root / "src" / "main"
        src_dir.mkdir(parents=True)
        (src_dir / "Bar.kt").write_text("""\
/**
 * Returns the user by ID. The user is fetched from
 * the API first, then from the local database.
 */
fun getUser(id: String): User = api.fetch(id)

/**
 * Deletes the user from all storage backends.
 * This is irreversible.
 */
fun deleteUser(id: String) {}
""")
        style = detect_kdoc_style(self.root)
        self.assertFalse(style["uses_tags"])


class TestBuildStylePrompt(unittest.TestCase):

    def test_tags_style(self):
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        prompt = build_style_prompt(style)
        self.assertIn("@param", prompt)
        self.assertIn("third-person", prompt)

    def test_prose_style(self):
        style = {"uses_tags": False, "summary_style": "imperative",
                 "max_width": 80, "examples": []}
        prompt = build_style_prompt(style)
        self.assertIn("prose", prompt)
        self.assertIn("imperative", prompt)


# ---------------------------------------------------------------------------
# Update mode
# ---------------------------------------------------------------------------

class TestFindOutdatedKdoc(unittest.TestCase):

    def test_detects_extra_param(self):
        src = """\
/**
 * Gets data.
 *
 * @param id The ID.
 * @param name The name.
 * @return The data.
 */
fun getData(id: String): Data {
    return fetch(id)
}
"""
        decls = scan_declarations(src)
        outdated = find_outdated_kdoc(src, decls, uses_tags=True)
        self.assertEqual(len(outdated), 1)
        self.assertEqual(outdated[0].name, "getData")

    def test_detects_missing_return(self):
        src = """\
/**
 * Gets data.
 *
 * @param id The ID.
 */
fun getData(id: String): Data {
    return fetch(id)
}
"""
        decls = scan_declarations(src)
        outdated = find_outdated_kdoc(src, decls, uses_tags=True)
        self.assertEqual(len(outdated), 1)

    def test_detects_return_on_unit(self):
        src = """\
/**
 * Does something.
 *
 * @return The result.
 */
fun doSomething() {
    println("hi")
}
"""
        decls = scan_declarations(src)
        outdated = find_outdated_kdoc(src, decls, uses_tags=True)
        self.assertEqual(len(outdated), 1)

    def test_no_false_positives_on_correct_kdoc(self):
        src = """\
/**
 * Gets the user.
 *
 * @param id The user ID.
 * @return The user object.
 */
fun getUser(id: String): User {
    return fetch(id)
}
"""
        decls = scan_declarations(src)
        outdated = find_outdated_kdoc(src, decls, uses_tags=True)
        self.assertEqual(len(outdated), 0)

    def test_skips_when_no_tags(self):
        src = """\
/**
 * Gets data.
 */
fun getData(id: String): Data {
    return fetch(id)
}
"""
        decls = scan_declarations(src)
        outdated = find_outdated_kdoc(src, decls, uses_tags=False)
        self.assertEqual(len(outdated), 0)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

class TestResolveDocTarget(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_file_path(self):
        (self.root / "Foo.kt").write_text("class Foo {}\n")
        result = resolve_doc_target(self.root, "Foo.kt", {})
        self.assertEqual(result, ["Foo.kt"])

    def test_file_not_found(self):
        with self.assertRaises(ValueError):
            resolve_doc_target(self.root, "NoSuch.kt", {})

    def test_directory(self):
        d = self.root / "feature"
        d.mkdir()
        (d / "A.kt").write_text("class A {}\n")
        (d / "B.kt").write_text("class B {}\n")
        (d / "readme.txt").write_text("not kotlin\n")
        result = resolve_doc_target(self.root, "feature", {})
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r.endswith(".kt") for r in result))

    def test_directory_skips_test_and_build(self):
        d = self.root / "mod"
        (d / "src" / "main").mkdir(parents=True)
        (d / "src" / "test").mkdir(parents=True)
        (d / "build").mkdir(parents=True)
        (d / "src" / "main" / "A.kt").write_text("class A\n")
        (d / "src" / "test" / "ATest.kt").write_text("class ATest\n")
        (d / "build" / "Gen.kt").write_text("class Gen\n")
        result = resolve_doc_target(self.root, "mod", {})
        self.assertEqual(len(result), 1)
        self.assertIn("main/A.kt", result[0])

    def test_class_name_from_repo_map(self):
        repo_map = {
            "app/src/main/com/example/Foo.kt": {
                "defs": [
                    (1, 0, "class", "Foo", "class Foo"),
                ],
            },
        }
        result = resolve_doc_target(self.root, "Foo", repo_map)
        self.assertEqual(result, ["app/src/main/com/example/Foo.kt"])

    def test_class_not_found(self):
        with self.assertRaises(ValueError):
            resolve_doc_target(self.root, "NonExistent", {})


# ---------------------------------------------------------------------------
# Build messages
# ---------------------------------------------------------------------------

class TestBuildDocMessages(unittest.TestCase):

    def test_basic_messages(self):
        d = Declaration(
            kind="fun", name="getUser", start_line=1, decl_line=1, end_line=3,
            visibility="public", has_kdoc=False, is_override=False,
            signature="fun getUser(id: String): User", annotations=[],
        )
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        msgs = build_doc_messages("Foo.kt", "fun getUser(id: String): User {}", [d], style)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("KDoc", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("getUser", msgs[1]["content"])

    def test_update_mode_messages(self):
        d = Declaration(
            kind="fun", name="getUser", start_line=1, decl_line=3, end_line=5,
            visibility="public", has_kdoc=True, is_override=False,
            signature="fun getUser(id: String): User", annotations=[],
        )
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        msgs = build_doc_messages("Foo.kt", "source", [d], style, update_mode=True)
        self.assertIn("Update", msgs[1]["content"])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestFormatDocReport(unittest.TestCase):

    def test_basic_report(self):
        report = format_doc_report(5, 2, 3, 1, 0, True)
        self.assertIn("5 declarations", report)
        self.assertIn("2 files", report)
        self.assertIn("3 already documented", report)
        self.assertIn("1 override", report)
        self.assertIn("Code unchanged: verified", report)
        self.assertIn("/undo", report)

    def test_update_report(self):
        report = format_doc_report(0, 1, 0, 0, 3, True)
        self.assertIn("3 outdated KDoc", report)

    def test_single_file_singular(self):
        report = format_doc_report(1, 1, 0, 0, 0, True)
        self.assertIn("1 declaration ", report)
        self.assertIn("1 file", report)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestDocConfig(unittest.TestCase):

    def test_config_from_constructor(self):
        cfg = Config(doc_properties=True, doc_visibility="public",
                     doc_per_step=4, doc_max_fix=3)
        self.assertTrue(cfg.doc_properties)
        self.assertEqual(cfg.doc_visibility, "public")
        self.assertEqual(cfg.doc_per_step, 4)
        self.assertEqual(cfg.doc_max_fix, 3)

    def test_config_defaults(self):
        cfg = Config()
        self.assertFalse(cfg.doc_properties)
        self.assertEqual(cfg.doc_visibility, "internal")
        self.assertEqual(cfg.doc_per_step, 8)
        self.assertEqual(cfg.doc_max_fix, 2)


# ---------------------------------------------------------------------------
# Integration: end-to-end with FakeServer
# ---------------------------------------------------------------------------

class TestDocGenIntegration(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_doc_gen_adds_kdoc_and_undo_works(self):
        """Full flow: scan, generate, verify, undo."""
        from tests.fake_server import FakeServer
        from agent_cli.coder import Coder

        # Write a Kotlin file
        kt_file = self.root / "Foo.kt"
        original = "class Foo {\n    fun bar(x: Int): String {\n        return x.toString()\n    }\n}\n"
        kt_file.write_text(original)

        # The model's reply: a SEARCH/REPLACE block that adds KDoc
        model_reply = """\
Here is the KDoc:

Foo.kt
<<<<<<< SEARCH
class Foo {
    fun bar(x: Int): String {
=======
class Foo {
    /**
     * Converts the given integer to its string representation.
     *
     * @param x The integer to convert.
     * @return The string representation.
     */
    fun bar(x: Int): String {
>>>>>>> REPLACE
"""
        server = FakeServer([model_reply])
        server.start()
        try:
            cfg = Config(
                workdir=self.root,
                base_url=server.base_url,
                auto_approve=True,
                stream=False,
            )
            coder = Coder(cfg)
            coder.doc_gen("Foo.kt")

            # Check KDoc was added
            content = kt_file.read_text()
            self.assertIn("/**", content)
            self.assertIn("@param x", content)
            self.assertIn("fun bar", content)

            # Check undo works
            self.assertEqual(len(coder.undo_stack), 1)
            coder.undo()
            self.assertEqual(kt_file.read_text(), original)
        finally:
            server.stop()

    def test_doc_gen_reverts_on_code_change(self):
        """If the model changes code, the batch is reverted."""
        from tests.fake_server import FakeServer
        from agent_cli.coder import Coder

        kt_file = self.root / "Bar.kt"
        original = "fun greet(name: String): String {\n    return \"Hello $name\"\n}\n"
        kt_file.write_text(original)

        # Model reply that ALSO changes code (return value changed)
        model_reply = """\
Bar.kt
<<<<<<< SEARCH
fun greet(name: String): String {
    return "Hello $name"
}
=======
/**
 * Greets a person.
 *
 * @param name The name.
 * @return The greeting.
 */
fun greet(name: String): String {
    return "Hi $name"
}
>>>>>>> REPLACE
"""
        server = FakeServer([model_reply])
        server.start()
        try:
            cfg = Config(
                workdir=self.root,
                base_url=server.base_url,
                auto_approve=True,
                stream=False,
            )
            coder = Coder(cfg)
            coder.doc_gen("Bar.kt")

            # Code change should be reverted
            content = kt_file.read_text()
            self.assertEqual(content, original)
        finally:
            server.stop()

    def test_doc_gen_skips_documented_declarations(self):
        """Already-documented declarations are skipped."""
        from tests.fake_server import FakeServer
        from agent_cli.coder import Coder

        kt_file = self.root / "Baz.kt"
        kt_file.write_text("""\
/**
 * A documented class.
 */
class Baz {
    /** Already has docs. */
    fun documented() {}
}
""")
        # Server should not be called since everything is documented
        server = FakeServer([])
        server.start()
        try:
            cfg = Config(
                workdir=self.root,
                base_url=server.base_url,
                auto_approve=True,
                stream=False,
            )
            coder = Coder(cfg)
            coder.doc_gen("Baz.kt")
            # No changes, no undo
            self.assertEqual(len(coder.undo_stack), 0)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
