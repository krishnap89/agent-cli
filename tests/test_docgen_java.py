"""Tests for /doc on Java sources (Javadoc)."""
import tempfile
import unittest
from pathlib import Path

from agent_cli.config import Config
from agent_cli.docgen import (
    scan_declarations, filter_declarations, detect_kdoc_style,
    build_style_prompt, build_doc_messages, find_outdated_kdoc,
    verify_code_unchanged, resolve_doc_target, language_of,
    _java_param_names, _java_returns_void,
)


SAMPLE_JAVA = """\
package com.example;

import java.util.List;
import java.util.Map;

/**
 * Already documented.
 */
public class UserRepository implements Repository {

    private static final String TAG = "UserRepository";
    public static final int MAX_USERS = 100;
    final Map<String, List<User>> cache = new HashMap<>();

    private final UserApi api;

    public UserRepository(UserApi api) {
        this.api = api;
    }

    UserRepository() {
        this(null);
    }

    public User findUser(String id, boolean refresh) {
        if (refresh) {
            String s = api.fetch(id);
            return parse(s);
        }
        return cache.get(id).get(0);
    }

    @Override
    public void clear() {
        cache.clear();
    }

    @Override public String toString() { return TAG; }

    protected <T extends User> List<T> filter(
            List<T> users,
            Predicate<T> keep) {
        Runnable r = () -> {
            int count = 0;
        };
        return users;
    }

    void reset() {
        new Thread(new Runnable() {
            public void run() {
                clear();
            }
        }).start();
    }

    private User parse(String raw) {
        return null;
    }

    public static class Builder {
        public Builder withApi(UserApi api) {
            return this;
        }
    }

    enum State {
        IDLE("idle"),
        LOADING("loading");

        private final String label;

        State(String label) {
            this.label = label;
        }

        public String label() {
            return label;
        }
    }
}

interface Repository {
    void clear();

    default int size() {
        return 0;
    }
}

@interface Cached {
    int seconds() default 60;
}

record Point(int x, int y) {
    public double length() {
        return Math.sqrt(x * x + y * y);
    }
}
"""


def _by_name(decls):
    return {(d.kind, d.name): d for d in decls}


class TestLanguageOf(unittest.TestCase):

    def test_extension(self):
        self.assertEqual(language_of("app/src/main/java/Foo.java"), "java")
        self.assertEqual(language_of("app/src/main/kotlin/Foo.kt"), "kotlin")
        self.assertEqual(language_of("build.gradle.kts"), "kotlin")


class TestScanJava(unittest.TestCase):

    def setUp(self):
        self.decls = scan_declarations(SAMPLE_JAVA, lang="java")
        self.by = _by_name(self.decls)

    def test_types(self):
        for key in [("class", "UserRepository"), ("class", "Builder"), ("enum", "State"),
                    ("interface", "Repository"), ("annotation", "Cached"),
                    ("record", "Point")]:
            self.assertIn(key, self.by, key)

    def test_existing_javadoc_detected(self):
        self.assertTrue(self.by[("class", "UserRepository")].has_kdoc)
        self.assertFalse(self.by[("method", "findUser")].has_kdoc)

    def test_methods_and_constructors(self):
        for key in [("method", "findUser"), ("method", "clear"), ("method", "toString"),
                    ("method", "filter"), ("method", "reset"), ("method", "parse"),
                    ("method", "withApi"), ("method", "label"), ("method", "size"),
                    ("method", "seconds"), ("method", "length"),
                    ("constructor", "UserRepository"), ("constructor", "State")]:
            self.assertIn(key, self.by, key)

    def test_fields(self):
        for name in ("TAG", "MAX_USERS", "cache", "api", "label"):
            self.assertIn(("field", name), self.by, name)

    def test_no_statements_or_locals(self):
        """Code inside method bodies, lambdas and anonymous classes is skipped."""
        names = {d.name for d in self.decls}
        for local in ("s", "count", "r", "run", "fetch", "IDLE", "LOADING", "get"):
            self.assertNotIn(local, names, local)
        # clear() appears in the class and the interface only (not the call sites)
        self.assertEqual(sum(1 for d in self.decls if d.name == "clear"), 2)

    def test_two_constructors(self):
        ctors = [d for d in self.decls if d.kind == "constructor" and d.name == "UserRepository"]
        self.assertEqual(len(ctors), 2)
        self.assertEqual({c.visibility for c in ctors}, {"public", "package"})

    def test_visibility(self):
        self.assertEqual(self.by[("method", "findUser")].visibility, "public")
        self.assertEqual(self.by[("method", "filter")].visibility, "protected")
        self.assertEqual(self.by[("method", "reset")].visibility, "package")
        self.assertEqual(self.by[("method", "parse")].visibility, "private")
        self.assertEqual(self.by[("record", "Point")].visibility, "package")

    def test_interface_members_are_public(self):
        iface_clear = [d for d in self.decls if d.name == "clear" and d.visibility == "public"
                       and not d.is_override]
        self.assertEqual(len(iface_clear), 1)
        self.assertEqual(self.by[("method", "size")].visibility, "public")
        self.assertEqual(self.by[("method", "seconds")].visibility, "public")

    def test_override_from_annotation(self):
        self.assertTrue(self.by[("method", "toString")].is_override)  # same-line @Override
        overridden_clear = [d for d in self.decls if d.name == "clear" and d.is_override]
        self.assertEqual(len(overridden_clear), 1)  # @Override on the line above
        self.assertIn("@Override", overridden_clear[0].annotations)
        self.assertFalse(self.by[("method", "findUser")].is_override)

    def test_multiline_signature(self):
        sig = self.by[("method", "filter")].signature
        self.assertIn("Predicate<T> keep", sig)

    def test_decl_line_points_at_declaration(self):
        lines = SAMPLE_JAVA.splitlines()
        d = self.by[("method", "findUser")]
        self.assertIn("findUser(", lines[d.decl_line - 1])

    def test_strings_and_comments_ignored(self):
        src = (
            "class A {\n"
            "    // public void commented() {}\n"
            "    String s = \"public void inString() {\";\n"
            "    /* public void blockCommented() {} */\n"
            "    public void real() {}\n"
            "}\n"
        )
        names = {d.name for d in scan_declarations(src, lang="java")}
        self.assertIn("real", names)
        self.assertNotIn("commented", names)
        self.assertNotIn("inString", names)
        self.assertNotIn("blockCommented", names)

    def test_android_annotation_patterns(self):
        src = (
            "public final class MainViewModel extends ViewModel {\n"
            "    @Inject\n"
            "    public MainViewModel(@NonNull Repo repo,\n"
            "                         SavedStateHandle h) { }\n"
            "    public @NonNull LiveData<List<Item>> getItems() { return items; }\n"
            "    @Nullable public String getName() { return null; }\n"
            "    @SuppressWarnings(\"unchecked\") public static <K, V> Map<K, V> cast(Object o) { return null; }\n"
            "    public abstract static class Base<T> implements Comparable<T> {\n"
            "        protected abstract void onBind(T item);\n"
            "    }\n"
            "}\n"
        )
        by = _by_name(scan_declarations(src, lang="java"))
        self.assertEqual(by[("constructor", "MainViewModel")].annotations, ["@Inject"])
        self.assertIn(("method", "getItems"), by)  # annotation after the modifier
        self.assertEqual(by[("method", "getName")].annotations, ["@Nullable"])
        # A one-line member above is not mistaken for an annotation of the next one
        self.assertEqual(len(by[("method", "cast")].annotations), 1)
        self.assertEqual(by[("class", "Base")].annotations, [])
        self.assertEqual(by[("class", "Base")].start_line, 8)
        self.assertIn(("method", "onBind"), by)

    def test_kotlin_default_unchanged(self):
        """Without lang, the Kotlin scanner is used (Java methods aren't found)."""
        names = {d.name for d in scan_declarations(SAMPLE_JAVA)}
        self.assertNotIn("findUser", names)


class TestFilterJava(unittest.TestCase):

    def setUp(self):
        self.decls = scan_declarations(SAMPLE_JAVA, lang="java")
        self.rel = "app/src/main/java/com/example/UserRepository.java"

    def _names(self, **kw):
        return {(d.kind, d.name) for d in filter_declarations(
            self.decls, self.rel, lang="java", **kw)}

    def test_internal_includes_package_private(self):
        names = self._names(visibility_filter="internal")
        self.assertIn(("method", "reset"), names)
        self.assertIn(("method", "filter"), names)
        self.assertIn(("record", "Point"), names)
        self.assertNotIn(("method", "parse"), names)  # private
        self.assertNotIn(("method", "toString"), names)  # @Override
        self.assertNotIn(("class", "UserRepository"), names)  # already documented

    def test_public_keeps_public_and_protected(self):
        names = self._names(visibility_filter="public")
        self.assertIn(("method", "findUser"), names)
        self.assertIn(("method", "filter"), names)
        self.assertNotIn(("method", "reset"), names)

    def test_fields_only_with_properties(self):
        self.assertNotIn(("field", "MAX_USERS"), self._names())
        with_props = self._names(include_properties=True)
        self.assertIn(("field", "MAX_USERS"), with_props)
        self.assertNotIn(("field", "TAG"), with_props)  # private

    def test_test_sources_skipped(self):
        self.assertEqual(filter_declarations(
            self.decls, "app/src/test/java/FooTest.java", lang="java"), [])


class TestJavaSignatureHelpers(unittest.TestCase):

    def test_param_names(self):
        self.assertEqual(_java_param_names("public void f(String id, boolean refresh) {"),
                         {"id", "refresh"})
        self.assertEqual(_java_param_names("void f(Map<String, List<Integer>> m, int... xs)"),
                         {"m", "xs"})
        self.assertEqual(_java_param_names("void f(@NonNull final String s, int[] arr)"),
                         {"s", "arr"})
        self.assertEqual(_java_param_names("void f()"), set())

    def test_returns_void(self):
        self.assertTrue(_java_returns_void("public void clear() {", "clear"))
        self.assertFalse(_java_returns_void("public String voidish() {", "voidish"))
        self.assertFalse(_java_returns_void("public <T> List<T> f(List<T> x)", "f"))


class TestOutdatedJavadoc(unittest.TestCase):

    def _outdated(self, src):
        decls = scan_declarations(src, lang="java")
        return {d.name for d in find_outdated_kdoc(src, decls, uses_tags=True, lang="java")}

    def test_up_to_date(self):
        src = (
            "class A {\n"
            "    /**\n     * Finds.\n     * @param id the id\n     * @return the user\n     */\n"
            "    public User find(String id) { return null; }\n"
            "    /**\n     * Clears.\n     */\n"
            "    public void clear() {}\n"
            "    /**\n     * Creates.\n     * @param x the x\n     */\n"
            "    A(int x) {}\n"
            "}\n"
        )
        self.assertEqual(self._outdated(src), set())

    def test_renamed_param(self):
        src = ("class A {\n    /**\n     * Finds.\n     * @param oldName x\n     * @return u\n     */\n"
               "    public User find(String id) { return null; }\n}\n")
        self.assertEqual(self._outdated(src), {"find"})

    def test_return_on_void_and_constructor(self):
        src = ("class A {\n    /**\n     * Clears.\n     * @return nothing\n     */\n"
               "    public void clear() {}\n"
               "    /**\n     * Creates.\n     * @return nope\n     */\n"
               "    A() {}\n}\n")
        self.assertEqual(self._outdated(src), {"clear", "A"})

    def test_missing_return(self):
        src = ("class A {\n    /**\n     * Counts.\n     */\n"
               "    public int count() { return 0; }\n}\n")
        self.assertEqual(self._outdated(src), {"count"})

    def test_type_param_tag_is_not_extra(self):
        src = ("class A {\n    /**\n     * Picks.\n     * @param <T> the type\n"
               "     * @param xs the items\n     * @return first\n     */\n"
               "    public <T> T first(List<T> xs) { return null; }\n}\n")
        self.assertEqual(self._outdated(src), set())


class TestJavaPrompts(unittest.TestCase):

    def test_messages_use_javadoc(self):
        decls = scan_declarations(SAMPLE_JAVA, lang="java")
        target = [d for d in decls if d.name == "findUser"]
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        msgs = build_doc_messages("Foo.java", SAMPLE_JAVA, target, style, lang="java")
        self.assertIn("Javadoc", msgs[0]["content"])
        self.assertNotIn("KDoc", msgs[0]["content"])
        self.assertIn("Add Javadoc to these declarations", msgs[1]["content"])
        self.assertIn("```java", msgs[1]["content"])
        self.assertIn("non-void methods", msgs[1]["content"])

    def test_update_prompt_mentions_void(self):
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        msgs = build_doc_messages("Foo.java", SAMPLE_JAVA, [], style,
                                  update_mode=True, lang="java")
        self.assertIn("void methods and constructors", msgs[1]["content"])

    def test_kotlin_prompt_unchanged(self):
        style = {"uses_tags": True, "summary_style": "third-person",
                 "max_width": 100, "examples": []}
        self.assertIn("non-Unit functions", build_style_prompt(style))
        msgs = build_doc_messages("Foo.kt", "class Foo", [], style)
        self.assertIn("KDoc", msgs[0]["content"])
        self.assertIn("```kotlin", msgs[1]["content"])


class TestJavaStyleAndTargets(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        src = self.root / "app" / "src" / "main" / "java"
        src.mkdir(parents=True)
        (src / "A.java").write_text(
            "class A {\n    /**\n     * Returns the answer to everything here.\n"
            "     * @return the answer\n     */\n    int answer() { return 42; }\n}\n")
        (src / "B.kt").write_text("/**\n * Kotlin prose doc that is long enough.\n */\nclass B\n")
        (self.root / "app" / "src" / "test" / "java").mkdir(parents=True)
        (self.root / "app" / "src" / "test" / "java" / "ATest.java").write_text("class ATest {}\n")

    def tearDown(self):
        self.td.cleanup()

    def test_style_sampled_from_java_only(self):
        style = detect_kdoc_style(self.root, "app", lang="java")
        self.assertTrue(style["uses_tags"])
        self.assertTrue(all("answer" in e for e in style["examples"]))

    def test_resolve_java_file(self):
        rel = "app/src/main/java/A.java"
        self.assertEqual(resolve_doc_target(self.root, rel, {}), [rel])

    def test_resolve_folder_includes_java_and_kotlin_not_tests(self):
        files = resolve_doc_target(self.root, "app", {})
        self.assertEqual(files, ["app/src/main/java/A.java", "app/src/main/java/B.kt"])

    def test_resolve_java_class_from_repo_map(self):
        repo_map = {"app/src/main/java/A.java": {"defs": [(1, 0, "class", "A")]}}
        self.assertEqual(resolve_doc_target(self.root, "A", repo_map),
                         ["app/src/main/java/A.java"])


class TestJavaGuardrail(unittest.TestCase):

    def test_javadoc_only_change_is_accepted(self):
        old = "class A {\n    public void f(String s) { g(\"// not a comment\"); }\n}\n"
        new = ("class A {\n    /**\n     * Does f.\n     * @param s input\n     */\n"
               "    public void f(String s) { g(\"// not a comment\"); }\n}\n")
        self.assertIsNone(verify_code_unchanged(old, new))

    def test_code_change_is_caught(self):
        old = "class A {\n    public void f(String s) { g(s); }\n}\n"
        new = "class A {\n    /** Does f. */\n    public void f(String t) { g(t); }\n}\n"
        self.assertIsNotNone(verify_code_unchanged(old, new))


class TestJavaDocGenIntegration(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_doc_gen_adds_javadoc_and_undo_works(self):
        from tests.fake_server import FakeServer
        from agent_cli.coder import Coder

        java_file = self.root / "Foo.java"
        original = ("public class Foo {\n"
                    "    public String bar(int x) {\n"
                    "        return String.valueOf(x);\n"
                    "    }\n"
                    "}\n")
        java_file.write_text(original)

        model_reply = """\
Foo.java
<<<<<<< SEARCH
public class Foo {
=======
/**
 * Formats numbers.
 */
public class Foo {
>>>>>>> REPLACE

Foo.java
<<<<<<< SEARCH
    public String bar(int x) {
=======
    /**
     * Returns the decimal string for the given number.
     *
     * @param x the number to format
     * @return the decimal string
     */
    public String bar(int x) {
>>>>>>> REPLACE
"""
        server = FakeServer([model_reply])
        server.start()
        try:
            cfg = Config(workdir=self.root, base_url=server.base_url,
                         auto_approve=True, stream=False)
            coder = Coder(cfg)
            coder.doc_gen("Foo.java")

            content = java_file.read_text()
            self.assertIn(" * Formats numbers.", content)
            self.assertIn("     * @param x the number to format", content)
            self.assertIn("public String bar(int x) {", content)

            sent = server.requests[0]["messages"]
            self.assertIn("Javadoc", sent[0]["content"])
            self.assertIn("```java", sent[1]["content"])

            self.assertEqual(len(coder.undo_stack), 1)
            coder.undo()
            self.assertEqual(java_file.read_text(), original)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
