"""Tests for Java extraction in the repo map (tree-sitter-java, with regex fallback)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_cli import repomap
from agent_cli.repomap import RepoMap, extract


SAMPLE = """\
package com.example;

import java.util.List;
import com.example.net.UserApi;

@Singleton
public class UserRepository implements Repository {
    private static final String TAG = "UserRepository";
    private final UserApi api;
    private String first, second = "x";

    public UserRepository(UserApi api) {
        this.api = api;
    }

    @Override
    public void clear() {
        cache.clear();
        new Thread(new Runnable() {
            public void run() { clear(); }
        }).start();
    }

    protected <T extends User> List<T> filter(
            List<T> users) {
        return users;
    }

    enum State {
        IDLE("idle"),
        LOADING("loading");

        State(String label) {}

        public String label() { return ""; }
    }
}

interface Repository {
    void clear();
}

@interface Cached {
    int seconds() default 60;
}

record Point(int x, int y) {
    public double length() { return 0; }
}
"""


def _names(defs):
    return [(d[2], d[3]) for d in defs]


@unittest.skipIf(repomap._java_parser is None, "tree-sitter-java not installed")
class TestJavaTreeSitterDefs(unittest.TestCase):

    def setUp(self):
        self.defs = repomap._java_defs(SAMPLE)
        self.names = _names(self.defs)

    def test_types_are_class(self):
        for name in ("UserRepository", "State", "Repository", "Cached", "Point"):
            self.assertIn(("class", name), self.names)

    def test_members_are_def(self):
        for name in ("clear", "filter", "label", "seconds", "length", "api", "first", "second"):
            self.assertIn(("def", name), self.names, name)
        # constructors: the class's and the enum's
        self.assertIn(("def", "UserRepository"), self.names)
        self.assertIn(("def", "State"), self.names)

    def test_no_calls_constants_or_anonymous_members(self):
        names = [n for _, n in self.names]
        self.assertNotIn("IDLE", names)      # enum constants
        self.assertNotIn("TAG", names)       # ALL_CAPS constant field
        self.assertNotIn("run", names)       # anonymous class member
        self.assertNotIn("start", names)     # call
        # clear(): declared in the class and the interface, never the call sites
        self.assertEqual(names.count("clear"), 2)

    def test_depth_follows_nesting(self):
        by = {(d[2], d[3], d[0]): d[1] for d in self.defs}
        depth = {n: d for (k, n, _), d in by.items() if k == "class"}
        self.assertEqual(depth["UserRepository"], 0)
        self.assertEqual(depth["State"], 1)
        label = [d for d in self.defs if d[3] == "label"][0]
        self.assertEqual(label[1], 2)

    def test_signatures_stop_at_body_and_initializer(self):
        sigs = {(d[2], d[3]): d[4] for d in self.defs}
        self.assertEqual(sigs[("def", "filter")],
                         "protected <T extends User> List<T> filter( List<T> users)")
        self.assertEqual(sigs[("class", "UserRepository")],
                         "@Singleton public class UserRepository implements Repository")
        self.assertEqual(sigs[("def", "UserRepository")], "public UserRepository(UserApi api)")
        self.assertEqual(sigs[("def", "second")], "private String first, second")
        self.assertEqual(sigs[("def", "seconds")], "int seconds() default 60")

    def test_line_numbers(self):
        lines = SAMPLE.splitlines()
        length = [d for d in self.defs if d[3] == "length"][0]
        self.assertIn("length()", lines[length[0] - 1])

    def test_refs(self):
        refs = repomap._java_refs(SAMPLE)
        for name in ("UserApi", "List", "Repository", "User", "Thread", "Runnable"):
            self.assertIn(name, refs)

    def test_extract_uses_tree_sitter_for_java(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "UserRepository.java"
            p.write_text(SAMPLE)
            info = extract(p)
        self.assertIn(("class", "UserRepository"), _names(info["defs"]))
        self.assertNotIn("IDLE", [d[3] for d in info["defs"]])


class TestJavaRegexFallback(unittest.TestCase):

    def test_falls_back_without_grammar(self):
        with patch.object(repomap, "_java_parser", None):
            defs = repomap._java_defs(SAMPLE)
            refs = repomap._java_refs(SAMPLE)
        self.assertEqual(defs, repomap._regex_defs(SAMPLE, ".java"))
        self.assertIn(("class", "UserRepository"), _names(defs))
        self.assertIn("UserApi", refs)


class TestCacheTracksParsers(unittest.TestCase):

    def test_parser_change_reindexes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A.java").write_text("class A { void f() {} }\n")
            rm = RepoMap(root)
            self.assertEqual(rm.build()[1], 1)       # first build parses
            self.assertEqual(RepoMap(root).build()[1], 0)  # cached

            data = json.loads(rm.cache_path.read_text())
            self.assertEqual(data["parsers"], repomap._parser_tag())

            # A different set of grammars (e.g. one just installed) re-parses.
            with patch.object(repomap, "_parser_tag", return_value=["something-else"]):
                self.assertEqual(RepoMap(root).build()[1], 1)

    def test_old_cache_without_parsers_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A.java").write_text("class A {}\n")
            rm = RepoMap(root)
            rm.build()
            data = json.loads(rm.cache_path.read_text())
            del data["parsers"]
            rm.cache_path.write_text(json.dumps(data))
            expected = 0 if repomap._parser_tag() == [] else 1
            self.assertEqual(RepoMap(root).build()[1], expected)


if __name__ == "__main__":
    unittest.main()
