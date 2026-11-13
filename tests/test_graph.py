"""Code graph on the sample Android project in tests/fixtures/android_sample:
resolution + provenance, every query, the incremental cache, the CLI and
repo-map ranking."""
import io
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_cli.graph import queries
from agent_cli.graph.build import index, load_graph
from agent_cli.graph.cli import run as run_cli
from agent_cli.graph.model import AMBIGUOUS, EXTRACTED, INFERRED
from agent_cli.graph.store import JsonFactStore, MemoryFactStore

SAMPLE = Path(__file__).parent / "fixtures" / "android_sample"
APP = "com.example.app"
DATA = "com.example.data"
REPO_FILE = "core/data/src/main/java/com/example/data/UserRepository.kt"


def copy_sample() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="graph-sample-"))
    dst = tmp / "proj"
    shutil.copytree(SAMPLE, dst, ignore=shutil.ignore_patterns(".agent-cache"))
    return dst


class SampleGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = copy_sample()
        cls.g, cls.info = load_graph(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root.parent, ignore_errors=True)

    def edges(self, src, dst, kind):
        return [e for e in self.g.out.get(src, ()) if e.dst == dst and e.kind == kind]

    def assertEdge(self, src, dst, kind, prov=None):
        es = self.edges(src, dst, kind)
        self.assertTrue(es, f"missing {src} -{kind}-> {dst}; out: {self.g.out.get(src)}")
        if prov:
            self.assertEqual(es[0].prov, prov, es[0])
        return es[0]

    # ---- nodes / structure ----

    def test_nodes_for_nested_classes_companions_and_extensions(self):
        for nid in (f"{APP}.Outer.Nested.Deep.pong", f"{APP}.MainViewModel.Loader.Step.run",
                    f"{APP}.MainActivity.Companion.log", f"{DATA}.isValidId", f"{APP}.extra",
                    "res:layout/activity_main", "res:id/refresh_button", "res:string/app_name",
                    "module::app", "module::core:data"):
            self.assertIn(nid, self.g.nodes)
        self.assertEqual(self.g.nodes[f"{DATA}.isValidId"].sig, "fun String.isValidId(): Boolean")
        self.assertNotIn("res:id/ghost", self.g.nodes)  # commented out in the XML
        self.assertEqual(self.g.nodes[f"file:{REPO_FILE}"].module, ":core:data")

    def test_contains_and_imports(self):
        self.assertEdge(f"file:{REPO_FILE}", f"{DATA}.UserRepository", "contains")
        self.assertEdge(f"{DATA}.UserRepository", f"{DATA}.UserRepository.load", "contains")
        self.assertEdge("module::app", "file:app/src/main/java/com/example/app/MainActivity.kt",
                        "contains")
        self.assertEdge("file:app/src/main/java/com/example/app/MainActivity.kt",
                        f"{DATA}.isValidId", "imports", EXTRACTED)

    # ---- calls + provenance ----

    def test_calls_resolved_through_scope_are_extracted(self):
        self.assertEdge(f"{APP}.MainActivity.onCreate", f"{APP}.MainActivity.currentId", "calls", EXTRACTED)
        self.assertEdge(f"{APP}.MainActivity.render", f"{APP}.MainActivity.Companion.log", "calls", EXTRACTED)
        # inner classes see the outer class's members; nested-in-nested too
        self.assertEdge(f"{APP}.MainViewModel.Loader.start", f"{APP}.MainViewModel.refresh", "calls", EXTRACTED)
        self.assertEdge(f"{APP}.MainViewModel.Loader.Step.run", f"{APP}.MainViewModel.Loader.start",
                        "calls", EXTRACTED)
        # constructor calls point at the class
        self.assertEdge(f"{APP}.Outer.Nested.Deep.pong", f"{APP}.Outer.Nested", "calls", EXTRACTED)

    def test_calls_via_receiver_types_are_inferred(self):
        # property typed by a by-viewModels delegate, from inside a lambda
        e = self.assertEdge(f"{APP}.MainActivity.onCreate", f"{APP}.MainViewModel.refresh", "calls", INFERRED)
        self.assertEqual(e.line, 23)
        # constructor param property
        self.assertEdge(f"{APP}.MainViewModel.refresh", f"{DATA}.UserRepository.load", "calls", INFERRED)
        self.assertEdge(f"{DATA}.UserRepository.clear", f"{DATA}.UserCache.clear", "calls", INFERRED)
        # `val x = Foo()` local, and a parameter typed by an import alias
        self.assertEdge(f"{APP}.Outer.lambdas", f"{APP}.Outer.Nested.ping", "calls", INFERRED)
        self.assertEdge(f"{APP}.Outer.lambdas", f"{DATA}.UserRepository.clear", "calls", INFERRED)
        # chains: Outer().compute(), Outer.Nested().extra() (extension on a nested class)
        self.assertEdge(f"{APP}.Outer.Nested.ping", f"{APP}.Outer.compute", "calls", INFERRED)
        self.assertEdge(f"{APP}.useExtensions", f"{APP}.extra", "calls", INFERRED)
        self.assertEdge(f"{APP}.useExtensions", f"{APP}.squared", "calls", INFERRED)
        # bare call inside an extension function resolves on the receiver type
        self.assertEdge(f"{APP}.extra", f"{APP}.Outer.Nested.ping", "calls", INFERRED)
        # test code: local inferred from a constructor call
        self.assertEdge(f"{APP}.MainViewModelTest.refreshLoads", f"{APP}.MainViewModel.refresh",
                        "calls", INFERRED)

    def test_calls_from_string_templates(self):
        self.assertEdge(f"{APP}.Outer.template", f"{APP}.Outer.compute", "calls")
        e = self.assertEdge(f"{APP}.Outer.raw", f"{APP}.Outer.compute", "calls")
        self.assertEqual(e.line, 11)
        # raw-string template in MainActivity.render
        self.assertEdge(f"{APP}.MainActivity.render", f"{DATA}.displayName", "calls")
        ghost = [e for e in self.g.edges if e.kind == "calls" and "haunt" in e.dst]
        self.assertEqual(ghost, [])

    def test_ambiguous_call_keeps_candidates(self):
        es = self.g.out_edges(f"{APP}.invalidateFirst", ("calls",))
        self.assertEqual(len(es), 1)
        e = es[0]
        self.assertEqual(e.prov, AMBIGUOUS)
        self.assertEqual(set(e.cands), {f"{DATA}.UserCache.invalidate",
                                        f"{APP}.ui.UserAdapter.invalidate"})
        self.assertAlmostEqual(e.conf, 0.4, places=2)  # min(guess cap 0.4, 1/2)
        # stdlib-shaped names on an unknown receiver are not guessed
        self.assertEqual(self.g.out_edges(f"{APP}.clearFirst", ("calls",)), [])
        self.assertGreaterEqual(self.g.stats["calls"]["common_skipped"], 1)

    def test_no_guess_for_stdlib_or_external_calls(self):
        # `all {}` inside String.isValidId and `run {}` must not match project methods
        self.assertEqual(self.g.out_edges(f"{DATA}.isValidId", ("calls",)), [])
        targets = {e.dst for e in self.g.out_edges(f"{APP}.Outer.lambdas", ("calls",))}
        self.assertNotIn(f"{APP}.MainViewModel.Loader.Step.run", targets)
        # super.onCreate() goes to the framework
        targets = {e.dst for e in self.g.out_edges(f"{APP}.MainActivity.onCreate", ("calls",))}
        self.assertNotIn(f"{APP}.MainActivity.onCreate", targets)

    def test_function_typed_value_invocations_are_not_calls(self):
        self.assertGreaterEqual(self.g.stats["calls"].get("invoke", 0), 1)

    # ---- other edge kinds ----

    def test_inherits_injects_overrides(self):
        self.assertEdge(f"{DATA}.UserRepository", f"{DATA}.Repository", "inherits", EXTRACTED)
        self.assertEdge(f"{DATA}.UserRepository", f"{DATA}.UserApi", "injects", EXTRACTED)
        self.assertEdge(f"{DATA}.UserRepository", f"{DATA}.UserCache", "injects", EXTRACTED)
        self.assertEdge(f"{APP}.MainViewModel", f"{DATA}.UserRepository", "injects", EXTRACTED)
        self.assertEdge(f"{APP}.MainActivity", f"{APP}.MainViewModel", "injects")  # by viewModels()
        self.assertEdge(f"{DATA}.UserRepository.clear", f"{DATA}.Repository.clear", "overrides", EXTRACTED)

    def test_uses_type(self):
        self.assertEdge(f"{DATA}.UserRepository.load", f"{DATA}.User", "uses_type", EXTRACTED)
        self.assertEdge(f"{APP}.MainActivity.render", f"{DATA}.User", "uses_type", EXTRACTED)

    def test_gradle_depends_on(self):
        e = self.assertEdge("module::app", "module::core:data", "depends_on", EXTRACTED)
        self.assertEqual(e.file, "app/build.gradle.kts")
        self.assertNotIn("module::ghost", self.g.nodes)  # commented-out dependency

    def test_resource_references(self):
        self.assertEdge(f"{APP}.MainActivity.onCreate", "res:layout/activity_main", "references", EXTRACTED)
        self.assertEdge(f"{APP}.MainActivity.onCreate", "res:id/refresh_button", "references", EXTRACTED)
        # ViewBinding class name -> layout
        self.assertEdge(f"{APP}.MainActivity.onCreate", "res:layout/activity_main", "references")
        self.assertTrue(any(e.prov == INFERRED for e in self.edges(
            f"{APP}.MainActivity.binding", "res:layout/activity_main", "references")))
        # XML -> classes: custom view tag, tools:context, manifest component
        self.assertEdge("res:layout/activity_main", f"{APP}.ui.UserListView", "references", EXTRACTED)
        self.assertEdge("res:layout/activity_main", f"{APP}.MainActivity", "references", EXTRACTED)
        self.assertEdge("file:app/src/main/AndroidManifest.xml", f"{APP}.MainActivity", "references",
                        EXTRACTED)

    # ---- queries ----

    def test_callers(self):
        out = queries.callers(self.g, "UserRepository.load")
        self.assertIn("MainViewModel.refresh", out)
        self.assertIn("app/src/main/java/com/example/app/MainViewModel.kt:20", out)
        self.assertIn("[inferred 0.90]", out)
        # interface method: callers reach it through the override
        out = queries.callers(self.g, "Repository.clear")
        self.assertIn("via overrides", out)
        self.assertIn("MainViewModel.onCleared", out)

    def test_callers_of_class_are_constructor_calls(self):
        out = queries.callers(self.g, f"{APP}.MainViewModel")
        self.assertIn("MainViewModelTest.refreshLoads", out)

    def test_callees(self):
        out = queries.callees(self.g, "MainActivity.onCreate")
        self.assertIn("MainViewModel.refresh", out)
        self.assertIn("MainActivity.currentId", out)
        out = queries.callees(self.g, "invalidateFirst")
        self.assertIn("ambiguous", out)
        self.assertIn("2 candidates", out)
        self.assertIn("or: ", out)

    def test_path(self):
        out = queries.path(self.g, "MainActivity", "UserCache")
        self.assertIn("path MainActivity -> UserCache", out)
        self.assertIn("-injects-> UserRepository", out)
        out = queries.path(self.g, "MainActivity.onCreate", "UserRepository.load", kinds=("calls",))
        self.assertIn("2 hops", out)
        self.assertIn("MainViewModel.refresh", out)
        out = queries.path(self.g, "UserCache", "MainActivity", max_depth=4)
        self.assertIn("the other way", out)
        out = queries.path(self.g, "User", "MainActivity", max_depth=2, kinds=("calls",))
        self.assertIn("No path", out)

    def test_explain(self):
        out = queries.explain(self.g, "UserRepository")
        self.assertIn("core/data/src/main/java/com/example/data/UserRepository.kt:9-23", out)
        self.assertIn("module :core:data", out)
        for part in ("supertypes (1)", "Repository", "injects (2)", "injected into (1)",
                     "MainViewModel", "members (3): load, clear, all", "calls ("):
            self.assertIn(part, out)
        out = queries.explain(self.g, "res:layout/activity_main")
        self.assertIn("referenced by", out)
        self.assertIn("MainActivity.onCreate", out)

    def test_impact(self):
        out = queries.impact(self.g, [REPO_FILE])
        self.assertIn("app/src/main/java/com/example/app/MainViewModel.kt", out)
        self.assertIn("d2 app/src/main/java/com/example/app/MainActivity.kt", out)
        self.assertIn("tests to run (1): app/src/test/java/com/example/app/MainViewModelTest.kt", out)
        self.assertNotIn(f"  d1 {REPO_FILE}", out)
        # depth limit
        out1 = queries.impact(self.g, [REPO_FILE], depth=1)
        self.assertNotIn("MainActivity.kt", out1)
        # build file -> dependent modules
        out = queries.impact(self.g, ["core/data/build.gradle.kts"])
        self.assertIn("module :app", out)
        # unknown files are reported, suffixes are accepted
        out = queries.impact(self.g, ["nope.kt", "data/UserApi.kt"])
        self.assertIn("not in graph: nope.kt", out)
        self.assertIn("UserRepository.kt", out)

    def test_query(self):
        out = queries.query(self.g, "user cache")
        first = out.splitlines()[1]
        self.assertIn("UserCache [class]", first)
        self.assertIn("UserRepository.kt:25", first)
        self.assertIn("No symbols", queries.query(self.g, "zzzzqqq"))

    def test_budget_truncates(self):
        out = queries.explain(self.g, "UserRepository", budget=250)
        self.assertLessEqual(len(out.splitlines()[0]), 250)
        self.assertIn("more lines (raise --budget)", out)
        self.assertLess(len(out), 400)

    def test_unknown_symbol(self):
        self.assertIn("No symbol matching", queries.callers(self.g, "DoesNotExist"))

    def test_build_summary_reports_resolution_rate_and_cache(self):
        s = queries.build_summary(self.g)
        self.assertIn("call resolution:", s)
        self.assertRegex(s, r"\d+/\d+ resolved \(\d+\.\d%\)")
        self.assertIn("calls to names defined in this project", s)
        self.assertIn("cache ", s)
        self.assertIn("load ", s)
        self.assertIn("codegraph.json", s)
        c = self.g.stats["calls"]
        internal = c["internal_resolved"] + c.get("internal_unresolved", 0)
        self.assertGreaterEqual(c["internal_resolved"] / internal, 0.9)


class SourcesTest(unittest.TestCase):
    def test_gradle_parsing(self):
        from agent_cli.graph.sources import parse_gradle
        f = parse_gradle('''
pluginManagement { repositories { google { content { includeGroupByRegex("com\\.android.*") } } } }
include(":app", ":core:data")
include ':legacy'
''', settings=True)
        self.assertEqual([i[0] for i in f["includes"]], [":app", ":core:data", ":legacy"])
        f = parse_gradle('''
android { namespace = "com.x" }
dependencies {
    implementation(project(":core:data"))
    api project(path: ':lib')
    implementation(projects.featureLogin)
    /* implementation(project(":ghost")) */
}
''')
        self.assertEqual(f["namespace"], "com.x")
        self.assertEqual([d[:2] for d in f["deps"]],
                         [["implementation", ":core:data"], ["api", ":lib"],
                          ["implementation", "~:featureLogin"]])

    def test_res_xml_parsing(self):
        from agent_cli.graph.sources import classify, parse_res_xml
        self.assertEqual(classify("app/src/main/res/layout-land/main.xml"), "res")
        self.assertEqual(classify("app/src/main/AndroidManifest.xml"), "manifest")
        self.assertIsNone(classify("app/build/generated/res/layout/x.xml"))
        self.assertIsNone(classify("app/src/main/assets/data.xml"))
        f = parse_res_xml('<resources><string name="a.b">x</string><dimen name="pad">1dp</dimen>'
                          '</resources>', "app/src/main/res/values-de/strings.xml")
        self.assertEqual([v[:2] for v in f["values"]], [["string", "a_b"], ["dimen", "pad"]])
        self.assertEqual(f["rname"], "")


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.root = copy_sample()

    def tearDown(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def test_incremental_by_mtime_and_size(self):
        store = JsonFactStore(self.root / ".agent-cache" / "codegraph.json")
        _facts, info = index(self.root, store)
        self.assertEqual(info.parsed, info.files)
        self.assertTrue(store.path.exists())
        self.assertEqual((self.root / ".agent-cache" / ".gitignore").read_text(), "*\n")
        _facts, info = index(self.root, store)
        self.assertEqual(info.parsed, 0)
        f = self.root / REPO_FILE
        f.write_text(f.read_text() + "\nfun added() = 1\n")
        _facts, info = index(self.root, store)
        self.assertEqual(info.parsed, 1)
        g, _ = load_graph(self.root, store=store)
        self.assertIn("com.example.data.added", g.nodes)
        (self.root / "app/src/main/java/com/example/app/EdgeCases.kt").unlink()
        _facts, info = index(self.root, store)
        self.assertEqual(info.removed, 1)
        self.assertGreater(info.cache_bytes, 0)

    def test_stale_or_corrupt_cache_is_ignored(self):
        store = JsonFactStore(self.root / ".agent-cache" / "codegraph.json")
        store.path.parent.mkdir(parents=True)
        store.path.write_text("{not json")
        _facts, info = index(self.root, store)
        self.assertEqual(info.parsed, info.files)
        store.path.write_text('{"version": -1, "files": {}}')
        self.assertEqual(store.load(), {})

    def test_storage_is_swappable(self):
        store = MemoryFactStore()
        g, info = load_graph(self.root, store=store)
        self.assertEqual(info.parsed, info.files)
        self.assertIn(REPO_FILE, store.entries)
        g2, info2 = load_graph(self.root, store=store)
        self.assertEqual(info2.parsed, 0)
        self.assertEqual(len(g.edges), len(g2.edges))
        self.assertFalse((self.root / ".agent-cache").exists())


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = copy_sample()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root.parent, ignore_errors=True)

    def cli(self, *args):
        buf = io.StringIO()
        code = run_cli(list(args) + ["--dir", str(self.root)], out=buf)
        return code, buf.getvalue()

    def test_build(self):
        code, out = self.cli("build")
        self.assertEqual(code, 0)
        self.assertIn("call resolution:", out)
        self.assertIn("cache ", out)
        code, out = self.cli("build")
        self.assertIn("(0 parsed,", out)
        code, out = self.cli("build", "--force")
        self.assertNotIn("(0 parsed,", out)

    def test_each_command(self):
        cases = [
            (("callers", "UserRepository.load"), "MainViewModel.refresh"),
            (("callees", "MainViewModel.refresh"), "UserRepository.load"),
            (("path", "MainActivity", "UserCache"), "hops"),
            (("path", "MainActivity.onCreate", "UserCache.clear", "--kinds", "calls"), "No path"),
            (("explain", "MainViewModel"), "injected into"),
            (("impact", REPO_FILE), "tests to run"),
            (("query", "view", "model"), "MainViewModel [class]"),
        ]
        for args, expect in cases:
            code, out = self.cli(*args)
            self.assertEqual(code, 0, args)
            self.assertIn(expect, out, args)

    def test_budget_and_bad_kind(self):
        code, out = self.cli("explain", "UserRepository", "--budget", "300")
        self.assertLess(len(out), 450)
        code, out = self.cli("path", "A", "B", "--kinds", "bogus")
        self.assertEqual(code, 2)
        self.assertIn("Unknown edge kind", out)

    def test_absolute_impact_path(self):
        code, out = self.cli("impact", str(self.root / REPO_FILE))
        self.assertIn("MainViewModel.kt", out)

    def test_entry_point_dispatches_graph(self):
        from agent_cli import main as main_mod
        buf = io.StringIO()
        with patch("sys.argv", ["agent", "graph", "build", "--dir", str(self.root)]), \
                patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as cm:
                main_mod.app()
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("call resolution:", buf.getvalue())


class RepoMapRankingTest(unittest.TestCase):
    def setUp(self):
        self.root = copy_sample()

    def tearDown(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def test_graph_ranking_and_fallback(self):
        from agent_cli.repomap import RepoMap
        rm = RepoMap(self.root)
        rm.build()
        self.assertEqual(rm.rank_source, "graph")
        ranks = rm._file_ranks
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=3)
        # UserRepository (injected + called) outranks the leaf activity
        self.assertGreater(ranks[REPO_FILE], ranks["app/src/main/java/com/example/app/MainActivity.kt"])
        self.assertTrue((self.root / ".agent-cache" / "codegraph.json").exists())

        with patch.dict(os.environ, {"AGENT_GRAPH_RANK": "0"}):
            rm2 = RepoMap(self.root)
            rm2.build()
        self.assertEqual(rm2.rank_source, "refs")

        with patch("agent_cli.graph.rank.graph_file_ranks", side_effect=RuntimeError("boom")):
            rm3 = RepoMap(self.root)
            rm3.build()
        self.assertEqual(rm3.rank_source, "refs")
        self.assertTrue(rm3.render())

    def test_non_kotlin_repo_keeps_old_ranking(self):
        from agent_cli.repomap import RepoMap
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "a.py").write_text("def f():\n    return g()\n")
            (tmp / "b.py").write_text("def g():\n    return 1\n")
            rm = RepoMap(tmp)
            rm.build()
            self.assertEqual(rm.rank_source, "refs")
            self.assertFalse((tmp / ".agent-cache" / "codegraph.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PerformanceSmokeTest(unittest.TestCase):
    def test_large_generated_project_builds_quickly(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "app" / "src" / "main" / "java" / "p"
            src.mkdir(parents=True)
            for i in range(150):
                body = "\n".join(
                    f"    fun m{j}(x: Int): Int = dep.m{(j + 1) % 20}(x) + helper{j}(x)\n"
                    f"    private fun helper{j}(x: Int) = listOf(x).map {{ it + {j} }}.sum()"
                    for j in range(20))
                (src / f"C{i}.kt").write_text(
                    f"package p\n\nclass C{i}(private val dep: C{(i + 1) % 150}) {{\n{body}\n}}\n")
            t0 = time.perf_counter()
            g, info = load_graph(tmp)
            elapsed = time.perf_counter() - t0
            self.assertEqual(info.files, 150)
            self.assertLess(elapsed, 20.0)
            st = g.stats["calls"]
            self.assertEqual(st["internal_resolved"], 150 * 20 * 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
