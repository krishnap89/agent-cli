"""Swift / iOS support in the code graph: lexer + parser edge cases, iOS
project files, and resolution + every query on tests/fixtures/ios_sample
(an Xcode app target + test target using a local Swift package)."""
import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agent_cli.graph import queries
from agent_cli.graph.build import load_graph
from agent_cli.graph.cli import run as run_cli
from agent_cli.graph.ios_sources import (asset_kind, parse_ib, parse_openstep_plist,
                                         parse_package_swift, parse_pbxproj, parse_strings,
                                         parse_xcstrings)
from agent_cli.graph.kotlin_lexer import ID, STR
from agent_cli.graph.model import AMBIGUOUS, EXTRACTED, INFERRED
from agent_cli.graph.sources import classify
from agent_cli.graph.store import MemoryFactStore
from agent_cli.graph.swift_lexer import tokenize_swift
from agent_cli.graph.swift_parser import parse_swift

SAMPLE = Path(__file__).parent / "fixtures" / "ios_sample"
REPO_FILE = "Packages/CoreKit/Sources/CoreKit/UserRepository.swift"


def ids(toks):
    return [t[1] for t in toks if t[0] == ID]


def decl(facts, name, kind=None):
    for d in facts["decls"]:
        if d["name"] == name and (kind is None or d["kind"] == kind):
            return d
    raise AssertionError(f"no decl {name}")


def calls_named(facts, name):
    return [c for c in facts["calls"] if c[1] == name]


class SwiftLexerTest(unittest.TestCase):
    def test_interpolation_is_code_string_text_is_not(self):
        toks = tokenize_swift('let s = "a { b } \\(f(x)) c"')
        self.assertIn("f", ids(toks))
        self.assertNotIn("b", ids(toks))

    def test_raw_strings(self):
        toks = tokenize_swift('let r = #"no \\(notCode()) yes \\#(code())"#; let z = 1')
        self.assertIn("code", ids(toks))
        self.assertNotIn("notCode", ids(toks))
        self.assertEqual(ids(toks)[-2:], ["let", "z"])
        toks = tokenize_swift('let r = ##"a "# still "## + b')
        self.assertEqual(ids(toks), ["let", "r", "b"])

    def test_multiline_and_multiline_raw_strings(self):
        src = 'let m = """\n  func ghost() { haunt() }\n  \\(real())\n  """\nlet after = 1'
        toks = tokenize_swift(src)
        self.assertIn("real", ids(toks))
        self.assertNotIn("haunt", ids(toks))
        self.assertEqual(toks[-1][2], 5)
        toks = tokenize_swift('let m = #"""\n "\\(no())" \\#(yes())\n"""#')
        self.assertEqual(ids(toks), ["let", "m", "yes"])

    def test_plain_literal_text_is_kept_for_resource_names(self):
        toks = tokenize_swift('UIImage(named: "logo")')
        self.assertEqual([t[1] for t in toks if t[0] == STR], ["logo"])
        toks = tokenize_swift('Text("hi \\(name)")')
        self.assertEqual([t[1] for t in toks if t[0] == STR], [""])

    def test_nested_comments_directives_and_specials(self):
        src = "a /* x /* y */ z */ b\n#if DEBUG\nc()\n#elseif os(iOS)\nd()\n#endif\n$0 `class` x?.y"
        toks = tokenize_swift(src)
        self.assertEqual(ids(toks), ["a", "b", "c", "d", "$0", "class", "x", "y"])
        self.assertIn("?.", [t[1] for t in toks])


class SwiftParserTest(unittest.TestCase):
    def test_imports(self):
        f = parse_swift("import UIKit\n@testable import App\nimport struct CoreKit.User\n")
        self.assertEqual([i[:3] for i in f["imports"]],
                         [["UIKit", None, True], ["App", None, True], ["CoreKit.User", None, False]])

    def test_types_supertypes_generics_and_nesting(self):
        f = parse_swift("""
public final class A<T: Equatable, U>: Base, P1 & P2 where T: Hashable {
    struct Inner { enum Deep { case a } }
}
protocol P: AnyObject { associatedtype Item; func f(_ x: Item) }
actor Store {}
""")
        a = decl(f, "A")
        self.assertEqual(a["tparams"], ["T", "U"])
        self.assertEqual([s[0] for s in a["supers"]], ["Base", "P1", "P2"])
        self.assertIn("final", a["mods"])
        self.assertEqual(decl(f, "Deep")["kind"], "enum")
        self.assertEqual(f["decls"][decl(f, "Deep")["parent"]]["name"], "Inner")
        self.assertEqual(decl(f, "P")["kind"], "protocol")
        self.assertEqual(decl(f, "Item")["kind"], "typealias")
        self.assertEqual(decl(f, "Store")["kind"], "actor")

    def test_functions_labels_defaults_variadics_effects(self):
        f = parse_swift("""
struct S {
    func load(from url: URL, _ id: Int = 0, tags: String..., completion: @escaping (Result<Data, Error>) -> Void) async throws -> [User] { [] }
    static func == (lhs: S, rhs: S) -> Bool { true }
    init?(json: [String: Any]) { }
    subscript(index: Int) -> Int { 0 }
    deinit { }
}
""")
        p = decl(f, "load")["params"]
        self.assertEqual([(x[6], x[0]) for x in p],
                         [("from", "url"), ("_", "id"), ("tags", "tags"), ("completion", "completion")])
        self.assertTrue(p[1][3])
        self.assertTrue(p[2][5])
        self.assertIn("->", p[3][1])
        self.assertEqual(decl(f, "load")["ret"], "[User]")
        self.assertEqual(decl(f, "==")["params"][0][0], "lhs")
        self.assertEqual(decl(f, "init", "init")["params"][0][6], "json")
        self.assertEqual(decl(f, "subscript")["ret"], "Int")
        decl(f, "deinit")

    def test_properties_wrappers_computed_and_observers(self):
        f = parse_swift("""
final class VM {
    @Published private(set) var users: [User] = []
    @Injected(\\.repo) private var repository: Repo
    static let shared = VM()
    lazy var formatter: DateFormatter = { DateFormatter() }()
    var count: Int { users.count }
    var name = "" { didSet { save() } }
    let a = 1, b = Builder()
    var body: some View { Text("x") }
}
""")
        self.assertEqual(decl(f, "users")["annos"], ["Published"])
        self.assertIn("private", decl(f, "users")["mods"])
        self.assertEqual(decl(f, "repository")["annos"], ["Injected"])
        self.assertEqual(decl(f, "shared")["itype"], "VM")
        self.assertIn("static", decl(f, "shared")["mods"])
        self.assertEqual(decl(f, "formatter")["type"], "DateFormatter")
        self.assertEqual(decl(f, "b")["itype"], "Builder")
        self.assertEqual(decl(f, "body")["type"], "some View")
        owner = {c[1]: f["decls"][c[0]]["name"] for c in f["calls"]}
        self.assertEqual(owner["save"], "name")
        self.assertEqual(owner["Text"], "body")

    def test_enum_cases(self):
        f = parse_swift("enum E { case a, b(Int), c = 3\n indirect case d(E)\n var x: Int { 0 } }")
        self.assertEqual([d["name"] for d in f["decls"] if d["kind"] == "entry"], ["a", "b", "c", "d"])

    def test_extensions(self):
        f = parse_swift("public extension String { func trimmed() -> String { self } }\n"
                        "extension Outer.Inner: Equatable where T == Int { var x: Int { 1 } }\n")
        ext = [d for d in f["decls"] if d["kind"] == "extension"]
        self.assertEqual([e["name"] for e in ext], ["String", "Outer.Inner"])
        self.assertEqual(ext[1]["supers"][0][0], "Equatable")
        self.assertEqual(f["decls"][decl(f, "trimmed")["parent"]]["kind"], "extension")

    def test_closures_trailing_closures_and_labels(self):
        f = parse_swift("""
func f(vm: VM) {
    items.forEach { item in render(item) }
    Task { [weak self] in await self?.reload() }
    Button { tap() } label: { Text("x") }
    vm.select(user)
    vm.refresh(id: currentID(), force: true)
    let t: (Int) -> Int = { $0 * 2 }
    super.viewDidLoad()
    self.init(x: 1)
}
""")
        self.assertEqual(calls_named(f, "forEach")[0][6], ["*"])
        self.assertEqual(calls_named(f, "Button")[0][6], ["*", "*"])
        self.assertEqual(calls_named(f, "select")[0][6], ["_"])
        self.assertEqual(calls_named(f, "refresh")[0][6], ["id", "force"])
        self.assertEqual(calls_named(f, "reload")[0][2], [["this", 0]])
        self.assertEqual(calls_named(f, "viewDidLoad")[0][2], [["super", 0]])
        self.assertEqual(calls_named(f, "init")[0][2], [["this", 0]])
        for n in ("render", "tap", "Text", "currentID"):
            self.assertTrue(calls_named(f, n), n)
        self.assertFalse(calls_named(f, "item"))
        self.assertEqual(decl(f, "f")["locals"]["t"], ["(Int) -> Int", "decl"])

    def test_statement_conditions_are_not_trailing_closures(self):
        f = parse_swift("""
func f() {
    if isReady { go() }
    guard let user else { return }
    while running { tick() }
    for x in items where x.valid { use(x) }
    switch state { case .a: break }
    if items.contains(where: { $0 > 1 }) { done() }
    get()
}
""")
        names = [c[1] for c in f["calls"]]
        for bad in ("isReady", "running", "items", "state", "valid"):
            self.assertNotIn(bad, names)
        for good in ("go", "tick", "use", "contains", "done", "get"):
            self.assertIn(good, names)

    def test_local_types(self):
        f = parse_swift("""
func f() {
    let a = Foo()
    let b: Bar = make()
    let c = Outer.Inner(x: 1)
    if let d = cache[id] as? Baz { }
    let e = try await Qux.load()
}
""")
        loc = decl(f, "f")["locals"]
        self.assertEqual(loc["a"], ["Foo", "ctor"])
        self.assertEqual(loc["b"], ["Bar", "decl"])
        self.assertEqual(loc["c"], ["Outer.Inner", "ctor"])
        self.assertEqual(loc["d"], ["Baz", "decl"])
        self.assertNotIn("e", loc)

    def test_resource_references(self):
        f = parse_swift("""
func f() {
    _ = UIImage(named: "logo")
    _ = UIImage(systemName: "star")
    _ = Image("hero")
    _ = Color("Brand")
    _ = NSLocalizedString("login_title", comment: "")
    _ = String(localized: "k1")
    _ = String("not a resource")
    _ = Text("users_title")
    _ = UIStoryboard(name: "Main", bundle: nil).instantiateViewController(withIdentifier: "Home")
    _ = R.image.icon()
    _ = R.string.localizable.greeting()
}
""")
        got = [(r[1], r[2], r[4]) for r in f["rrefs"]]
        self.assertEqual(got, [
            ("image", "logo", "literal"), ("image", "hero", "literal"), ("color", "Brand", "literal"),
            ("string", "login_title", "literal"), ("string", "k1", "literal"),
            ("string", "users_title", "literal"), ("storyboard", "Main", "literal"),
            ("vcid", "Home", "literal"), ("image", "icon", "R"), ("string", "greeting", "R")])

    def test_di_resolve_and_type_refs(self):
        f = parse_swift("func f() { let x = container.resolve(APIClient.self)!\n if y is Foo {}\n z as? Bar }")
        self.assertIn([0, "APIClient", 1, "inject"], f["trefs"])
        self.assertEqual({(t[1], t[3]) for t in f["trefs"]} >= {("Foo", "is"), ("Bar", "as")}, True)

    def test_preview_macro_and_malformed_input(self):
        f = parse_swift("struct A {}\n#Preview { A() }\nstruct B {}\n")
        self.assertEqual([d["name"] for d in f["decls"]], ["A", "B"])
        for src in ("class {", "func (", "}}}", 'let x = "unterminated', '"""never', "#\"raw",
                    "extension { func", "@", "enum E { case }", "init(", "var x: { get"):
            parse_swift(src)


class IosSourcesTest(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("Packages/Core/Package.swift"), "spm")
        self.assertEqual(classify("App/A.swift"), "swift")
        self.assertEqual(classify("App.xcodeproj/project.pbxproj"), "xcodeproj")
        self.assertEqual(classify("App/Base.lproj/Main.storyboard"), "ib")
        self.assertEqual(classify("App/Cell.xib"), "ib")
        self.assertEqual(classify("App/en.lproj/Localizable.strings"), "strings")
        self.assertEqual(classify("App/Localizable.xcstrings"), "xcstrings")
        self.assertEqual(classify("App/Assets.xcassets/logo.imageset/Contents.json"), "xcasset")
        self.assertIsNone(classify("App/Assets.xcassets/Contents.json"))
        self.assertIsNone(classify("Pods/Alamofire/Source/AF.swift"))
        self.assertIsNone(classify("App/.build/checkouts/x/Package.swift"))

    def test_package_swift(self):
        text = (SAMPLE / "Packages/CoreKit/Package.swift").read_text()
        t = parse_package_swift(text)["targets"]
        self.assertEqual([(x[0], x[1], x[2], x[3]) for x in t], [
            ("Networking", "Sources/Networking", "target", []),
            ("CoreKit", "Sources/CoreKit", "target", ["Networking", "Factory"]),  # product kept
            ("CoreKitTests", "Tests/CoreKitTests", "test", ["CoreKit"])])
        t = parse_package_swift('.executableTarget(name: "Tool", dependencies: [.target(name: "Lib")], path: "src/tool")')
        self.assertEqual(t["targets"][0][:4], ["Tool", "src/tool", "executable", ["Lib"]])

    def test_openstep_plist(self):
        v = parse_openstep_plist('{ a = 1; b = ( x, "y z", ); c = { d = "e\\"f"; }; /* c */ }')
        self.assertEqual(v, {"a": "1", "b": ["x", "y z"], "c": {"d": 'e"f'}})

    def test_pbxproj(self):
        rel = "Sample.xcodeproj/project.pbxproj"
        t = parse_pbxproj((SAMPLE / rel).read_text(), rel)["targets"]
        app, tests = t
        self.assertEqual(app[:2], ["App", "app"])
        self.assertIn("App/SampleApp.swift", app[2])
        self.assertIn("App/Base.lproj/Main.storyboard", app[2])
        self.assertIn("App/Assets.xcassets", app[2])
        self.assertIn("App/en.lproj/Localizable.strings", app[2])
        self.assertEqual(app[4], ["CoreKit"])
        self.assertEqual(tests[:2], ["AppTests", "test"])
        self.assertEqual(tests[2], ["AppTests/UserListViewModelTests.swift"])
        self.assertEqual(tests[4], ["App"])
        # synchronized folders (Xcode 16) and a nested project dir
        sync = """{ objects = {
            R = {isa = PBXProject; mainGroup = G; targets = (T);};
            G = {isa = PBXGroup; children = (S); sourceTree = "<group>";};
            S = {isa = PBXFileSystemSynchronizedRootGroup; path = "My App"; sourceTree = "<group>";};
            T = {isa = PBXNativeTarget; name = "My App"; productType = "com.apple.product-type.application";
                 fileSystemSynchronizedGroups = (S); buildPhases = ();};
        }; rootObject = R; }"""
        t = parse_pbxproj(sync, "ios/My.xcodeproj/project.pbxproj")["targets"]
        self.assertEqual(t[0][0], "My_App")
        self.assertEqual(t[0][3], ["ios/My App"])
        self.assertEqual(parse_pbxproj("garbage {", "x/project.pbxproj"), {"targets": []})

    def test_ib_strings_catalogs_assets(self):
        f = parse_ib((SAMPLE / "App/Base.lproj/Main.storyboard").read_text(), "App/Base.lproj/Main.storyboard")
        self.assertEqual((f["rtype"], f["rname"]), ("storyboard", "Main"))
        self.assertEqual([c[0] for c in f["classes"]], ["App.LoginViewController", "App.HomeViewController"])
        self.assertEqual([i[0] for i in f["ids"]], ["Login", "Home"])  # Ghost is commented out
        self.assertEqual([a[0] for a in f["actions"]], ["loginTapped"])
        self.assertEqual(parse_strings('/* c */\n"a" = "x";\n"b.c" = "y";')["keys"], [["a", 2], ["b.c", 3]])
        self.assertEqual([k[0] for k in parse_xcstrings('{"strings": {"hello": {}, "bye": {}}}')["keys"]],
                         ["hello", "bye"])
        self.assertEqual(asset_kind("A/Assets.xcassets/Brand.colorset/Contents.json"), ("color", "Brand"))


class IosSampleGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="graph-ios-"))
        cls.root = cls.tmp / "proj"
        shutil.copytree(SAMPLE, cls.root, ignore=shutil.ignore_patterns(".agent-cache"))
        cls.g, _ = load_graph(cls.root, store=MemoryFactStore())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def edge(self, src, dst, kind, prov=None):
        es = [e for e in self.g.out.get(src, ()) if e.dst == dst and e.kind == kind]
        self.assertTrue(es, f"missing {src} -{kind}-> {dst}; out: {self.g.out.get(src)}")
        if prov:
            self.assertEqual(es[0].prov, prov, es[0])
        return es[0]

    def test_modules_from_xcode_and_spm(self):
        mods = {n.name: n for n in self.g.nodes.values() if n.kind == "module"}
        self.assertEqual(set(mods), {"App", "AppTests", "CoreKit", "Networking", "CoreKitTests"})
        self.assertNotIn("Ghost", mods)
        self.edge("module:App", "module:CoreKit", "depends_on", EXTRACTED)
        self.edge("module:AppTests", "module:App", "depends_on", EXTRACTED)
        self.edge("module:CoreKit", "module:Networking", "depends_on", EXTRACTED)
        self.assertEqual(self.g.nodes["file:App/SampleApp.swift"].module, "App")
        self.assertEqual(self.g.nodes["file:App/Info.plist"].module, "App")  # folder fallback
        self.assertEqual(self.g.nodes[f"file:{REPO_FILE}"].module, "CoreKit")
        self.assertEqual(self.g.nodes["file:App/Assets.xcassets/logo.imageset/Contents.json"].module, "App")

    def test_node_ids_use_module_and_labels(self):
        for nid in ("CoreKit.UserRepository.load(id:)", "App.UserListViewModel.select(_:)",
                    "App.UserListViewModel.select(id:)", "Networking.URLSessionClient.init(session:)",
                    "App.Formatter.Nested.Deep.pong()", "CoreKit.String.trimmed()",
                    "CoreKit.User.displayName()", "res:image/logo", "res:color/AccentColor",
                    "res:string/users_title", "res:storyboard/Main", "res:vcid/Home"):
            self.assertIn(nid, self.g.nodes)
        self.assertNotIn("res:vcid/Ghost", self.g.nodes)
        self.assertEqual(self.g.find("UserListViewModel.select"),
                         ["App.UserListViewModel.select(_:)", "App.UserListViewModel.select(id:)"])

    def test_extensions_attach_to_their_types(self):
        # internal type: members land on User; conformance becomes inherits
        self.assertEqual(self.g.nodes["CoreKit.User.displayName()"].parent, "CoreKit.User")
        # external type: receiver-typed declaration
        self.assertEqual(self.g.nodes["CoreKit.String.trimmed()"].extra["decl"]["recv"], "String")
        self.edge("CoreKit.User.description", "CoreKit.User.displayName()", "calls", EXTRACTED)
        self.edge("CoreKit.User.displayName()", "CoreKit.String.trimmed()", "calls", INFERRED)

    def test_calls_scope_receivers_and_labels(self):
        g = self.g
        self.edge("App.UserListView.body", "App.UserListView.currentID()", "calls", EXTRACTED)
        # labels pick the overload: select(user) -> select(_:)
        e = self.edge("App.UserListView.body", "App.UserListViewModel.select(_:)", "calls", INFERRED)
        self.assertIsNone(e.cands)
        self.edge("App.UserListView.body", "App.UserListViewModel.refresh(id:)", "calls", INFERRED)
        # self.repository inside a Task closure -> protocol requirement
        self.edge("App.UserListViewModel.refresh(id:)", "CoreKit.UserRepositoryProtocol.load(id:)",
                  "calls", INFERRED)
        # property initialized by constructor, static shared instance chain
        self.edge("CoreKit.UserRepository.load(id:)", "CoreKit.UserCache.get(_:)", "calls", INFERRED)
        self.edge("CoreKit.UserRepository.load(id:)", "Networking.APIClient.get(_:)", "calls", INFERRED)
        self.edge("App.SampleApp.model", "App.Container.makeViewModel()", "calls")
        # local typed by `as?` and by a qualified constructor
        self.edge("App.LoginViewController.performLogin()", "App.HomeViewController.configure(with:)",
                  "calls", INFERRED)
        self.edge("App.closures(_:repo:)", "App.Formatter.Nested.ping()", "calls", INFERRED)
        self.edge("App.Formatter.Nested.ping()", "App.Formatter.Nested.Deep.pong()", "calls")
        # code in #if branches and in string interpolation (incl. raw strings)
        self.edge("App.closures(_:repo:)", "App.debugLog(_:)", "calls", EXTRACTED)
        self.edge("App.Formatter.raw", "App.Formatter.realCode()", "calls", EXTRACTED)
        self.edge("App.Formatter.multi", "App.Formatter.realCode()", "calls", EXTRACTED)
        self.assertFalse([e for e in g.edges if e.kind == "calls" and ("haunt" in e.dst or "notCode" in e.dst)])
        # constructor calls and calls from tests across modules (@testable import)
        self.edge("App.Container.repository", "CoreKit.UserRepository", "calls", EXTRACTED)
        self.edge("AppTests.UserListViewModelTests.testResetClearsRepository()",
                  "App.UserListViewModel.reset()", "calls", INFERRED)
        self.assertGreaterEqual(g.stats["calls"].get("invoke", 0), 1)  # transform(3)

    def test_ambiguous_and_skipped(self):
        es = self.g.out_edges("App.invalidateFirst(_:)", ("calls",))
        self.assertEqual(len(es), 1)
        self.assertEqual(es[0].prov, AMBIGUOUS)
        self.assertEqual(set(es[0].cands), {"App.HomeViewController.invalidate()",
                                             "CoreKit.UserCache.invalidate()"})
        targets = {e.dst for e in self.g.out_edges("App.closures(_:repo:)", ("calls",))}
        self.assertFalse(any(t.endswith(("map", "filter")) for t in targets))

    def test_inherits_overrides_injects(self):
        self.edge("CoreKit.UserRepository", "CoreKit.UserRepositoryProtocol", "inherits", EXTRACTED)
        self.edge("Networking.URLSessionClient", "Networking.APIClient", "inherits", EXTRACTED)
        # protocol requirement implementations (no `override` keyword)
        self.edge("CoreKit.UserRepository.load(id:)", "CoreKit.UserRepositoryProtocol.load(id:)",
                  "overrides", INFERRED)
        self.edge("Networking.URLSessionClient.get(_:)", "Networking.APIClient.get(_:)", "overrides")
        self.edge("AppTests.MockRepository.clear()", "CoreKit.UserRepositoryProtocol.clear()", "overrides")
        # @EnvironmentObject
        self.edge("App.UserListView", "App.UserListViewModel", "injects", EXTRACTED)

    def test_resource_references(self):
        self.edge("App.LoginViewController.viewDidLoad()", "res:image/logo", "references", INFERRED)
        self.edge("App.LoginViewController.viewDidLoad()", "res:string/login_title", "references")
        self.edge("App.LoginViewController.viewDidLoad()", "res:color/AccentColor", "references")
        self.edge("App.LoginViewController.performLogin()", "res:vcid/Home", "references")
        self.edge("App.UserListView.body", "res:string/users_title", "references")
        self.edge("res:storyboard/Main", "App.LoginViewController", "references", EXTRACTED)
        self.edge("res:storyboard/Main", "App.LoginViewController.loginTapped(_:)", "references", INFERRED)
        self.edge("file:App/Info.plist", "App.SceneDelegate", "references", EXTRACTED)

    # ---- queries ----

    def test_callers_include_protocol_dispatch(self):
        out = queries.callers(self.g, "UserRepository.load")
        self.assertIn("via overrides", out)
        self.assertIn("UserListViewModel.refresh(id:)", out)
        self.assertIn("App/UserListViewModel.swift:17", out)

    def test_callees_explain_path_impact_query(self):
        g = self.g
        out = queries.callees(g, "UserListView")
        self.assertIn("UserListViewModel.select(_:)", out)
        out = queries.explain(g, "LoginViewController")
        self.assertIn("referenced by (1)", out)
        self.assertIn("res:storyboard/Main", out)
        out = queries.explain(g, "module:CoreKit")
        self.assertIn("used by modules (2)", out)
        out = queries.path(g, "Container", "UserCache")
        self.assertIn("3 hops", out)
        out = queries.impact(g, [REPO_FILE])
        for f in ("App/Container.swift", "App/UserListViewModel.swift", "App/Base.lproj/Main.storyboard",
                  "AppTests/UserListViewModelTests.swift"):
            self.assertIn(f, out)
        self.assertIn("tests to run (2)", out)
        out = queries.impact(g, ["Packages/CoreKit/Package.swift"])
        self.assertIn("module App", out)
        out = queries.query(g, "user repository")
        self.assertIn("UserRepositoryProtocol [protocol]", out.splitlines()[1])

    def test_resolution_rate(self):
        c = self.g.stats["calls"]
        internal = c["internal_resolved"] + c.get("internal_unresolved", 0)
        self.assertEqual(c["internal_resolved"], internal)
        s = queries.build_summary(self.g)
        self.assertIn("call resolution:", s)


class IosCliAndRankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="graph-ios-cli-"))
        self.root = self.tmp / "proj"
        shutil.copytree(SAMPLE, self.root, ignore=shutil.ignore_patterns(".agent-cache"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_build_and_cache(self):
        buf = io.StringIO()
        self.assertEqual(run_cli(["build", "--dir", str(self.root)], out=buf), 0)
        self.assertIn("call resolution:", buf.getvalue())
        self.assertTrue((self.root / ".agent-cache" / "codegraph.json").exists())
        buf = io.StringIO()
        run_cli(["build", "--dir", str(self.root)], out=buf)
        self.assertIn("(0 parsed,", buf.getvalue())
        buf = io.StringIO()
        run_cli(["callers", "UserListViewModel.select(_:)", "--dir", str(self.root)], out=buf)
        self.assertIn("UserListView.body", buf.getvalue())

    def test_repo_map_ranks_swift_with_graph(self):
        from agent_cli.repomap import RepoMap
        rm = RepoMap(self.root)
        rm.build()
        self.assertEqual(rm.rank_source, "graph")
        r = rm._file_ranks
        self.assertGreater(r[REPO_FILE], r["App/SceneDelegate.swift"])


class SwiftPerformanceSmokeTest(unittest.TestCase):
    def test_generated_module(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "Sources" / "Big"
            src.mkdir(parents=True)
            (tmp / "Package.swift").write_text('let p = Package(name: "Big", targets: [.target(name: "Big")])')
            for i in range(150):
                body = "\n".join(
                    f"    func m{j}(_ x: Int) -> Int {{ dep.m{(j + 1) % 20}(x) + helper{j}(value: x) }}\n"
                    f"    private func helper{j}(value: Int) -> Int {{ [value].map {{ $0 + {j} }}.reduce(0, +) }}"
                    for j in range(20))
                (src / f"C{i}.swift").write_text(
                    f"final class C{i} {{\n    private let dep: C{(i + 1) % 150}\n"
                    f"    init(dep: C{(i + 1) % 150}) {{ self.dep = dep }}\n{body}\n}}\n")
            t0 = time.perf_counter()
            g, info = load_graph(tmp, store=MemoryFactStore())
            self.assertLess(time.perf_counter() - t0, 20.0)
            self.assertEqual(info.files, 151)
            self.assertEqual(g.stats["calls"]["internal_resolved"], 150 * 20 * 2)
            self.assertIn("Big.C3.m4(_:)", g.nodes)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
