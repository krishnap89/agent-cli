"""TypeScript / JavaScript support in the code graph: lexer (JSX, regex,
templates), parser, module resolution (tsconfig paths, workspaces, barrels,
CommonJS) and resolution + every query on tests/fixtures/ts_sample (a
workspace monorepo: packages/core library + apps/web React/Angular-style app)."""
import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agent_cli.graph import queries
from agent_cli.graph.build import load_graph
from agent_cli.graph.cli import run as run_cli
from agent_cli.graph.kotlin_lexer import ID, OP, STR
from agent_cli.graph.kotlin_parser import _head
from agent_cli.graph.model import AMBIGUOUS, EXTRACTED, INFERRED
from agent_cli.graph.sources import classify
from agent_cli.graph.store import MemoryFactStore
from agent_cli.graph.ts_lexer import tokenize_ts
from agent_cli.graph.ts_modules import TsModules, parse_package_json, parse_tsconfig
from agent_cli.graph.ts_parser import parse_ts

SAMPLE = Path(__file__).parent / "fixtures" / "ts_sample"
CORE = "packages/core/src/"
WEB = "apps/web/src/"
REPO = CORE + "repo/UserRepository."


def ids(toks):
    return [t[1] for t in toks if t[0] == ID]


def decl(facts, name, kind=None):
    for d in facts["decls"]:
        if d["name"] == name and (kind is None or d["kind"] == kind):
            return d
    raise AssertionError(f"no decl {name}")


def calls_named(facts, name):
    return [c for c in facts["calls"] if c[1] == name]


class TsLexerTest(unittest.TestCase):
    def test_templates_nest_and_keep_code(self):
        toks = tokenize_ts("const s = `a ${f(`b ${g(x)}`)} c { notCode() }`; next()")
        self.assertEqual(ids(toks), ["const", "s", "f", "g", "x", "next"])

    def test_regex_vs_division(self):
        toks = tokenize_ts('x = a / b / c; const re = /"[^/]*\'/g; y = (1) / 2')
        self.assertEqual(ids(toks), ["x", "a", "b", "c", "const", "re", "y"])
        self.assertEqual([t[1] for t in toks if t[0] == OP].count("/"), 3)
        toks = tokenize_ts("return /it's/.test(s)")
        self.assertEqual(ids(toks), ["return", "test", "s"])

    def test_strings_comments_private_and_optional_chaining(self):
        toks = tokenize_ts("import x from './a'\n/* it's */ // don't\nthis.#p?.q ? .5 : 1")
        self.assertEqual([t[1] for t in toks if t[0] == STR], ["./a"])
        self.assertEqual(ids(toks), ["import", "x", "from", "this", "#p", "q"])
        self.assertEqual([t[1] for t in toks if t[0] == OP], [".", "?.", "?", ".", ":"])

    def test_jsx_text_attributes_and_children(self):
        src = ("const v = <div title='it is' onClick={go}>\n  Don't {/* skip() */}\n"
               "  <Tabs.Panel x={f(1)} />{cond && <A/>}</div>\nafter()")
        toks = tokenize_ts(src, jsx=True)
        self.assertEqual(ids(toks), ["const", "v", "div", "go", "Tabs", "Panel", "f", "cond", "A",
                                     "after"])
        self.assertEqual(toks[-3][1:3], ("after", 4))
        self.assertTrue(toks[-3][3])
        self.assertFalse(any(t[3] for t in toks[:-3]))  # no newline flag inside JSX
        self.assertIn("<attr:onClick", [t[1] for t in toks])

    def test_jsx_only_where_an_expression_starts(self):
        self.assertEqual(ids(tokenize_ts("if (a < b) c(d > e)", jsx=True)), ["if", "a", "b", "c", "d", "e"])
        self.assertEqual(ids(tokenize_ts("const f = <T,>(x: T) => x", jsx=True)),
                         ["const", "f", "T", "x", "T", "x"])
        self.assertNotIn("<jsx", [t[1] for t in tokenize_ts("const x = <Foo>y", jsx=False)])
        # an element that never closes is not JSX
        self.assertEqual(ids(tokenize_ts("x = <a\n", jsx=True)), ["x", "a"])


class TsParserTest(unittest.TestCase):
    def test_imports_and_exports(self):
        f = parse_ts("import React, { useState as useS, type FC } from 'react'\n"
                     "import * as core from '@acme/core'\nimport './side-effect'\n"
                     "const lazy = () => import('./Page')\nconst { a, b: c } = require('./cjs')\n"
                     "export * from './a'\nexport * as ns from './n'\nexport { x as y } from './x'\n"
                     "export { lazy }\nexport default lazy")
        self.assertEqual([i[:3] for i in f["imports"]], [
            ["react", "default", "React"], ["react", "useState", "useS"], ["react", "FC", "FC"],
            ["@acme/core", "*", "core"], ["./side-effect", None, None], ["./Page", None, None],
            ["./cjs", "a", "a"], ["./cjs", "b", "c"]])
        self.assertEqual([e[:3] for e in f["exports"]], [
            ["*", None, "./a"], ["ns", "*", "./n"], ["y", "x", "./x"], ["lazy", "lazy", None],
            ["default", "lazy", None]])
        self.assertTrue(f["module"])
        self.assertFalse(parse_ts("declare function track(e: string): void")["module"])

    def test_classes_members_and_parameter_properties(self):
        f = parse_ts("""
@Injectable()
export abstract class Repo<T extends { id: string } = Base> extends Base<T> implements A, B {
  private cache = new Map<string, User>()
  #secret = 1
  static { boot() }
  constructor(private readonly client: Http, plain: number) { super() }
  get size(): number { return this.cache.size }
  set size(v: number) { resize(v) }
  abstract load(id: string): Promise<T | null>
  handle = (e: Event): void => this.load(e.id)
  [key: string]: unknown
}""")
        cls = decl(f, "Repo", "class")
        self.assertEqual(cls["tparams"], ["T"])
        self.assertEqual(cls["supers"], [["Base<T>", True, 3], ["A", False, 3], ["B", False, 3]])
        self.assertTrue(cls["inject_ctor"])
        self.assertEqual(cls["mods"], ["abstract"])
        names = [(d["kind"], d["name"]) for d in f["decls"] if d["parent"] == 0]
        self.assertEqual(names, [("prop", "cache"), ("prop", "#secret"), ("init", "<init>"),
                                 ("init", "constructor"), ("prop", "client"), ("prop", "size"),
                                 ("fun", "load"), ("fun", "handle")])
        self.assertEqual(decl(f, "client")["type"], "Http")
        self.assertEqual(decl(f, "size")["type"], "number")
        self.assertEqual(decl(f, "load")["ret"], "Promise<T | null>")
        self.assertEqual(decl(f, "handle")["params"][0][:2], ["e", "Event"])
        self.assertEqual(calls_named(f, "resize")[0][0], f["decls"].index(decl(f, "size")))
        self.assertEqual(len(calls_named(f, "boot")), 1)

    def test_interfaces_enums_aliases_namespaces(self):
        f = parse_ts("""
export interface Repo<T> extends Base, Other<T> {
  readonly id: string
  load(id: string): Promise<T>
  onChange?: (v: T) => void
  [k: string]: unknown
  (call: number): void
}
export const enum Role { Admin, Member = 'member' }
type Props = { user: User | null; cb: (id: string) => void }
namespace Log { export function tag(s: string) { return s } }
declare module 'ext' { export function ghost(): void }
""")
        self.assertEqual([s[0] for s in decl(f, "Repo")["supers"]], ["Base", "Other<T>"])
        self.assertEqual([(d["kind"], d["name"]) for d in f["decls"] if d["parent"] == 0],
                         [("prop", "id"), ("fun", "load"), ("prop", "onChange")])
        self.assertEqual(decl(f, "onChange")["type"], "(v: T) => void")
        self.assertEqual([d["name"] for d in f["decls"] if d["kind"] == "entry"], ["Admin", "Member"])
        self.assertEqual(decl(f, "Props")["type"], "{user: User | null; cb: (id: string) => void}")
        self.assertEqual(decl(f, "tag")["parent"], f["decls"].index(decl(f, "Log", "namespace")))
        self.assertNotIn("ghost", [d["name"] for d in f["decls"]])

    def test_functions_arrows_objects_and_module_code(self):
        f = parse_ts("""
export default function ({ user, onSelect }: Props) {
  const label = fmt(user)
  const handle = async () => onSelect?.(user.id)
  return label
}
export const useX = <T,>(id: T): T => id
export const api = { async load(id) { return repo.load(id) }, reset: () => repo.clear(), helper }
export const config = { retries: 3, base: url() }
app.use(cors())
""", "src/components/UserCard.tsx")
        card = decl(f, "UserCard")  # anonymous default export: named after the file
        self.assertEqual(card["params"][0][:2], ["<destructured>", "Props"])
        self.assertEqual(card["vars"][:2], ["user", "onSelect"])
        self.assertEqual(decl(f, "handle")["parent"], f["decls"].index(card))
        self.assertEqual(decl(f, "useX")["tparams"], ["T"])
        api = decl(f, "api", "object")
        members = [(d["kind"], d["name"]) for d in f["decls"] if d["parent"] == f["decls"].index(api)]
        self.assertEqual(members, [("fun", "load"), ("fun", "reset"), ("prop", "helper")])
        self.assertEqual(decl(f, "helper")["alias"], "helper")
        self.assertEqual(decl(f, "config")["kind"], "prop")  # no functions: a plain value
        mod = decl(f, "<module>")
        self.assertEqual({c[1] for c in f["calls"] if c[0] == f["decls"].index(mod)}, {"use", "cors"})
        self.assertIn(["default", "UserCard", None, 2], f["exports"])

    def test_call_sites(self):
        f = parse_ts("""
function run(svc: Service) {
  const repo = new Repo<User>(svc)
  const o = { method() { inner() }, other: x }
  if (i < n) { return a > (b) }
  svc.get<Map<string, User>>(key)
  maybe?.(1, 2)
  tag`template ${t()}`
  return <Panel.Body onClick={this.save} label={x} />
}""", "a.tsx")
        names = [c[1] for c in f["calls"]]
        self.assertEqual(names, ["Repo", "inner", "get", "maybe", "t", "Body", "save"])
        get = calls_named(f, "get")[0]
        self.assertEqual((get[2], get[4]), ([["svc", 0]], 1))
        self.assertEqual(calls_named(f, "maybe")[0][4], 2)
        body = calls_named(f, "Body")[0]
        self.assertEqual((body[2], body[4]), ([["Panel", 0]], -1))
        self.assertTrue(calls_named(f, "save")[0][5])
        self.assertEqual(decl(f, "run")["locals"]["repo"], ["Repo", "ctor"])

    def test_commonjs(self):
        f = parse_ts("""
const format = require('./format')
function shout(u) { return u }
exports.report = function (users) { return format.shout(users) }
module.exports.flag = true
module.exports = { shout, loud: shout }
""", "lib/fmt.js")
        self.assertEqual(f["imports"][0][:3], ["./format", "=", "format"])
        self.assertEqual([e[:2] for e in f["exports"]],
                         [["report", "report"], ["flag", "flag"], ["default", "default"]])
        obj = decl(f, "default", "object")
        aliases = [d.get("alias") for d in f["decls"] if d["parent"] == f["decls"].index(obj)]
        self.assertEqual(aliases, ["shout", "shout"])

    def test_this_fields_in_js_constructors(self):
        f = parse_ts("class A {\n  constructor(repo) {\n    this.cache = new Cache()\n"
                     "    this.repo = repo\n  }\n  go() { this.cache.get() }\n}")
        self.assertEqual(decl(f, "cache")["itype"], "Cache")
        self.assertEqual(decl(f, "repo")["parent"], 0)

    def test_head_skips_nullish_union_members(self):
        self.assertEqual(_head("null | User"), "User")
        self.assertEqual(_head("readonly User[]"), "User")
        self.assertEqual(_head("List<User>?"), "List")


class TsModulesTest(unittest.TestCase):
    def test_jsonc_and_package_json(self):
        cfg = parse_tsconfig('{\n // c\n "$schema": "https://x/y", "extends": "./base",\n'
                             ' "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"],},},}')
        self.assertEqual(cfg, {"extends": ["./base"], "baseUrl": ".", "paths": {"@/*": ["src/*"]}})
        self.assertEqual(parse_tsconfig("{oops")["paths"], None)
        pkg = parse_package_json('{"name": "@a/b", "exports": {".": {"types": "./dist/i.d.ts", '
                                 '"import": "./dist/i.js"}, "./sub": "./dist/sub.js"},\n'
                                 '"dependencies": {"@a/c": "1"}}')
        self.assertEqual(pkg["exports"], {".": ["./dist/i.d.ts", "./dist/i.js"], "./sub": ["./dist/sub.js"]})
        self.assertEqual(pkg["deps"], [["@a/c", 2]])

    def test_resolution(self):
        files = ["app/src/a.ts", "app/src/lib/b.tsx", "app/src/lib/index.ts", "pkgs/c/src/index.ts",
                 "pkgs/c/src/sub.ts", "app/src/esm.ts", "app/legacy/x.js"]
        facts = {
            "tsconfig.base.json": {"lang": "tsconfig", "extends": [], "baseUrl": ".",
                                   "paths": {"#shared/*": ["app/src/lib/*"]}},
            "app/tsconfig.json": {"lang": "tsconfig", "extends": ["../tsconfig.base.json"],
                                  "baseUrl": None, "paths": None},
            "pkgs/c/package.json": {"lang": "npm", "name": "@a/c", "main": "dist/index.js",
                                    "types": "", "module": "", "exports": {}},
        }
        m = TsModules(facts, files)
        self.assertEqual(m.resolve("app/src/a.ts", "./lib/b"), "app/src/lib/b.tsx")
        self.assertEqual(m.resolve("app/src/a.ts", "./lib"), "app/src/lib/index.ts")
        self.assertEqual(m.resolve("app/src/a.ts", "./esm.js"), "app/src/esm.ts")
        self.assertEqual(m.resolve("app/src/a.ts", "#shared/b"), "app/src/lib/b.tsx")  # via extends
        self.assertEqual(m.resolve("app/src/a.ts", "app/legacy/x"), "app/legacy/x.js")  # baseUrl
        self.assertEqual(m.resolve("app/src/a.ts", "@a/c"), "pkgs/c/src/index.ts")  # dist -> src
        self.assertEqual(m.resolve("app/src/a.ts", "@a/c/sub"), "pkgs/c/src/sub.ts")
        for external in ("react", "node:fs", "./missing", "../../../outside", "./a.css"):
            self.assertIsNone(m.resolve("app/src/a.ts", external), external)

    def test_classify(self):
        self.assertEqual(classify("src/a.tsx"), "ts")
        self.assertEqual(classify("types/globals.d.ts"), "ts")
        self.assertEqual(classify("src/legacy.cjs"), "ts")
        self.assertEqual(classify("package.json"), "npm")
        self.assertEqual(classify("apps/web/tsconfig.app.json"), "tsconfig")
        self.assertEqual(classify("jsconfig.json"), "tsconfig")
        self.assertIsNone(classify("public/vendor.min.js"))
        self.assertIsNone(classify("node_modules/react/index.js"))
        self.assertIsNone(classify("apps/web/.next/server/page.js"))
        self.assertIsNone(classify("apps/web/dist/bundle.js"))


class TsSampleGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="graph-ts-"))
        cls.root = cls.tmp / "proj"
        shutil.copytree(SAMPLE, cls.root, ignore=shutil.ignore_patterns(".agent-cache"))
        (cls.root / "apps/web/dist").mkdir()
        (cls.root / "apps/web/dist/bundle.js").write_text("export function ghostBundle() {}")
        cls.g, cls.info = load_graph(cls.root, store=MemoryFactStore())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def edge(self, src, dst, kind, prov=None):
        es = [e for e in self.g.out.get(src, ()) if e.dst == dst and e.kind == kind]
        self.assertTrue(es, f"missing {src} -{kind}-> {dst}; out: {self.g.out.get(src)}")
        if prov:
            self.assertEqual(es[0].prov, prov, es[0])
        return es[0]

    def no_edge(self, src, dst, kind):
        self.assertFalse([e for e in self.g.out.get(src, ()) if e.dst == dst and e.kind == kind])

    def test_modules_files_and_ids(self):
        mods = {n.name for n in self.g.nodes.values() if n.kind == "module"}
        self.assertEqual(mods, {"ts-sample", "@acme/core", "web"})
        self.edge("module:web", "module:@acme/core", "depends_on", EXTRACTED)
        self.assertEqual(self.g.nodes[f"file:{WEB}api.ts"].module, "web")
        self.assertEqual(self.g.nodes[f"file:{CORE}cache.ts"].module, "@acme/core")
        self.assertNotIn("file:apps/web/dist/bundle.js", self.g.nodes)
        for nid in (REPO + "UserRepository.load", CORE + "cache.UserCache.get",
                    WEB + "components/UserCard.UserCard", WEB + "components/UserList.UserList.onSelect",
                    WEB + "api.userApi.fetchUser", WEB + "main.<module>", WEB + "services/Logger.Log.tag",
                    CORE + "models/user.Role.Admin"):
            self.assertIn(nid, self.g.nodes)
        self.assertNotIn(CORE + "cache.UserCache.User", self.g.nodes)  # Map<string, User> is a type
        self.assertEqual(self.g.find("UserRepository.load"), [REPO + "UserRepository.load"])
        self.assertEqual(self.g.short(REPO + "UserRepository.load"), "UserRepository.load")
        self.assertEqual(self.g.nodes[REPO + "UserRepository.load"].sig,
                         "async load(id: string): Promise<User>")

    def test_imports_follow_paths_workspaces_and_barrels(self):
        # '@acme/core' -> packages/core (main: dist/index.js -> src/index.ts) -> re-export
        self.edge(f"file:{WEB}api.ts", REPO + "UserRepository", "imports", EXTRACTED)
        self.edge(f"file:{WEB}api.ts", f"file:{CORE}index.ts", "imports")
        self.edge(f"file:{WEB}api.ts", "module:@acme/core", "imports")
        # `export * from './models/user'`, `export { default as Cache }`, type-only imports
        self.edge(f"file:{WEB}components/UserCard.tsx", CORE + "models/user.Role", "imports")
        self.edge(f"file:{WEB}hooks/useUser.ts", CORE + "models/user.User", "imports")
        # tsconfig paths '@/*' (JSONC config) and an ESM-style './cache.js' import of cache.ts
        self.edge(f"file:{WEB}hooks/useUser.ts", WEB + "api.userApi", "imports", EXTRACTED)
        self.edge(f"file:{CORE}repo/UserRepository.ts", CORE + "cache.UserCache", "imports")
        # CommonJS: require of module.exports = {...} and destructured require
        self.edge(f"file:{WEB}legacy/report.js", WEB + "legacy/format.whisper", "imports")

    def test_calls(self):
        e = self.edge
        e(REPO + "UserRepository.load", CORE + "cache.UserCache.get", "calls", INFERRED)  # this.cache = new …
        e(REPO + "UserRepository.load", CORE + "http/client.HttpClient.get", "calls", INFERRED)  # param prop
        e(REPO + "UserRepository.load", REPO + "BaseRepository.log", "calls", EXTRACTED)  # inherited
        e(REPO + "UserRepository.load", CORE + "models/user.displayName", "calls", EXTRACTED)  # in `${}`
        e(REPO + "UserRepository.cache", CORE + "cache.UserCache", "calls", EXTRACTED)  # new
        e(WEB + "hooks/useUser.useUser", WEB + "api.userApi.fetchUser", "calls", EXTRACTED)
        e(WEB + "api.userApi.fetchUser", REPO + "UserRepository.load", "calls", INFERRED)
        e(WEB + "components/UserList.UserList", WEB + "components/UserCard.UserCard", "calls", EXTRACTED)  # JSX
        e(WEB + "components/UserList.UserList", WEB + "components/UserList.UserList.onSelect", "calls")
        e(WEB + "components/UserCard.UserCard", WEB + "components/UserCard.UserCard.handleClick", "calls")
        e(WEB + "components/UserList.UserList", CORE + "models/user.displayName", "calls", EXTRACTED)  # ns
        e(WEB + "components/UserList.UserList", CORE + "http/client.createClient", "calls", EXTRACTED)
        e(WEB + "main.<module>", WEB + "components/UserList.UserList", "calls", EXTRACTED)
        e(WEB + "main.<module>", REPO + "UserRepository.clear", "calls", INFERRED)
        e(WEB + "legacy/format.whisper", WEB + "legacy/format.shout", "calls", EXTRACTED)
        e(WEB + "legacy/report.report", WEB + "legacy/format.shout", "calls")  # format.shout(u)
        e(WEB + "legacy/report.report", WEB + "legacy/format.whisper", "calls", EXTRACTED)
        e(WEB + "services/Logger.Log.line", WEB + "services/Logger.Log.tag", "calls", EXTRACTED)
        e(WEB + "services/UserService.UserService.refresh", WEB + "services/Logger.Log.line", "calls")
        e(WEB + "services/UserService.UserService.refresh", WEB + "services/Logger.Logger.info", "calls")
        e(WEB + "services/UserService.UserService.refresh", REPO + "UserRepository.load", "calls")  # inject()
        e(WEB + "services/UserService.UserService.refresh", WEB + "globals.track", "calls", INFERRED)  # .d.ts
        e(WEB + "__tests__/UserRepository.test.<module>", REPO + "UserRepository.clear", "calls", INFERRED)
        e(WEB + "__tests__/UserRepository.test.<module>", WEB + "__tests__/UserRepository.test.MockRepo.clear",
          "calls")
        # nothing from comments / JSX text; Map.clear() is not UserRepository.clear
        dsts = {x.dst for x in self.g.edges if x.kind == "calls"}
        self.assertFalse([d for d in dsts if "ghost" in d.lower()])
        self.no_edge(CORE + "cache.UserCache.invalidate", REPO + "UserRepository.clear", "calls")
        self.assertGreaterEqual(self.g.stats["calls"]["invoke"], 1)  # onSelect?.(…)

    def test_ambiguous_unknown_receiver(self):
        es = self.g.out_edges(WEB + "legacy/format.flush", ("calls",))
        self.assertEqual(len(es), 1)
        self.assertEqual(es[0].prov, AMBIGUOUS)
        self.assertEqual(set(es[0].cands), {CORE + "cache.UserCache.invalidate",
                                             WEB + "services/Logger.Logger.invalidate"})

    def test_inherits_overrides_injects_types(self):
        e = self.edge
        e(REPO + "UserRepository", REPO + "BaseRepository", "inherits", EXTRACTED)
        e(REPO + "BaseRepository", REPO + "Repository", "inherits", EXTRACTED)
        e(CORE + "http/client.FetchClient", CORE + "http/client.HttpClient", "inherits", EXTRACTED)
        e(WEB + "__tests__/UserRepository.test.MockRepo", REPO + "Repository", "inherits", EXTRACTED)
        e(REPO + "UserRepository.clear", REPO + "BaseRepository.clear", "overrides", EXTRACTED)  # keyword
        e(REPO + "UserRepository.load", REPO + "BaseRepository.load", "overrides", INFERRED)
        e(CORE + "http/client.FetchClient.get", CORE + "http/client.HttpClient.get", "overrides", INFERRED)
        e(REPO + "UserRepository", CORE + "http/client.HttpClient", "injects", EXTRACTED)  # @Injectable ctor
        e(WEB + "services/UserService.UserService", WEB + "services/Logger.Logger", "injects")
        e(WEB + "services/UserService.UserService", REPO + "UserRepository", "injects")  # inject(…)
        e(WEB + "components/UserCard.UserCard", WEB + "components/UserCard.Props", "uses_type")
        e(WEB + "components/UserCard.Props", CORE + "models/user.User", "uses_type")
        e(WEB + "components/UserCard.UserCard", CORE + "models/user.Role", "uses_type")  # Role.Admin

    # ---- queries ----

    def test_callers_follow_interface_dispatch(self):
        out = queries.callers(self.g, "HttpClient.get")
        self.assertIn("UserRepository.load", out)
        out = queries.callers(self.g, "FetchClient.get")
        self.assertIn("via overrides", out)
        self.assertIn("packages/core/src/repo/UserRepository.ts:32", out)

    def test_callees_explain_path_impact_query(self):
        g = self.g
        out = queries.callees(g, "UserList")
        self.assertIn("UserCard", out)
        out = queries.explain(g, "UserRepository")
        self.assertIn("injected into (1)", out)
        self.assertIn("@Injectable class UserRepository extends BaseRepository<User>", out)
        out = queries.explain(g, "module:@acme/core")
        self.assertIn("used by modules (1)", out)
        out = queries.path(g, "UserList", "UserCache")
        self.assertIn("4 hops", out)
        out = queries.impact(g, [CORE + "cache.ts"])
        for f in (CORE + "repo/UserRepository.ts", WEB + "api.ts", WEB + "services/UserService.ts",
                  WEB + "main.tsx", WEB + "hooks/useUser.ts"):
            self.assertIn(f, out)
        self.assertIn("tests to run (1): apps/web/src/__tests__/UserRepository.test.ts", out)
        out = queries.query(g, "user repository")
        self.assertIn("UserRepository [class]", out)

    def test_resolution_rate(self):
        c = self.g.stats["calls"]
        internal = c["internal_resolved"] + c.get("internal_unresolved", 0)
        self.assertEqual(c["internal_resolved"], internal)
        self.assertIn("call resolution:", queries.build_summary(self.g))


class TsCliAndRankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="graph-ts-cli-"))
        self.root = self.tmp / "proj"
        shutil.copytree(SAMPLE, self.root, ignore=shutil.ignore_patterns(".agent-cache"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_build_cache_and_query(self):
        buf = io.StringIO()
        self.assertEqual(run_cli(["build", "--dir", str(self.root)], out=buf), 0)
        self.assertIn("call resolution:", buf.getvalue())
        buf = io.StringIO()
        run_cli(["build", "--dir", str(self.root)], out=buf)
        self.assertIn("(0 parsed,", buf.getvalue())
        buf = io.StringIO()
        run_cli(["callers", "UserRepository.load", "--dir", str(self.root)], out=buf)
        self.assertIn("UserService.refresh", buf.getvalue())

    def test_repo_map_ranks_ts_with_graph(self):
        from agent_cli.repomap import RepoMap
        rm = RepoMap(self.root)
        rm.build()
        self.assertEqual(rm.rank_source, "graph")
        r = rm._file_ranks
        self.assertGreater(r[CORE + "repo/UserRepository.ts"], r[WEB + "main.tsx"])


class TsRepoMapExtractTest(unittest.TestCase):
    """The repo map lists TS/JS definitions from the code graph's parser."""

    def defs(self, rel):
        from agent_cli.repomap import extract
        info = extract(SAMPLE / rel)
        return info, [(d[0], d[1], d[2], d[3]) for d in info["defs"]], {d[3]: d[4] for d in info["defs"]}

    def test_components_objects_and_exports(self):
        info, defs, sigs = self.defs(WEB + "components/UserCard.tsx")
        self.assertEqual(defs, [(4, 0, "class", "Props"), (6, 0, "def", "UserCard")])
        self.assertEqual(sigs["UserCard"], "export default function UserCard({…}: Props)")
        self.assertFalse({"ghost", "ghostText", "panic"} & set(info["refs"]))  # comments / JSX text
        self.assertIn("displayName", info["refs"])
        _, defs, sigs = self.defs(WEB + "api.ts")
        self.assertEqual(defs, [(3, 0, "def", "repository"), (5, 0, "class", "userApi"),
                                (6, 1, "def", "fetchUser"), (9, 1, "def", "reset")])
        self.assertEqual(sigs["repository"], "export const repository = new UserRepository(…)")

    def test_classes_decorators_and_members(self):
        _, defs, sigs = self.defs(CORE + "repo/UserRepository.ts")
        names = [(d[1], d[3]) for d in defs]
        self.assertIn((1, "constructor"), names)
        self.assertNotIn((1, "client"), names)  # fields are data; the graph keeps them
        self.assertNotIn((1, "cache"), names)
        self.assertEqual(sigs["UserRepository"],
                         "@Injectable export class UserRepository extends BaseRepository<User>")
        self.assertEqual(sigs["log"], "protected log(msg: string): void")

    def test_skips_locals_entries_fields_and_aliases(self):
        _, defs, _ = self.defs(WEB + "components/UserList.tsx")
        self.assertEqual([d[3] for d in defs], ["UserList"])  # not onSelect / first / re
        _, defs, _ = self.defs(CORE + "models/user.ts")
        self.assertEqual([d[3] for d in defs], ["User", "UserId", "Role", "displayName"])
        _, defs, sigs = self.defs(WEB + "legacy/format.js")
        self.assertEqual([d[3] for d in defs], ["shout", "whisper", "flush", "default"])
        self.assertEqual(sigs["default"], "export default { … }")

    def test_only_names_other_files_can_use_are_link_targets(self):
        from agent_cli.repomap import _ts_defs_refs
        defs, _ = _ts_defs_refs("export function a() {}\nconst b = 1\nexport class C { m() {} }\n"
                                "export default function () {}", "x/Widget.ts")
        self.assertEqual([(d[3], d[5]) for d in defs], [("a", 1), ("b", 0), ("C", 1), ("m", 0),
                                                        ("Widget", 1)])
        defs, _ = _ts_defs_refs("declare function track(e: string): void", "globals.d.ts")
        self.assertEqual([(d[3], d[5]) for d in defs], [("track", 1)])  # a script: global

    def test_vue_keeps_regex_and_cache_version_reindexes(self):
        from agent_cli import repomap
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "A.vue").write_text("<script>\nexport function setup() {}\n</script>\n")
            self.assertEqual([d[3] for d in repomap.extract(tmp / "A.vue")["defs"]], ["setup"])
            (tmp / "a.ts").write_text("export const f = () => 1\n")
            (tmp / ".agent-cache").mkdir()
            (tmp / ".agent-cache" / "repomap.json").write_text(
                '{"version": 4, "parsers": [], "files": {}}')
            rm = repomap.RepoMap(tmp)
            _total, reparsed = rm.build()
            self.assertEqual(reparsed, 2)
            self.assertEqual(rm.files["a.ts"]["defs"][0][3:], ["f", "export const f = () => …", 1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TsPerformanceSmokeTest(unittest.TestCase):
    def test_generated_project(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "src"
            src.mkdir()
            for i in range(150):
                nxt = (i + 1) % 150
                body = "\n".join(
                    f"  m{j}(x: number): number {{ return this.dep.m{(j + 1) % 20}(x) + helper{j}(x) }}"
                    for j in range(20))
                helpers = "\n".join(
                    f"function helper{j}(v: number): number {{ return [v].map(n => n + {j})[0] }}"
                    for j in range(20))
                (src / f"c{i}.ts").write_text(
                    f"import {{ C{nxt} }} from './c{nxt}'\n{helpers}\n"
                    f"export class C{i} {{\n  constructor(private dep: C{nxt}) {{}}\n{body}\n}}\n")
            t0 = time.perf_counter()
            g, info = load_graph(tmp, store=MemoryFactStore())
            self.assertLess(time.perf_counter() - t0, 20.0)
            self.assertEqual(info.files, 150)
            self.assertEqual(g.stats["calls"]["internal_resolved"], 150 * 20 * 2)
            self.assertIn("src/c3.C3.m4", g.nodes)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
