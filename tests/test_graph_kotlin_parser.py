"""Kotlin tokenizer / parser edge cases for the code graph."""
import unittest

from agent_cli.graph.kotlin_lexer import ID, STR, tokenize
from agent_cli.graph.kotlin_parser import parse_kotlin


def names(facts, kind=None):
    return [d["name"] for d in facts["decls"] if kind is None or d["kind"] == kind]


def calls(facts):
    return [c[1] for c in facts["calls"]]


def decl(facts, name, kind=None):
    for d in facts["decls"]:
        if d["name"] == name and (kind is None or d["kind"] == kind):
            return d
    raise AssertionError(f"no decl {name}")


def path_of(facts, idx):
    parts = []
    while idx >= 0:
        parts.append(facts["decls"][idx]["name"])
        idx = facts["decls"][idx]["parent"]
    return ".".join(reversed(parts))


class LexerTest(unittest.TestCase):
    def test_comments_are_dropped_including_nested_block_comments(self):
        toks = tokenize("a /* x /* y */ still comment */ b // c\nd")
        self.assertEqual([t[1] for t in toks], ["a", "b", "d"])
        self.assertEqual(toks[2][2], 2)
        self.assertTrue(toks[2][3], "newline before d")

    def test_string_contents_are_not_code_but_templates_are(self):
        toks = tokenize('val s = "a { b } ${f(x)} $name c"')
        texts = [t[1] for t in toks]
        self.assertIn("f", texts)
        self.assertNotIn("name", texts)  # $name is a reference, dropped
        self.assertNotIn("b", texts)
        self.assertEqual(sum(1 for t in toks if t[0] == STR), 1)

    def test_raw_string_with_extra_closing_quotes_and_template(self):
        src = 'val r = """\n  fun x() { y() }\n  ${g()}\n""""\nval after = 1'
        toks = tokenize(src)
        texts = [t[1] for t in toks]
        self.assertIn("g", texts)
        self.assertNotIn("y", texts)
        self.assertEqual(texts[-4:], ["val", "after", "=", "1"])
        self.assertEqual(toks[-1][2], 5)

    def test_nested_string_inside_template_with_brace(self):
        toks = tokenize('"v: ${h("}")} end" + k')
        self.assertEqual([t[1] for t in toks if t[0] == ID], ["h", "k"])

    def test_dollar_escape_template_is_literal(self):
        toks = tokenize('"${\'$\'}{notCode()}"')
        self.assertNotIn("notCode", [t[1] for t in toks])

    def test_char_literals_and_labels(self):
        toks = tokenize("if (c == '{' || c == '\\'') return@forEach this@Outer.x loop@ for")
        texts = [t[1] for t in toks]
        self.assertNotIn("{", texts)
        self.assertIn("return", texts)
        self.assertNotIn("forEach", texts)
        self.assertNotIn("Outer", texts)
        self.assertIn("this", texts)

    def test_backticked_identifier_and_ranges(self):
        toks = tokenize("fun `my test`() { for (i in 1..10) {} }")
        texts = [t[1] for t in toks]
        self.assertIn("my test", texts)
        self.assertIn("..", texts)
        self.assertIn("1", texts)


class ParserTest(unittest.TestCase):
    def test_package_imports_alias_and_star(self):
        f = parse_kotlin("@file:JvmName(\"X\")\npackage a.b\n\nimport c.D\nimport e.*\nimport f.G as H\n")
        self.assertEqual(f["package"], "a.b")
        self.assertEqual([i[:3] for i in f["imports"]],
                         [["c.D", None, False], ["e", None, True], ["f.G", "H", False]])

    def test_class_header_primary_ctor_and_supertypes(self):
        f = parse_kotlin("""
class Repo @Inject constructor(
    private val api: Api,
    val cache: Map<String, List<User>> = emptyMap(),
    plain: Int,
) : Base(api), Iface, Other by impl {
}
""")
        d = decl(f, "Repo")
        self.assertTrue(d["inject_ctor"])
        self.assertEqual([(p[0], p[1], p[2], p[3]) for p in d["params"]],
                         [("api", "Api", True, False),
                          ("cache", "Map<String, List<User>>", True, True),
                          ("plain", "Int", False, False)])
        self.assertEqual([s[:2] for s in d["supers"]],
                         [["Base", True], ["Iface", False], ["Other", False]])
        self.assertIn("emptyMap", calls(f))

    def test_annotation_on_next_declaration_is_not_a_constructor(self):
        f = parse_kotlin("class A\n@Suppress(\"x\") class B(val x: Int)\n")
        self.assertEqual(decl(f, "A")["params"], [])
        self.assertEqual(decl(f, "B")["annos"], ["Suppress"])
        self.assertEqual(len(decl(f, "B")["params"]), 1)

    def test_nested_inner_companion_and_object(self):
        f = parse_kotlin("""
class Outer {
    class Nested { inner class Deeper { fun d() {} } }
    companion object { fun create() = Outer() }
    object Single { val x = 1 }
    enum class Mode { A, B }
    sealed interface State
    fun interface Listener { fun on() }
}
""")
        paths = {path_of(f, i) for i in range(len(f["decls"]))}
        for p in ("Outer.Nested.Deeper.d", "Outer.Companion.create", "Outer.Single.x",
                  "Outer.Mode.A", "Outer.State", "Outer.Listener.on"):
            self.assertIn(p, paths)
        self.assertEqual(decl(f, "Mode")["kind"], "enum")
        self.assertEqual(decl(f, "Listener")["kind"], "interface")
        self.assertIn("companion", decl(f, "Companion")["mods"])

    def test_extension_functions_and_properties(self):
        f = parse_kotlin("""
fun String.isEmail(): Boolean = contains("@")
fun <T> List<T>.second(): T = this[1]
fun Outer.Nested.extra(x: Int = 1, vararg rest: String) {}
fun String?.orBlank() = this ?: ""
val String.firstChar: Char get() = this[0]
""")
        self.assertEqual(decl(f, "isEmail")["recv"], "String")
        self.assertEqual(decl(f, "second")["recv"], "List<T>")
        self.assertEqual(decl(f, "second")["tparams"], ["T"])
        self.assertEqual(decl(f, "extra")["recv"], "Outer.Nested")
        self.assertEqual(decl(f, "orBlank")["recv"], "String?")
        ex = decl(f, "extra")["params"]
        self.assertTrue(ex[0][3])   # has default
        self.assertTrue(ex[1][5])   # vararg
        self.assertEqual(decl(f, "firstChar", "prop")["recv"], "String")
        self.assertEqual(decl(f, "firstChar", "prop")["type"], "Char")

    def test_lambdas_calls_receivers_and_trailing_lambdas(self):
        f = parse_kotlin("""
fun f(repo: Repo) {
    items.forEach { item -> render(item) }
    repo.load().items.map(::toUi)
    button.setOnClickListener { vm.refresh(id()) }
    val g: (Int) -> Int = { x -> x + 1 }
    launch { delay(1) }
    listOf<String>()
    Foo::bar
}
""")
        by_name = {c[1]: c for c in f["calls"]}
        self.assertEqual(by_name["forEach"][2], [["items", 0]])
        self.assertEqual(by_name["forEach"][4], 1)           # trailing lambda counts
        self.assertIsNone(by_name["render"][2])
        self.assertEqual(by_name["map"][2], [["repo", 0], ["load", 1], ["items", 0]])
        self.assertTrue(by_name["toUi"][5])                   # callable reference
        self.assertEqual(by_name["refresh"][2], [["vm", 0]])
        self.assertIn("id", by_name)
        self.assertIn("launch", by_name)
        self.assertIn("listOf", by_name)                      # generic call
        self.assertEqual(by_name["bar"][2], [["Foo", 0]])
        self.assertNotIn("x", by_name)                        # lambda param
        self.assertNotIn("item", by_name)
        self.assertEqual(decl(f, "f")["locals"]["g"], ["(Int) -> Int", "decl"])
        owner = {f["decls"][c[0]]["name"] for c in f["calls"]}
        self.assertEqual(owner, {"f"})                        # lambda calls belong to f

    def test_expression_bodies_end_at_newline_unless_continued(self):
        f = parse_kotlin("""
class C {
    fun a() = listOf(1)
        .map { it }
    fun b() = if (x) one()
        else two()
    val p = 5
        private set
    fun c() = three()
}
""")
        owner_of = {c[1]: f["decls"][c[0]]["name"] for c in f["calls"]}
        self.assertEqual(owner_of["map"], "a")
        self.assertEqual(owner_of["two"], "b")
        self.assertEqual(owner_of["three"], "c")
        self.assertEqual(names(f, "fun"), ["a", "b", "c"])

    def test_property_accessors_and_delegates(self):
        f = parse_kotlin("""
class A {
    val count: Int
        get() = repo.count()
    private val vm: MainViewModel by viewModels()
    private val api by inject<Api>()
    val cache = Cache(1)
    @Inject lateinit var tracker: Tracker
}
""")
        self.assertEqual(decl(f, "vm")["delegate"], "viewModels")
        self.assertEqual(decl(f, "api")["dtype"], "Api")
        self.assertEqual(decl(f, "cache")["itype"], "Cache")
        self.assertEqual(decl(f, "tracker")["annos"], ["Inject"])
        self.assertIn("lateinit", decl(f, "tracker")["mods"])
        count_call = [c for c in f["calls"] if c[1] == "count"][0]
        self.assertEqual(f["decls"][count_call[0]]["name"], "count")

    def test_enum_entries_with_args_and_bodies(self):
        f = parse_kotlin("""
enum class Color(val rgb: Int) {
    RED(red()) { override fun label() = "r" },
    GREEN(2);
    open fun label() = "?"
}
""")
        self.assertEqual(names(f, "entry"), ["RED", "GREEN"])
        labels = [d for d in f["decls"] if d["name"] == "label"]
        self.assertEqual(len(labels), 2)
        self.assertEqual(f["decls"][labels[0]["parent"]]["name"], "RED")
        self.assertEqual(f["decls"][labels[1]["parent"]]["name"], "Color")
        red_call = [c for c in f["calls"] if c[1] == "red"][0]
        self.assertEqual(f["decls"][red_call[0]]["kind"], "init")

    def test_init_blocks_secondary_ctors_and_anonymous_objects(self):
        f = parse_kotlin("""
class A(x: Int) : B() {
    init { setup() }
    constructor() : this(compute()) { after() }
    fun f() {
        view.setOnClickListener(object : View.OnClickListener {
            override fun onClick(v: View) { handle(v) }
        })
    }
}
""")
        init = [d for d in f["decls"] if d["kind"] == "init"]
        self.assertEqual(len(init), 1)
        owner_of = {c[1]: f["decls"][c[0]] for c in f["calls"]}
        for n in ("setup", "compute", "after"):
            self.assertEqual(owner_of[n]["kind"], "init", n)
        self.assertEqual(owner_of["handle"]["name"], "onClick")
        on_click = decl(f, "onClick")
        self.assertEqual(f["decls"][on_click["parent"]]["name"], "f")
        self.assertIn([f["decls"].index(decl(f, "f")), "View.OnClickListener", 6, "obj"], f["trefs"])

    def test_raw_strings_and_comments_hide_fake_code(self):
        f = parse_kotlin('''
/* class Fake { fun fake() {} } */
val q = """
    class Ghost { fun ghost() { haunt() } }
""""
// fun commented() {}
fun real() = 1
''')
        self.assertEqual(names(f), ["q", "real"])
        self.assertEqual(calls(f), [])

    def test_resource_and_binding_references(self):
        f = parse_kotlin("""
fun f() {
    setContentView(R.layout.activity_main)
    val b = findViewById<Button>(R.id.ok)
    val c = android.R.id.content
    val bind = ActivityMainBinding.inflate(inflater)
}
""")
        self.assertEqual([r[1:3] for r in f["rrefs"]], [["layout", "activity_main"], ["id", "ok"]])
        self.assertEqual([b[1] for b in f["brefs"]], ["ActivityMainBinding"])

    def test_is_and_as_type_refs_and_typealias(self):
        f = parse_kotlin("typealias Handler = (Event) -> Unit\n"
                         "fun f(x: Any) { if (x is Foo) (x as? Bar)?.go() }\n")
        self.assertEqual(decl(f, "Handler")["type"], "(Event) -> Unit")
        self.assertEqual({(t[1], t[3]) for t in f["trefs"]}, {("Foo", "is"), ("Bar", "as")})

    def test_local_functions_and_ctor_inferred_locals(self):
        f = parse_kotlin("fun outer() {\n  fun helper(a: Int) = a\n  val r = Repo()\n  helper(r.size())\n}\n")
        helper = decl(f, "helper")
        self.assertEqual(f["decls"][helper["parent"]]["name"], "outer")
        self.assertEqual(decl(f, "outer")["locals"]["r"], ["Repo", "ctor"])
        self.assertEqual(sorted(calls(f)), ["Repo", "helper", "size"])

    def test_malformed_input_does_not_hang_or_raise(self):
        for src in ("class {", "fun (", "}}}", "val x = \"unterminated", '"""never closed',
                    "class A : { fun", "@", "fun f() = (((", "enum class E { A(, }"):
            parse_kotlin(src)


if __name__ == "__main__":
    unittest.main()
