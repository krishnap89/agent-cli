---
name: unit-test-style
description: How to write unit tests in this project
triggers:
  - test
  - tests
  - unit test
  - junit
  - mock
  - test-gen
---

## 1. Libraries

All tests use the **Python standard library only** — no pytest, no third-party
assertion or mocking libraries.

- Framework: `unittest.TestCase`
- Mocking: `unittest.mock.patch`, `unittest.mock.patch.dict`
- File isolation: `tempfile.TemporaryDirectory`
- Paths: `pathlib.Path`
- Integration: `tests.fake_server.FakeServer` (custom OpenAI-compatible HTTP server)

## 2. File layout

- Tests live in `tests/` at the project root.
- One file per module: `tests/test_<module>.py` (e.g. `test_review.py`, `test_skills.py`).
- Each file starts with a module docstring, then stdlib imports, then project imports.
- `tests/__init__.py` exists (empty).

## 3. Test naming

- **Classes**: `Test<Feature>(unittest.TestCase)` — PascalCase, `Test` prefix.
- **Methods**: `test_snake_case_descriptive` — e.g. `test_staged`, `test_no_match`,
  `test_review_sends_diff_and_untracked`.
- Many test methods have a **one-line docstring** explaining the scenario.
- No backtick names, no given/when/then naming, no parameterized tests.

## 4. Structure

### setUp / tearDown

```python
def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.root = Path(self.td.name)

def tearDown(self):
    self.td.cleanup()
```

Integration tests also init a git repo in `setUp` and stop `FakeServer` in `tearDown`:

```python
os.system(f"cd {self.root} && git init -q && git config user.email test@test && git config user.name test")
```

```python
def tearDown(self):
    if hasattr(self, "server"):
        self.server.stop()
    self.td.cleanup()
```

### Arrange / Act / Assert

Tests follow AAA but without labeling comments. Setup writes files to `self.root`,
act calls the function under test, assert uses `self.assert*`.

## 5. Mocking and fakes

- **FakeServer** (`tests/fake_server.FakeServer`): takes a list of scripted reply
  strings, records all requests for assertion via `self.server.requests`.
- `patch.dict(os.environ, {...})` for env vars.
- `patch.object(Confirm, "ask", return_value=True/False)` for interactive prompts.
- File-system fakes: helper functions create fake project structures in temp dirs
  (e.g. `_write_fake_gradlew`, `_setup_android_project`, `_make_skill_dir`).
- These helpers are **module-local** (defined at top of each test file), not shared.

### Integration test helper pattern

```python
def _make_coder(self, script):
    self.server = FakeServer(script)
    self.server.start()
    cfg = Config(
        base_url=self.server.base_url,
        workdir=self.root,
        auto_approve=True,
        stream=False,
        use_map=False,
    )
    return Coder(cfg, mode="code")
```

## 6. Assertions

Standard `unittest` assertions only:

- `self.assertEqual`, `self.assertTrue`, `self.assertFalse`
- `self.assertIn`, `self.assertNotIn`
- `self.assertIsNone`, `self.assertIsNotNone`
- `self.assertGreater`, `self.assertLessEqual`
- `self.assertRaises(ValueError)` as context manager

No `assertThat`, `shouldBe`, or third-party assertion libraries.

## 7. Examples

### Unit test (pure function)

From `test_skills.py` — testing front matter parsing:

```python
class TestParseFrontMatter(unittest.TestCase):
    def test_basic(self):
        meta, body = _parse_front_matter(SAMPLE_SKILL)
        self.assertEqual(meta["name"], "kotlin-style")
        self.assertEqual(meta["triggers"], ["kotlin", "style", "convention"])
        self.assertIn("camelCase", body)

    def test_no_front_matter(self):
        meta, body = _parse_front_matter("Just some text.")
        self.assertEqual(meta, {})
        self.assertEqual(body, "Just some text.")
```

### Integration test (Coder + FakeServer)

From `test_review.py` — testing review sends correct data:

```python
def test_review_sends_diff_and_untracked(self):
    """Review of uncommitted changes sends both tracked diff and untracked file."""
    (self.root / "a.py").write_text("x = 1\n")
    os.system(f"cd {self.root} && git add a.py && git commit -q -m 'add a'")
    (self.root / "a.py").write_text("x = 2\n")
    (self.root / "b.py").write_text("y = 1\n")

    coder = self._make_coder(["1. [low] a.py:1 - Minor\n   Problem: Minor.\n   Fix: Ok."])
    with patch.object(Confirm, "ask", return_value=False):
        coder.review()

    sent = self.server.requests[0]["messages"]
    self.assertIn("expert code reviewer", sent[0]["content"])
    self.assertIn("b.py", sent[1]["content"])
```

## 8. Don'ts

- **No pytest** — everything uses `unittest.TestCase`.
- **No third-party test deps** — no pytest, assertj, hypothesis, etc.
- **No shared test fixtures** — helpers are defined per test file, not in conftest or a shared module (except `FakeServer`).
- **No `@patch` decorator style** — `patch` and `patch.object` are used as context managers (`with`), not decorators.
- **No real network calls** — all HTTP goes through `FakeServer`.
- **No real Gradle** — fake `gradlew` shell scripts simulate build output.
