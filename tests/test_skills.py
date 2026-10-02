"""Tests for the skills feature."""
import os
import tempfile
import unittest
from pathlib import Path

from agent_cli.config import Config
from agent_cli.skills import (
    Skill, _parse_front_matter, load_skill, load_skills,
    score_skill, select_skills, format_skill_context, list_skills_display,
)


def _make_skill_dir(root: Path, name: str, content: str) -> Path:
    """Create .agent/skills/<name>/SKILL.md with the given content."""
    d = root / ".agent" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content)
    return d


SAMPLE_SKILL = """\
---
name: kotlin-style
description: Kotlin coding conventions
triggers:
  - kotlin
  - style
  - convention
---
Use camelCase for functions. Use PascalCase for classes.
"""

SAMPLE_SKILL_2 = """\
---
name: testing
description: Unit testing guidelines
triggers:
  - test
  - junit
  - mock
---
Always use AAA pattern: Arrange, Act, Assert.
"""


class TestParseFrontMatter(unittest.TestCase):
    def test_basic(self):
        meta, body = _parse_front_matter(SAMPLE_SKILL)
        self.assertEqual(meta["name"], "kotlin-style")
        self.assertEqual(meta["description"], "Kotlin coding conventions")
        self.assertEqual(meta["triggers"], ["kotlin", "style", "convention"])
        self.assertIn("camelCase", body)

    def test_no_front_matter(self):
        meta, body = _parse_front_matter("Just some text.")
        self.assertEqual(meta, {})
        self.assertEqual(body, "Just some text.")

    def test_unclosed_front_matter(self):
        meta, body = _parse_front_matter("---\nname: foo\nno closing")
        self.assertEqual(meta, {})

    def test_empty_body(self):
        text = "---\nname: x\n---\n"
        meta, body = _parse_front_matter(text)
        self.assertEqual(meta["name"], "x")
        self.assertEqual(body.strip(), "")


class TestLoadSkill(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_load_valid_skill(self):
        d = _make_skill_dir(self.root, "kotlin-style", SAMPLE_SKILL)
        skill = load_skill(d)
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "kotlin-style")
        self.assertEqual(skill.triggers, ["kotlin", "style", "convention"])
        self.assertIn("camelCase", skill.body)

    def test_no_skill_md(self):
        d = self.root / ".agent" / "skills" / "empty"
        d.mkdir(parents=True)
        self.assertIsNone(load_skill(d))

    def test_empty_body_returns_none(self):
        d = _make_skill_dir(self.root, "empty-body", "---\nname: x\n---\n")
        self.assertIsNone(load_skill(d))

    def test_name_from_dir_if_missing(self):
        content = "---\ndescription: something\n---\nSome instructions."
        d = _make_skill_dir(self.root, "my-skill", content)
        skill = load_skill(d)
        self.assertEqual(skill.name, "my-skill")


class TestLoadSkills(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_no_skills_dir(self):
        self.assertEqual(load_skills(self.root), [])

    def test_loads_multiple(self):
        _make_skill_dir(self.root, "a", SAMPLE_SKILL)
        _make_skill_dir(self.root, "b", SAMPLE_SKILL_2)
        skills = load_skills(self.root)
        self.assertEqual(len(skills), 2)
        names = {s.name for s in skills}
        self.assertIn("kotlin-style", names)
        self.assertIn("testing", names)

    def test_skips_invalid(self):
        _make_skill_dir(self.root, "good", SAMPLE_SKILL)
        # Create a dir with no SKILL.md
        (self.root / ".agent" / "skills" / "bad").mkdir(parents=True)
        skills = load_skills(self.root)
        self.assertEqual(len(skills), 1)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.skill = Skill(
            name="kotlin-style",
            description="Kotlin coding conventions",
            triggers=["kotlin", "style", "convention"],
            body="Instructions here.",
            path=Path("/fake"),
        )

    def test_matching_trigger(self):
        self.assertGreater(score_skill(self.skill, "fix the kotlin style"), 0)

    def test_no_match(self):
        self.assertEqual(score_skill(self.skill, "deploy the server"), 0)

    def test_description_words_count(self):
        score = score_skill(self.skill, "what are the coding conventions")
        self.assertGreater(score, 0)

    def test_case_insensitive(self):
        self.assertGreater(score_skill(self.skill, "KOTLIN STYLE"), 0)

    def test_empty_message(self):
        self.assertEqual(score_skill(self.skill, ""), 0)


class TestSelectSkills(unittest.TestCase):
    def setUp(self):
        self.s1 = Skill("kotlin-style", "Kotlin coding conventions",
                        ["kotlin", "style"], "body1", Path("/f1"))
        self.s2 = Skill("testing", "Unit testing guidelines",
                        ["test", "junit", "mock"], "body2", Path("/f2"))

    def test_selects_matching(self):
        result = select_skills([self.s1, self.s2], "fix the kotlin style")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "kotlin-style")

    def test_selects_both_if_both_match(self):
        result = select_skills([self.s1, self.s2], "kotlin test style")
        self.assertEqual(len(result), 2)

    def test_returns_empty_on_no_match(self):
        result = select_skills([self.s1, self.s2], "deploy the server")
        self.assertEqual(result, [])

    def test_max_two(self):
        s3 = Skill("extra", "extra extra", ["kotlin", "test", "extra"],
                    "body3", Path("/f3"))
        result = select_skills([self.s1, self.s2, s3],
                               "kotlin test style extra")
        self.assertLessEqual(len(result), 2)


class TestTruncation(unittest.TestCase):
    def test_no_truncation(self):
        s = Skill("x", "", [], "short body", Path("/f"))
        self.assertEqual(s.truncated(100), "short body")

    def test_truncation_with_marker(self):
        s = Skill("x", "", [], "A" * 5000, Path("/f"))
        result = s.truncated(100)
        self.assertIn("...(skill truncated)...", result)
        self.assertLessEqual(len(result), 200)


class TestFormatSkillContext(unittest.TestCase):
    def test_includes_header_and_body(self):
        s = Skill("my-skill", "a description", [], "Do this.", Path("/f"))
        text = format_skill_context(s, 3000)
        self.assertIn("# Skill: my-skill", text)
        self.assertIn("a description", text)
        self.assertIn("Do this.", text)


class TestListSkillsDisplay(unittest.TestCase):
    def test_empty(self):
        text = list_skills_display([])
        self.assertIn("No skills loaded", text)

    def test_with_skills(self):
        s = Skill("foo", "does foo", ["bar", "baz"], "body", Path("/f"))
        text = list_skills_display([s])
        self.assertIn("foo", text)
        self.assertIn("does foo", text)
        self.assertIn("bar, baz", text)


class TestCoderSkillsIntegration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(f"cd {self.root} && git init -q && git config user.email test@test && git config user.name test")

    def tearDown(self):
        self.td.cleanup()

    def _make_coder(self):
        from agent_cli.coder import Coder
        cfg = Config(
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        return Coder(cfg, mode="code")

    def test_skills_loaded_at_init(self):
        _make_skill_dir(self.root, "s1", SAMPLE_SKILL)
        _make_skill_dir(self.root, "s2", SAMPLE_SKILL_2)
        coder = self._make_coder()
        self.assertEqual(len(coder.skills), 2)

    def test_no_skills_dir(self):
        coder = self._make_coder()
        self.assertEqual(len(coder.skills), 0)

    def test_skill_in_context_messages(self):
        _make_skill_dir(self.root, "s1", SAMPLE_SKILL)
        (self.root / "app.py").write_text("x = 1\n")
        coder = self._make_coder()
        coder.add(["app.py"])
        msgs = coder._context_messages("fix the kotlin style")
        skill_msgs = [m for m in msgs if "Skill: kotlin-style" in m.get("content", "")]
        self.assertTrue(len(skill_msgs) > 0)

    def test_no_skill_when_no_match(self):
        _make_skill_dir(self.root, "s1", SAMPLE_SKILL)
        coder = self._make_coder()
        msgs = coder._context_messages("deploy server to production")
        skill_msgs = [m for m in msgs if "Skill:" in m.get("content", "")]
        self.assertEqual(len(skill_msgs), 0)

    def test_pinned_skill_always_used(self):
        _make_skill_dir(self.root, "s1", SAMPLE_SKILL)
        coder = self._make_coder()
        coder._pinned_skill = coder.skills[0]
        msgs = coder._context_messages("deploy server to production")
        skill_msgs = [m for m in msgs if "Skill: kotlin-style" in m.get("content", "")]
        self.assertTrue(len(skill_msgs) > 0)

    def test_skill_budget_truncation(self):
        long_body = "A" * 5000
        content = f"---\nname: long\ntriggers:\n  - deploy\n---\n{long_body}"
        _make_skill_dir(self.root, "long", content)
        coder = self._make_coder()
        coder.cfg.skill_chars = 100
        coder._pinned_skill = coder.skills[0]
        msgs = coder._context_messages("deploy something")
        skill_msgs = [m for m in msgs if "Skill:" in m.get("content", "")]
        self.assertTrue(len(skill_msgs) > 0)
        self.assertIn("...(skill truncated)...", skill_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
