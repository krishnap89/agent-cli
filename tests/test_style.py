"""AGENT_STYLE rules in the coder and micro Editor system prompts."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_cli.coder import CODE_SYSTEM, Coder
from agent_cli.config import Config
from agent_cli.micro.roles import Editor
from agent_cli.style import FULL_RULES, LITE_RULES, style_name, style_rules, with_style


class TestStyleRules(unittest.TestCase):
    def test_full_rules_are_six_to_eight_lines(self):
        bullets = [ln for ln in FULL_RULES.splitlines() if ln.startswith("- ")]
        self.assertTrue(6 <= len(bullets) <= 8, len(bullets))

    def test_lite_is_shorter_and_keeps_the_core_rule(self):
        self.assertLess(len(LITE_RULES), len(FULL_RULES))
        self.assertIn("smallest change", LITE_RULES)

    def test_style_name_default_and_unknown_mean_full(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_STYLE", None)
            self.assertEqual(style_name(), "full")
        self.assertEqual(style_name("LITE"), "lite")
        self.assertEqual(style_name(" off "), "off")
        self.assertEqual(style_name("ultra"), "full")

    def test_env_var_is_read(self):
        with mock.patch.dict(os.environ, {"AGENT_STYLE": "lite"}):
            self.assertEqual(style_rules(), LITE_RULES)

    def test_with_style_off_leaves_prompt_unchanged(self):
        self.assertEqual(with_style("SYS", "off"), "SYS")
        self.assertIn(FULL_RULES, with_style("SYS", "full"))

    def test_rules_ask_for_shortcut_markers_and_risk_line(self):
        self.assertIn("shortcut: <limit>, <when to upgrade>", FULL_RULES)
        self.assertIn("Risk:", FULL_RULES)
        self.assertIn("Risk:", LITE_RULES)


class TestCoderPrompt(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _system(self, style, mode="code"):
        coder = Coder(Config(workdir=self.root, use_map=False, style=style), mode=mode)
        return coder.build_messages("hi")[0]["content"]

    def test_full_style_in_code_mode(self):
        sys_msg = self._system("full")
        self.assertTrue(sys_msg.startswith(CODE_SYSTEM.rstrip()))
        self.assertIn(FULL_RULES, sys_msg)

    def test_lite_style(self):
        sys_msg = self._system("lite")
        self.assertIn(LITE_RULES, sys_msg)
        self.assertNotIn("Prefer deleting code", sys_msg)

    def test_off_style_is_the_plain_prompt(self):
        self.assertEqual(self._system("off"), CODE_SYSTEM)

    def test_ask_mode_has_no_style_rules(self):
        self.assertNotIn("smallest change", self._system("full", mode="ask"))


class TestEditorPrompt(unittest.TestCase):
    def test_full_style_editor(self):
        ed = Editor(lambda m: None, style="full")
        self.assertIn(FULL_RULES, ed.system)
        self.assertIn("Output the blocks, then the Risk line, and nothing else.", ed.system)

    def test_off_style_editor_is_unchanged(self):
        self.assertEqual(Editor(lambda m: None, style="off").system, Editor.system)

    def test_editor_records_model_risk_line(self):
        reply = "a.py\n```\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```\nRisk: low"
        ed = Editor(lambda m: reply, style="full")
        from agent_cli.micro.roles import PlanStep
        (blocks, problems), _ = ed.run(step=PlanStep("set x", ["a.py"]), file_blocks="a.py\n")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(ed.last_risk, "low")

    def test_editor_includes_graph_facts(self):
        from agent_cli.micro.roles import PlanStep
        user = Editor(lambda m: None).build(step=PlanStep("s", ["a.kt"]), file_blocks="",
                                            graph_facts="Code graph facts:\n- X")
        self.assertIn("Code graph facts:\n- X", user)


if __name__ == "__main__":
    unittest.main()
