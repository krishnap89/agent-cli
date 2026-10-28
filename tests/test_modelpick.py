"""Tests for the per-role model picker (agent_cli/micro/modelpick)."""
import unittest

from agent_cli.config import Config
from agent_cli.micro.modelpick import parse_map, model_for, ROLES


class TestParseMap(unittest.TestCase):
    def test_empty_or_blank_is_empty(self):
        self.assertEqual(parse_map(""), {})
        self.assertEqual(parse_map("   "), {})

    def test_malformed_json_is_empty(self):
        self.assertEqual(parse_map("{not json"), {})
        self.assertEqual(parse_map("[1,2,3]"), {})  # not an object

    def test_keeps_known_roles_only(self):
        m = parse_map('{"Editor":"code","Nonsense":"x"}')
        self.assertEqual(m, {"Editor": "code"})

    def test_role_keys_case_insensitive(self):
        self.assertEqual(parse_map('{"editor":"code"}'), {"Editor": "code"})

    def test_drops_non_string_and_blank_values(self):
        m = parse_map('{"Editor":"code","Planner":123,"Fixer":""}')
        self.assertEqual(m, {"Editor": "code"})

    def test_default_key_kept(self):
        m = parse_map('{"default":"base","Editor":"code"}')
        self.assertEqual(m, {"default": "base", "Editor": "code"})

    def test_all_six_roles_assignable(self):
        obj = ", ".join(f'"{r}":"{r.lower()}-m"' for r in ROLES)
        m = parse_map("{" + obj + "}")
        self.assertEqual(set(m), set(ROLES))


class TestModelFor(unittest.TestCase):
    _MAP = ('{"Editor":"code-m","Fixer":"code-m","Planner":"reason-m",'
            '"Locator":"fast-m","Reviewer":"review-m"}')

    def test_auto_off_returns_single_model(self):
        cfg = Config(model="solo", auto_model=False, model_map=self._MAP)
        self.assertEqual(model_for(cfg, "Editor"), "solo")

    def test_auto_on_maps_each_role(self):
        cfg = Config(model="base", auto_model=True, model_map=self._MAP)
        self.assertEqual(model_for(cfg, "Editor"), "code-m")
        self.assertEqual(model_for(cfg, "Fixer"), "code-m")
        self.assertEqual(model_for(cfg, "Planner"), "reason-m")
        self.assertEqual(model_for(cfg, "Locator"), "fast-m")
        self.assertEqual(model_for(cfg, "Reviewer"), "review-m")

    def test_unmapped_role_falls_back_to_cfg_model(self):
        cfg = Config(model="base", auto_model=True, model_map=self._MAP)
        # Summarizer is not in the map and there is no default key.
        self.assertEqual(model_for(cfg, "Summarizer"), "base")

    def test_default_key_covers_unmapped_roles(self):
        cfg = Config(model="base", auto_model=True,
                     model_map='{"default":"default-m","Editor":"code-m"}')
        self.assertEqual(model_for(cfg, "Editor"), "code-m")
        self.assertEqual(model_for(cfg, "Summarizer"), "default-m")

    def test_blank_map_falls_back_to_cfg_model(self):
        cfg = Config(model="base", auto_model=True, model_map="")
        self.assertEqual(model_for(cfg, "Editor"), "base")


if __name__ == "__main__":
    unittest.main()
