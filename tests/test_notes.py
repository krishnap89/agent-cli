"""Tests for Feature 2: Project notes file."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli.coder import Coder
from agent_cli.notes import find_notes, load_notes, notes_exist, gather_init_info
from tests.fake_server import FakeServer


class TestFindNotes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_no_notes(self):
        self.assertIsNone(find_notes(self.root))

    def test_agent_md(self):
        (self.root / "AGENT.md").write_text("# Notes\n")
        self.assertEqual(find_notes(self.root), self.root / "AGENT.md")

    def test_conventions_md(self):
        (self.root / "CONVENTIONS.md").write_text("# Conv\n")
        self.assertEqual(find_notes(self.root), self.root / "CONVENTIONS.md")

    def test_dot_agent_notes(self):
        (self.root / ".agent").mkdir()
        (self.root / ".agent" / "notes.md").write_text("# Notes\n")
        self.assertEqual(find_notes(self.root), self.root / ".agent" / "notes.md")

    def test_priority_order(self):
        """AGENT.md takes priority over CONVENTIONS.md."""
        (self.root / "AGENT.md").write_text("agent\n")
        (self.root / "CONVENTIONS.md").write_text("conv\n")
        self.assertEqual(find_notes(self.root).name, "AGENT.md")

    def test_env_override(self):
        """AGENT_NOTES_FILE overrides the candidate list."""
        (self.root / "custom.md").write_text("custom\n")
        (self.root / "AGENT.md").write_text("agent\n")
        self.assertEqual(find_notes(self.root, "custom.md"), self.root / "custom.md")

    def test_env_override_missing(self):
        """AGENT_NOTES_FILE pointing to a missing file returns None."""
        self.assertIsNone(find_notes(self.root, "nonexistent.md"))


class TestLoadNotes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_no_file(self):
        path, text = load_notes(self.root)
        self.assertIsNone(path)
        self.assertEqual(text, "")

    def test_loads_content(self):
        (self.root / "AGENT.md").write_text("Hello world\n")
        path, text = load_notes(self.root)
        self.assertEqual(text, "Hello world\n")

    def test_truncation(self):
        content = "A" * 5000
        (self.root / "AGENT.md").write_text(content)
        path, text = load_notes(self.root, max_chars=100, warn_once=False)
        self.assertIn("...(truncated)...", text)
        self.assertTrue(len(text) < 200)


class TestCoderNotesIntegration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(f"cd {self.root} && git init -q && git config user.email test@test && git config user.name test")

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

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

    def test_notes_in_context_messages(self):
        """With AGENT.md present, context messages contain notes before repo map."""
        (self.root / "AGENT.md").write_text("Use snake_case everywhere.\n")
        (self.root / "app.py").write_text("x = 1\n")

        coder = self._make_coder(["ok"])
        coder.add(["app.py"])
        msgs = coder._context_messages("test query")

        # First pair should be the notes
        self.assertIn("Project notes from the developer", msgs[0]["content"])
        self.assertIn("Use snake_case everywhere", msgs[0]["content"])

    def test_notes_not_in_ask_mode_context(self):
        """Notes are included regardless of mode (they go in context, not system prompt)."""
        (self.root / "AGENT.md").write_text("Use snake_case.\n")
        coder = self._make_coder(["ok"])
        coder.mode = "ask"
        msgs = coder._context_messages("q")
        self.assertIn("Project notes", msgs[0]["content"])

    def test_notes_disabled(self):
        """AGENT_NOTES=0 disables notes."""
        (self.root / "AGENT.md").write_text("notes\n")
        coder = self._make_coder(["ok"])
        coder.cfg.notes_enabled = False
        msgs = coder._context_messages("q")
        for m in msgs:
            self.assertNotIn("Project notes", m.get("content", ""))

    def test_notes_reread_each_request(self):
        """Editing notes during a session affects the next request."""
        (self.root / "AGENT.md").write_text("Version 1\n")
        (self.root / "app.py").write_text("x = 1\n")

        coder = self._make_coder(["ok", "ok"])
        coder.add(["app.py"])

        msgs1 = coder._context_messages("q")
        self.assertIn("Version 1", msgs1[0]["content"])

        (self.root / "AGENT.md").write_text("Version 2\n")
        msgs2 = coder._context_messages("q")
        self.assertIn("Version 2", msgs2[0]["content"])


class TestAgentNotesIntegration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_notes_in_agent_tool_mode(self):
        """Agent system prompt contains notes in tool mode."""
        (self.root / "AGENT.md").write_text("Always use type hints.\n")
        from agent_cli.agent import Agent
        cfg = Config(workdir=self.root, use_map=False, notes_enabled=True)
        agent = Agent(cfg, chat_only=False)
        system = agent.messages[0]["content"]
        self.assertIn("# Project notes", system)
        self.assertIn("Always use type hints", system)

    def test_notes_not_in_chat_mode(self):
        """Agent system prompt does NOT contain notes in chat mode."""
        (self.root / "AGENT.md").write_text("Always use type hints.\n")
        from agent_cli.agent import Agent
        cfg = Config(workdir=self.root, use_map=False, notes_enabled=True)
        agent = Agent(cfg, chat_only=True)
        system = agent.messages[0]["content"]
        self.assertNotIn("Project notes", system)


class TestInitNotes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(f"cd {self.root} && git init -q && git config user.email test@test && git config user.name test")

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def test_init_notes_creates_file(self):
        """--init-notes with confirmation writes AGENT.md."""
        self.server = FakeServer(["## Overview\nA test project.\n"])
        self.server.start()
        cfg = Config(
            base_url=self.server.base_url,
            workdir=self.root,
            auto_approve=True,
            stream=False,
            use_map=False,
        )
        from agent_cli.coder import _do_init_notes
        with patch("agent_cli.coder.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True
            _do_init_notes(cfg)

        self.assertTrue((self.root / "AGENT.md").exists())
        content = (self.root / "AGENT.md").read_text()
        self.assertIn("Overview", content)

    def test_init_notes_refuses_existing(self):
        """--init-notes refuses when a notes file already exists."""
        (self.root / "AGENT.md").write_text("existing\n")
        cfg = Config(workdir=self.root, use_map=False)
        from agent_cli.coder import _do_init_notes
        _do_init_notes(cfg)
        # File should be unchanged
        self.assertEqual((self.root / "AGENT.md").read_text(), "existing\n")


if __name__ == "__main__":
    unittest.main()
