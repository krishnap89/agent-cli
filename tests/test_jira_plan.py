"""Tests for the Jira -> subtask-plan pipeline (agent_cli/micro/jira_plan).

The LLM is answered by a scripted FakeServer and the Jira fetch is mocked, so
no network is touched. The relevance gate and LLM keyword expansion are turned
off (env) so the only model calls are Locator then Planner, keeping the script
to two replies.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_cli.config import Config
from agent_cli.coder import Coder
from agent_cli.micro import JiraPlanOrchestrator
from agent_cli.micro.jira_plan import _issue_to_request
from tests.fake_server import FakeServer


FIELDS = {
    "key": "PROJ-42",
    "summary": "Add CSV export to the report page",
    "status": "To Do",
    "issuetype": "Story",
    "priority": "Medium",
    "labels": ["backend"],
    "description": "Users want to export the report table as CSV. Add a button "
                   "and a download endpoint.",
    "comments": ["Reuse the existing report query."],
}


class RequestCompositionTest(unittest.TestCase):
    def test_includes_key_summary_description_comments(self):
        req = _issue_to_request(FIELDS)
        self.assertIn("PROJ-42: Add CSV export to the report page", req)
        self.assertIn("export the report table as CSV", req)
        self.assertIn("Reuse the existing report query.", req)

    def test_tolerates_missing_description_and_comments(self):
        req = _issue_to_request({"key": "X-1", "summary": "Only a title",
                                 "description": "", "comments": []})
        self.assertEqual(req, "X-1: Only a title")


class JiraPlanRunTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        os.system(f"cd {self.root} && git init -q && git config user.email t@t "
                  f"&& git config user.name t")
        (self.root / "report.py").write_text("def report():\n    return rows\n")
        self._env = mock.patch.dict(os.environ, {
            "AGENT_ANALYSIS_GATE": "0",
            "AGENT_ANALYSIS_LLM_KEYWORDS": "0",
            "AGENT_JIRA_URL": "https://x.atlassian.net",
            "AGENT_JIRA_EMAIL": "a@b.c",
            "AGENT_JIRA_TOKEN": "tok",
        })
        self._env.start()

    def tearDown(self):
        self._env.stop()
        if hasattr(self, "server"):
            self.server.stop()
        self.td.cleanup()

    def _coder(self, script):
        self.server = FakeServer(script)
        self.server.start()
        cfg = Config(base_url=self.server.base_url, workdir=self.root,
                     auto_approve=True, stream=False, use_map=False,
                     jira_url="https://x.atlassian.net", jira_email="a@b.c",
                     jira_token="tok")
        return Coder(cfg, mode="code")

    def test_plans_subtasks_from_issue(self):
        # Locator picks report.py; Planner emits two STEP/FILES subtasks.
        locator_reply = "report.py"
        planner_reply = (
            "STEP: Add a CSV serializer for report rows\n"
            "FILES: report.py\n"
            "STEP: Add a download endpoint that returns the CSV\n"
            "FILES: report.py\n"
        )
        coder = self._coder([locator_reply, planner_reply])
        orch = JiraPlanOrchestrator(coder)
        with mock.patch("agent_cli.jira.get_issue_fields", return_value=FIELDS) as g:
            ok = orch.run("proj-42")
        self.assertTrue(ok)
        g.assert_called_once()
        # Planner saw the composed ticket request.
        planner_req = self.server.requests[-1]["messages"][-1]["content"]
        self.assertIn("Add CSV export to the report page", planner_req)

    def test_unconfigured_returns_false_without_calling_model(self):
        coder = self._coder([])
        coder.cfg.jira_token = ""  # break config after construction
        orch = JiraPlanOrchestrator(coder)
        ok = orch.run("PROJ-42")
        self.assertFalse(ok)
        self.assertEqual(self.server.requests, [])

    def test_jira_error_stops_cleanly(self):
        from agent_cli import jira
        coder = self._coder([])
        orch = JiraPlanOrchestrator(coder)
        with mock.patch("agent_cli.jira.get_issue_fields",
                        side_effect=jira.JiraError("404")):
            ok = orch.run("PROJ-99")
        self.assertFalse(ok)
        self.assertEqual(self.server.requests, [])


if __name__ == "__main__":
    unittest.main()
