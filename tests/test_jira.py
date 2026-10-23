"""Tests for the read-only Jira client, its ADF flattening, and the tool gating.

No network: agent_cli.jira._get is patched to return canned API payloads.
"""
import unittest
from unittest.mock import patch

from agent_cli.config import Config
from agent_cli.agent import render_tool_docs
from agent_cli import jira
from agent_cli.tools import Tools


def configured() -> Config:
    return Config(jira_url="https://x.atlassian.net", jira_email="a@b.c", jira_token="tok")


ISSUE_PAYLOAD = {
    "key": "PROJ-123",
    "fields": {
        "summary": "Login button does nothing",
        "status": {"name": "In Progress"},
        "issuetype": {"name": "Bug"},
        "priority": {"name": "High"},
        "assignee": {"displayName": "Dana Dev"},
        "reporter": {"displayName": "Quinn QA"},
        "labels": ["frontend", "urgent"],
        "description": {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Clicking login "},
                {"type": "text", "text": "has no effect."},
            ]},
        ]},
        "comment": {"comments": [
            {"author": {"displayName": "Quinn QA"},
             "body": {"type": "doc", "content": [
                 {"type": "paragraph", "content": [{"type": "text", "text": "Repro on Chrome."}]},
             ]}},
        ]},
    },
}

SEARCH_PAYLOAD = {
    "total": 2,
    "issues": [
        {"key": "PROJ-1", "fields": {"summary": "First", "status": {"name": "To Do"},
                                     "issuetype": {"name": "Task"}, "assignee": None}},
        {"key": "PROJ-2", "fields": {"summary": "Second", "status": {"name": "Done"},
                                     "issuetype": {"name": "Story"},
                                     "assignee": {"displayName": "Dana Dev"}}},
    ],
}


class ConfigGatingTest(unittest.TestCase):
    def test_is_configured(self):
        self.assertTrue(jira.is_configured(configured()))
        self.assertFalse(jira.is_configured(Config()))
        self.assertFalse(jira.is_configured(
            Config(jira_url="https://x.atlassian.net", jira_email="a@b.c")))  # no token

    def test_tools_hidden_until_configured(self):
        self.assertNotIn("jira_issue", render_tool_docs(Config()))
        docs = render_tool_docs(configured())
        self.assertIn("jira_issue", docs)
        self.assertIn("jira_search", docs)

    def test_render_without_cfg_is_backward_compatible(self):
        self.assertNotIn("jira_issue", render_tool_docs())


class AdfTest(unittest.TestCase):
    def test_paragraphs_and_breaks(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Line one"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Line two"}]},
        ]}
        self.assertEqual(jira.adf_to_text(doc), "Line one\nLine two")

    def test_none_and_plain_string(self):
        self.assertEqual(jira.adf_to_text(None), "")
        self.assertEqual(jira.adf_to_text("already text"), "already text")


class GetIssueTest(unittest.TestCase):
    def test_summary_has_key_status_description_and_comment(self):
        with patch("agent_cli.jira._get", return_value=ISSUE_PAYLOAD):
            out = jira.get_issue(configured(), "proj-123")  # lower-case normalised
        self.assertIn("PROJ-123: Login button does nothing", out)
        self.assertIn("Status: In Progress", out)
        self.assertIn("Assignee: Dana Dev", out)
        self.assertIn("Labels: frontend, urgent", out)
        self.assertIn("Clicking login has no effect.", out)
        self.assertIn("[Quinn QA] Repro on Chrome.", out)

    def test_key_is_uppercased_in_request(self):
        with patch("agent_cli.jira._get", return_value=ISSUE_PAYLOAD) as g:
            jira.get_issue(configured(), "proj-123")
        self.assertIn("issue/PROJ-123", g.call_args[0][1])


class SearchTest(unittest.TestCase):
    def test_lists_matches(self):
        with patch("agent_cli.jira._get", return_value=SEARCH_PAYLOAD):
            out = jira.search_issues(configured(), "project = PROJ")
        self.assertIn("2 of 2 issue(s)", out)
        self.assertIn("PROJ-1", out)
        self.assertIn("[Done]", out)
        self.assertIn("@unassigned", out)
        self.assertIn("@Dana Dev", out)

    def test_empty(self):
        with patch("agent_cli.jira._get", return_value={"total": 0, "issues": []}):
            out = jira.search_issues(configured(), "project = NOPE")
        self.assertIn("No issues match", out)

    def test_max_results_clamped(self):
        with patch("agent_cli.jira._get", return_value=SEARCH_PAYLOAD) as g:
            jira.search_issues(configured(), "project = PROJ", max_results=999)
        self.assertEqual(g.call_args[0][2]["maxResults"], 50)


class ToolDispatchTest(unittest.TestCase):
    def test_tool_returns_message_when_unconfigured(self):
        t = Tools(Config())
        out = t.run("jira_issue", {"key": "PROJ-1"})
        self.assertIn("not configured", out)

    def test_tool_reads_issue_when_configured(self):
        t = Tools(configured())
        with patch("agent_cli.jira._get", return_value=ISSUE_PAYLOAD):
            out = t.run("jira_issue", {"key": "PROJ-123"})
        self.assertIn("Login button does nothing", out)

    def test_tool_surfaces_jira_errors(self):
        t = Tools(configured())
        with patch("agent_cli.jira._get", side_effect=jira.JiraError("boom")):
            out = t.run("jira_search", {"jql": "x"})
        self.assertIn("Jira error: boom", out)


if __name__ == "__main__":
    unittest.main()
