"""Tests for masking sensitive data before it reaches the LLM (agent_cli/redact.py)."""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from agent_cli import llm, redact
from agent_cli.config import Config
from agent_cli.editblock import apply_edit, parse_edit_blocks
from tests.fake_server import FakeServer

FIXTURE = Path(__file__).parent / "fixtures" / "redact" / "sample.txt"

SECRETS = [
    "MIIEowIBAAKCAQEAfakefakefakefakefakefakefakefakefakefakefakefake",
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABFAKEFAKE",
    "AK§IAIOSFODNN7EXAMPLE",
    "wJa§lrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "gh§p_FAKEaaaaBBBBccccDDDDeeeeFFFFgggg1234",
    "github§_pat_11FAKEFAKE0123456789_abcdefghijklmnopqrstuvwxyz",
    "gl§pat-FAKEfake1234567890ab",
    "xo§xb-1234567890-FAKEFAKEFAKE",
    "T00FAKE00/B00FAKE00/FAKEfakeFAKEfake1234",
    "AI§zaSyFAKE_fake-FAKEfakeFAKEfakeFAKE123",
    "sk§_test_FAKEfakeFAKEfake1234567890",
    "AT§ATT3xFfGF0FAKEfakeFAKEfake-1234567890",
    "sk§-proj-FAKEfakeFAKEfakeFAKEfake1234",
    "sk§-ant-api03-FAKEfakeFAKEfakeFAKEfake",
    "ey§JhbGciOiJIUzI1NiJ9.ey§JzdWIiOiIxMjM0NTY3ODkwIn0.FAKEsignatureFAKEsig",
    "Pg5ecret42",
    "dXNlcjpGQUtFcGFzcw==",
    "FAKEbearerTOKEN1234567890",
    "cs-FAKE-9f8e7d6c5b",
    "kotlinFakeKey123",
    "pyFakePass!9",
    "envFakePass77",
    "envFakeToken88",
    "propsStorePw1",
    "propsKeyPw2",
    "yamlFakePass3",
    "groovyStorePw4",
    "mapsFakeKey5",
]
# Token-shaped literals are split with "§" (see the fixture header) so that
# GitHub push protection does not flag these fake values.
SECRETS = [v.replace("§", "") for v in SECRETS]


def _fixture() -> str:
    return FIXTURE.read_text().replace("§", "")


def _pem(body: str, kind: str = "") -> str:
    """A fake PEM private-key block, assembled so the file holds no literal header."""
    label = (kind + " " if kind else "") + "PRIVATE" + " KEY"
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"

PERSONAL = [
    "max.mustermann@firma-beispiel.de",
    "4539 1488 0343 6467",
    "DE89 3704 0044 0532 0130 00",
    "GB82 WEST 1234 5698 7654 32",
    "+49 170 1234567",
    "+1 (415) 555-0132",
    "030 12345678",
    "0151 23456789",
    "86095742719",
    "85.214.132.117",
    "2a00:1450:4001:82a::200e",
    "Erika", "Mustermann", "1964-08-12", "Heidestrasse 17, 51147 Koeln",
    "anna.schmidt@firma-beispiel.de", "+49 151 99887766", "Anna",
    "Lukas", "lukas.weber@firma-beispiel.de",
    "Hoffmann",
    "Jonas Becker",
]

MUST_STAY = [
    "password: String",
    "${API_TOKEN}",
    "$SECRET_FROM_ENV",
    "<your-api-key>",
    '"****"',
    '"changeme"',
    '"allowedTestToken"',
    "info@example.com",
    "version 1.2.3.4",
    "timestamp 1700000000000",
    "127.0.0.1",
]

KOTLIN = '''package com.example.login

import kotlinx.coroutines.flow.MutableStateFlow

data class Credentials(val username: String, val password: String)

class LoginViewModel(private val repo: AuthRepository) : ViewModel() {
    private val _state = MutableStateFlow<LoginState>(LoginState.Idle)
    var password: String = ""
    val tokenType = "Bearer"

    companion object {
        const val KEY_PASSWORD = "key_password"
        const val PREF_AUTH_TOKEN = "pref_auth_token"
        private const val MAX_RETRIES = 3
    }

    fun login(username: String, password: String): Boolean {
        val token = repo.token
        password = newPassword
        val items = list.map(::toUi)
        val version = "1.2.3"
        return repo.login(Credentials(username, password))
    }

    fun onPasswordChanged(password: String) {
        _state.value = LoginState.Editing(password = password)
    }
}
'''


class _Base(unittest.TestCase):
    def setUp(self):
        redact.reset_sessions()
        self.reports = []
        redact.set_reporter(lambda msg, level: self.reports.append((level, msg)))
        self.tmp = Path(tempfile.mkdtemp())
        self._env = {k: os.environ.pop(k) for k in
                     ("AGENT_REDACT", "AGENT_REDACT_PII", "AGENT_REDACT_STRICT") if k in os.environ}

    def tearDown(self):
        redact.set_reporter(None)
        redact.reset_sessions()
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k in ("AGENT_REDACT", "AGENT_REDACT_PII", "AGENT_REDACT_STRICT"):
            os.environ.pop(k, None)
        os.environ.update(self._env)

    def write_config(self, **cfg):
        (self.tmp / ".agent").mkdir(exist_ok=True)
        (self.tmp / ".agent" / "redact.json").write_text(json.dumps(cfg))

    def red(self, **cfg) -> redact.Redactor:
        self.write_config(allow=["allowedTestToken"], **cfg)
        (self.tmp / ".agent" / "redact-names.txt").write_text("# names\nJonas Becker\n")
        return redact.get(self.tmp)


class TestFixture(_Base):
    def test_all_secrets_masked_and_restored(self):
        red = self.red()
        text = _fixture()
        masked = red.mask_text(text)
        for value in SECRETS:
            self.assertIn(value, text)
            self.assertNotIn(value, masked, f"secret not masked: {value}")
        for value in MUST_STAY:
            self.assertIn(value, masked, f"wrongly masked: {value}")
        # personal data is off by default
        self.assertIn("max.mustermann@firma-beispiel.de", masked)
        self.assertEqual(red.unmask_text(masked), text)

    def test_personal_data_masked_when_enabled(self):
        red = self.red(pii=True)
        text = _fixture()
        masked = red.mask_text(text)
        for value in SECRETS + PERSONAL:
            self.assertNotIn(value, masked, f"not masked: {value}")
        for value in MUST_STAY:
            self.assertIn(value, masked, f"wrongly masked: {value}")
        self.assertEqual(red.unmask_text(masked), text)

    def test_env_var_enables_pii(self):
        os.environ["AGENT_REDACT_PII"] = "1"
        red = self.red()
        self.assertNotIn("max.mustermann@firma-beispiel.de", red.mask_text(_fixture()))

    def test_only_the_value_is_replaced(self):
        red = self.red()
        self.assertEqual(red.mask_text('password = "hunter2xyz"'),
                         'password = "[REDACTED_PASSWORD_1]"')

    def test_masking_is_idempotent(self):
        red = self.red(pii=True)
        once = red.mask_text(_fixture())
        self.assertEqual(red.mask_text(once), once)

    def test_known_value_masked_without_context(self):
        red = self.red()
        red.mask_text("DB_PASSWORD=envFakePass77")
        self.assertEqual(red.mask_text("login failed for envFakePass77 (bad)"),
                         "login failed for [REDACTED_PASSWORD_1] (bad)")

    def test_custom_pattern(self):
        red = self.red(patterns=[{"name": "ticket secret", "regex": r"TS-(\d{6})", "group": 1}])
        self.assertEqual(red.mask_text("ref TS-123456"), "ref TS-[REDACTED_TICKET_SECRET_1]")

    def test_disabled(self):
        os.environ["AGENT_REDACT"] = "0"
        red = self.red()
        text = _fixture()
        self.assertEqual(red.mask_text(text), text)
        redact.startup_notice(Config(workdir=self.tmp))
        self.assertTrue(any(level == "warn" and "OFF" in msg for level, msg in self.reports))


class TestKotlinUnchanged(_Base):
    def test_normal_kotlin_code_unchanged(self):
        for pii in (False, True):
            redact.reset_sessions()
            red = self.red(pii=pii)
            self.assertEqual(red.mask_text(KOTLIN), KOTLIN)
            self.assertEqual(red.mask_text(KOTLIN, path="app/LoginViewModel.kt"), KOTLIN)


class TestValidators(_Base):
    def test_checksums(self):
        self.assertTrue(redact.luhn_ok("4539148803436467"))
        self.assertFalse(redact.luhn_ok("4539148803436468"))
        self.assertTrue(redact.iban_ok("DE89 3704 0044 0532 0130 00"))
        self.assertFalse(redact.iban_ok("DE89 3704 0044 0532 0130 01"))
        self.assertFalse(redact.iban_ok("DE89 3704 0044 0532 0130"))  # wrong length for DE
        self.assertTrue(redact.de_tax_id_ok("86095742719"))
        self.assertFalse(redact.de_tax_id_ok("86095742718"))

    def test_context_lowers_and_raises_score(self):
        red = self.red(pii=True)
        self.assertEqual(red.mask_text("released version 85.214.132.117"),
                         "released version 85.214.132.117")
        self.assertNotIn("85.214.132.117", red.mask_text("client 85.214.132.117"))
        # a landline without context scores below the threshold, with context above
        self.assertIn("030 12345678", red.mask_text("ref 030 12345678"))
        self.assertNotIn("030 12345678", red.mask_text("Telefon 030 12345678"))

    def test_luhn_failing_number_kept(self):
        red = self.red(pii=True)
        self.assertIn("4539 1488 0343 6468", red.mask_text("card 4539 1488 0343 6468"))

    def test_test_data_paths_skip_personal_data(self):
        red = self.red(pii=True)
        text = "val email = \"max.mustermann@firma-beispiel.de\"\nval password = \"hunter2xyz\"\n"
        masked = red.mask_text(text, path="app/src/test/java/LoginTest.kt")
        self.assertIn("max.mustermann@firma-beispiel.de", masked)
        self.assertNotIn("hunter2xyz", masked)  # secrets are masked even in tests
        self.assertNotIn("max.mustermann@firma-beispiel.de", red.mask_text(text, path="app/src/main/A.kt"))

    def test_file_blocks_in_prompt_get_their_path(self):
        red = self.red(pii=True)
        prompt = ("app/src/test/Fixture.kt\n```kotlin\nval e = \"anna.schmidt@firma-beispiel.de\"\n```\n"
                  "and mail lukas.weber@firma-beispiel.de")
        masked = red.mask_text(prompt)
        self.assertIn("anna.schmidt@firma-beispiel.de", masked)
        self.assertNotIn("lukas.weber@firma-beispiel.de", masked)


class TestStablePlaceholders(_Base):
    def test_same_value_same_placeholder_across_requests(self):
        red = self.red()
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": 'DB_PASSWORD=envFakePass77\ntoken = "tokFake12345"'}]
        first, counts = red.mask_messages(msgs)
        msgs.append({"role": "assistant", "content": "I see envFakePass77."})
        msgs.append({"role": "user", "content": [{"type": "text", "text": "use tokFake12345"}]})
        second, _ = red.mask_messages(msgs)
        self.assertEqual(first[1], second[1])   # identical prefix -> prefix caching works
        self.assertEqual(second[2]["content"], "I see [REDACTED_PASSWORD_1].")
        self.assertEqual(second[3]["content"][0]["text"], "use [REDACTED_TOKEN_1]")
        self.assertEqual(counts, {"PASSWORD": 1, "TOKEN": 1})
        self.assertEqual(msgs[1]["content"], 'DB_PASSWORD=envFakePass77\ntoken = "tokFake12345"')

    def test_end_to_end_through_llm_chat(self):
        self.red()
        log = self.tmp / "http.log"
        seen = []

        def reply(messages):
            seen.append(messages[-1]["content"])
            return "The password is [REDACTED_PASSWORD_1]."

        with FakeServer([reply, reply]) as srv:
            for stream in (False, True):
                cfg = Config(workdir=self.tmp, base_url=srv.base_url, model="m", stream=stream,
                             retries=0, http_log=str(log))
                out = llm.chat(cfg, [{"role": "user", "content": "DB_PASSWORD=envFakePass77"}])
                self.assertEqual(out.content, "The password is envFakePass77.")
        self.assertEqual(seen, ["DB_PASSWORD=[REDACTED_PASSWORD_1]"] * 2)
        self.assertNotIn("envFakePass77", log.read_text())
        self.assertIn(("info", "masked before sending: 1 password"), self.reports)


class TestEditBlocks(_Base):
    def test_edit_around_placeholder_applies_to_real_file(self):
        red = self.red()
        real = 'object Config {\n    const val DB_PASSWORD = "S3cretDbPw!"\n    const val TIMEOUT = 30\n}\n'
        (self.tmp / "Config.kt").write_text(real)
        prompt = f"Config.kt\n```kotlin\n{real}```\n"
        masked = red.mask_text(prompt)
        self.assertNotIn("S3cretDbPw!", masked)
        model_reply = (
            "Config.kt\n```kotlin\n<<<<<<< SEARCH\n"
            '    const val DB_PASSWORD = "[REDACTED_PASSWORD_1]"\n    const val TIMEOUT = 30\n'
            "=======\n"
            '    const val DB_PASSWORD = "[REDACTED_PASSWORD_1]"\n    const val TIMEOUT = 60\n'
            ">>>>>>> REPLACE\n```\n")
        blocks, problems = parse_edit_blocks(red.unmask_text(model_reply))
        self.assertEqual(problems, [])
        new = apply_edit(real, blocks[0].search, blocks[0].replace)
        self.assertEqual(new, real.replace("TIMEOUT = 30", "TIMEOUT = 60"))
        self.assertIn('"S3cretDbPw!"', new)

    def test_json_tool_call_values_are_escaped(self):
        red = self.red()
        key = _pem("AAAAfakeBODYline1\nBBBBfakeBODYline2", "RSA")
        masked = red.mask_text(key)
        ph = redact.PLACEHOLDER_RE.search(masked).group(0)
        call = '<tool_call>{"name": "write_file", "arguments": {"path": "k", "content": "' + ph + '"}}</tool_call>'
        restored = red.unmask_text(call)
        body = json.loads(restored[len("<tool_call>"):-len("</tool_call>")])
        self.assertEqual(body["arguments"]["content"], "AAAAfakeBODYline1\nBBBBfakeBODYline2")
        args = red.unmask_arguments('{"content": "' + ph + '"}')
        self.assertEqual(json.loads(args)["content"], "AAAAfakeBODYline1\nBBBBfakeBODYline2")
        self.assertEqual(red.unmask_arguments({"content": ph})["content"],
                         "AAAAfakeBODYline1\nBBBBfakeBODYline2")


class TestStreaming(_Base):
    def test_split_placeholder_is_buffered(self):
        red = self.red()
        red.mask_text('password = "hunter2xyz"')
        un = redact.StreamUnmasker(red)
        pieces = ['pass is [RED', 'ACTED_PASS', 'WORD_1', '] ok [', 'not one] [REDACTED_X']
        out = [un.feed(p) for p in pieces]
        self.assertEqual(out[0], "pass is ")
        self.assertEqual(out[1], "")
        self.assertEqual(out[3], "hunter2xyz ok ")
        self.assertEqual("".join(out) + un.flush(), "pass is hunter2xyz ok [not one] [REDACTED_X")

    def test_iter_stream_events_unmasks_deltas(self):
        red = self.red()
        red.mask_text('password = "hunter2xyz"')
        lines = []
        for piece in ["The value is [REDACT", "ED_PASSWORD", "_1]", "."]:
            lines.append("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}))
        lines.append("data: " + json.dumps({"choices": [{"delta": {
            "tool_calls": [{"index": 0, "function": {"name": "f", "arguments": '{"p": "[REDACTED_PASS'}}]}}]}))
        lines.append("data: " + json.dumps({"choices": [{"delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": 'WORD_1]"}'}}]}, "finish_reason": "stop"}]}))
        lines.append("data: [DONE]")
        cfg = Config(workdir=self.tmp)
        events = list(llm.iter_stream_events(cfg, [(ln + "\n").encode() for ln in lines]))
        content = "".join(v for k, v in events if k == "content")
        self.assertEqual(content, "The value is hunter2xyz.")
        args = "".join(v[2] or "" for k, v in events if k == "tool")
        self.assertEqual(json.loads(args), {"p": "hunter2xyz"})
        reply = llm._read_stream(cfg, [(ln + "\n").encode() for ln in lines], None)
        self.assertEqual(reply.content, "The value is hunter2xyz.")
        self.assertEqual(json.loads(reply.tool_calls[0]["arguments"]), {"p": "hunter2xyz"})


class TestFilePolicy(_Base):
    def setUp(self):
        super().setUp()
        for name in ("release.jks", "debug.keystore", "cert.p12", "server.pem", "tls.key", "id_rsa",
                     ".env", ".env.local", "local.properties", "gradle.properties",
                     "keystore.properties", "google-services.json", "GoogleService-Info.plist",
                     "Main.kt", "notes/private.txt"):
            p = self.tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("API_TOKEN=envFakeToken88\n")
        (self.tmp / "secret_in_disguise.txt").write_text(_pem("abc"))
        (self.tmp / ".agentignore").write_text("# never send\nnotes/\n*.secret\n!keep.secret\n")
        (self.tmp / "a.secret").write_text("x")
        (self.tmp / "keep.secret").write_text("x")

    def test_policies(self):
        red = redact.get(self.tmp)
        for name in ("release.jks", "debug.keystore", "cert.p12", "server.pem", "tls.key", "id_rsa",
                     "notes/private.txt", "a.secret", "secret_in_disguise.txt"):
            self.assertEqual(red.file_policy(name).action, "block", name)
        for name in (".env", ".env.local", "local.properties", "gradle.properties",
                     "keystore.properties", "google-services.json", "GoogleService-Info.plist"):
            self.assertEqual(red.file_policy(name).action, "warn", name)
        for name in ("Main.kt", "keep.secret", "id_rsa.pub"):
            self.assertEqual(red.file_policy(name).action, "ok", name)

    def test_warning_once(self):
        red = redact.get(self.tmp)
        self.assertIsNone(red.check_file(".env"))
        self.assertIsNone(red.check_file(".env"))
        self.assertEqual(sum(1 for lvl, _ in self.reports if lvl == "warn"), 1)

    def test_read_file_tool_refuses(self):
        from agent_cli.tools import ToolError, Tools
        tools = Tools(Config(workdir=self.tmp, auto_approve=True))
        with self.assertRaises(ToolError) as cm:
            tools.t_read_file("release.jks")
        self.assertIn("never sent", str(cm.exception))
        with self.assertRaises(ToolError):
            tools.t_read_file("notes/private.txt")
        self.assertIn("envFakeToken88", tools.t_read_file(".env"))  # masked later, in llm.py

    def test_mentions_refuse(self):
        from agent_cli.mentions import expand
        text, notes = expand("look at @server.pem and @Main.kt", self.tmp)
        self.assertIn("refused", notes[0])
        self.assertNotIn("--- server.pem ---", text)
        self.assertIn("--- Main.kt ---", text)

    def test_add_refuses(self):
        from agent_cli.coder import Coder
        coder = Coder(Config(workdir=self.tmp, use_map=False))
        with redirect_stdout(io.StringIO()):
            coder.add(["release.jks", "Main.kt", "id_rsa"])
        self.assertEqual(list(coder.editable), ["Main.kt"])

    def test_review_refuses(self):
        from agent_cli.review import collect_material
        material, files, _ = collect_material(self.tmp, "tls.key Main.kt")
        self.assertIn("refused", material)
        self.assertNotIn("tls.key\n", material.replace("tls.key: refused", ""))
        diff = ("diff --git a/tls.key b/tls.key\n--- a/tls.key\n+++ b/tls.key\n@@ -1 +1 @@\n-old\n+KEYDATA\n"
                "diff --git a/Main.kt b/Main.kt\n--- a/Main.kt\n+++ b/Main.kt\n@@ -1 +1 @@\n-a\n+b\n")
        filtered, refused = redact.filter_diff(diff, self.tmp)
        self.assertEqual(refused, ["tls.key"])
        self.assertNotIn("KEYDATA", filtered)
        self.assertIn("+b", filtered)

    def test_scan_shows_policy_and_four_chars(self):
        (self.tmp / "app.properties").write_text("x=1\nstorePassword=propsStorePw1\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            redact.scan_main(["app.properties", "release.jks", "--dir", str(self.tmp)])
        out = buf.getvalue()
        self.assertIn("release.jks: BLOCKED", out)
        self.assertIn("app.properties: sent masked", out)
        self.assertIn("2  PASSWORD", out)
        self.assertIn("prop…", out)
        self.assertNotIn("propsStorePw1", out)


class TestStrictMode(_Base):
    def test_local_model_findings_are_masked_too(self):
        red = self.red(strict=True)
        self.assertTrue(red.pii)
        log = "ERROR order failed for customer Hans Meier (DB_PASSWORD=envFakePass77)\n" + "x" * 600
        redact.note_untrusted(log, "log")
        sent = []

        def fake_chat(cfg, messages, **kw):
            sent.append(messages[-1]["content"])
            return llm.Reply("Hans Meier\nsomething not in the text", "stop", "")

        cfg = Config(workdir=self.tmp, model="big", model_map='{"pii": "small-local"}')
        with mock.patch.object(llm, "chat", side_effect=lambda c, m, **kw: (
                self.assertEqual(c.model, "small-local"), fake_chat(c, m, **kw))[1]):
            red.run_strict(cfg)
        self.assertNotIn("envFakePass77", sent[0])   # the PII model sees rule-masked text
        masked = red.mask_text(log)
        self.assertNotIn("Hans Meier", masked)
        self.assertNotIn("envFakePass77", masked)

    def test_failure_keeps_rules(self):
        red = self.red(strict=True)
        redact.note_untrusted("DB_PASSWORD=envFakePass77\n" + "y" * 600, "paste")
        with mock.patch.object(llm, "chat", side_effect=llm.LLMError("down")):
            red.run_strict(Config(workdir=self.tmp))
        self.assertTrue(any("strict PII check failed" in m for _, m in self.reports))
        self.assertNotIn("envFakePass77", red.mask_text("DB_PASSWORD=envFakePass77"))


class TestRunLog(_Base):
    def test_run_log_is_masked(self):
        from agent_cli.micro.loop import RunLogger
        self.red()
        logger = RunLogger(self.tmp)
        logger.log("Editor", "", 'val password = "hunter2xyz"', 0.1)
        self.assertNotIn("hunter2xyz", logger.path.read_text())


class TestReviewFixes(_Base):
    """Regressions for the leaks and gaps found in review."""

    KEY = _pem("MIIfakeKEYbodyLINEone\nMIIfakeKEYbodyLINEtwo")

    def test_json_escaped_known_values_are_masked(self):
        red = self.red(pii=True)
        red.mask_text(self.KEY)
        red.mask_text('password = "pa\\"ss!w0rd"')       # value with a backslash-quote
        red.learn("Jürgen Müller", "NAME")
        history = json.dumps({"name": "write_file", "arguments": {"content": self.KEY}})
        self.assertNotIn("MIIfakeKEYbody", red.mask_text("<tool_call>\n" + history + "\n</tool_call>"))
        self.assertNotIn("M\\u00fcller", red.mask_text(json.dumps({"who": "Jürgen Müller"})))
        self.assertNotIn("Müller", red.mask_text(json.dumps({"who": "Jürgen Müller"}, ensure_ascii=False)))

    def test_escaped_private_key_found_on_first_sight(self):
        red = self.red()
        text = json.dumps({"content": self.KEY})
        masked = red.mask_text(text)
        self.assertNotIn("MIIfakeKEYbody", masked)
        self.assertEqual(red.unmask_text(masked), text)

    def test_filter_diff_paths_with_spaces_and_quotes(self):
        diff = ('diff --git a/my keys/server.pem b/my keys/server.pem\n--- a/my keys/server.pem\n'
                '+++ b/my keys/server.pem\n@@ -1 +1 @@\n-old\n+SECRETBODY1\n'
                'diff --git "a/caf\\303\\251.key" "b/caf\\303\\251.key"\n--- "a/caf\\303\\251.key"\n'
                '+++ "b/caf\\303\\251.key"\n@@ -1 +1 @@\n-x\n+SECRETBODY2\n'
                'diff --git a/old.txt b/new.pem\nrename from old.txt\nrename to new.pem\n')
        filtered, refused = redact.filter_diff(diff, self.tmp)
        self.assertNotIn("SECRETBODY", filtered)
        self.assertEqual(sorted(refused), sorted(["my keys/server.pem", "café.key", "new.pem"]))
        self.assertEqual(redact.diff_paths("diff --git a/x y.kt b/x y.kt\n"), ["x y.kt"])

    def test_added_yaml_line_in_diff_is_masked(self):
        red = self.red(pii=True)
        diff = ("diff --git a/application.yml b/application.yml\n@@ -1,2 +1,2 @@\n"
                "-  password: oldS3cretValue\n+  password: newS3cretValue\n"
                "+  last_name: Hoffmann\n")
        masked = red.mask_text(diff)
        for v in ("oldS3cretValue", "newS3cretValue", "Hoffmann"):
            self.assertNotIn(v, masked)

    def test_blocked_embedded_file_withheld_even_when_off(self):
        os.environ["AGENT_REDACT"] = "0"
        red = self.red()
        prompt = "keys/release.pem\n```\nPEMBODY\n```\nand\n--- id_rsa ---\nRSABODY\n--- end id_rsa ---"
        msgs, _ = red.mask_messages([{"role": "user", "content": prompt}])
        self.assertNotIn("PEMBODY", msgs[0]["content"])
        self.assertNotIn("RSABODY", msgs[0]["content"])
        self.assertIn("withheld", msgs[0]["content"])

    def test_run_command_naming_blocked_file_refused(self):
        from agent_cli.tools import ToolError, Tools
        (self.tmp / "release.jks").write_text("x")
        tools = Tools(Config(workdir=self.tmp, auto_approve=True))
        with self.assertRaises(ToolError):
            tools.t_run_command("cat release.jks")
        self.assertIn("exit code: 0", tools.t_run_command("echo hi"))

    def test_repo_map_skips_agentignored_files(self):
        from agent_cli.repomap import RepoMap
        (self.tmp / "Secret.kt").write_text("class Secret\n")
        (self.tmp / "Open.kt").write_text("class Open\n")
        (self.tmp / ".agentignore").write_text("Secret.kt\n")
        files = RepoMap(self.tmp).list_files()
        self.assertIn("Open.kt", files)
        self.assertNotIn("Secret.kt", files)

    def test_strict_failure_requeues_and_pauses(self):
        red = self.red(strict=True)
        text = "customer Hans Meier called\n" + "z" * 600
        redact.note_untrusted(text, "log")
        cfg = Config(workdir=self.tmp, model_map='{"pii": "small"}')
        with mock.patch.object(llm, "chat", side_effect=llm.LLMError("down")) as m:
            red.run_strict(cfg)
            red.run_strict(cfg)          # paused: no second call
        self.assertEqual(m.call_count, 1)
        self.assertEqual(len(redact._pending_strict), 1)   # still queued, not marked done
        red._strict_retry_at = 0
        with mock.patch.object(llm, "chat", return_value=llm.Reply("Hans Meier", "stop", "")):
            red.run_strict(cfg)
        self.assertNotIn("Hans Meier", red.mask_text(text))

    def test_strict_model_and_server(self):
        red = self.red(strict=True)
        cfg = Config(workdir=self.tmp, model="big", base_url="http://remote/v1",
                     model_map='{"default": "local-small"}')
        self.assertEqual(red._pii_target(cfg), ("local-small", "http://remote/v1"))
        os.environ["AGENT_REDACT_PII_URL"] = "http://localhost:9000/v1"
        try:
            self.assertEqual(red._pii_target(cfg)[1], "http://localhost:9000/v1")
        finally:
            del os.environ["AGENT_REDACT_PII_URL"]
        red._pii_target(Config(workdir=self.tmp, model="big", base_url="http://remote/v1"))
        self.assertTrue(any("main model" in m for lvl, m in self.reports if lvl == "warn"))

    def test_nested_strict_call_keeps_queue(self):
        red = self.red(strict=True)
        redact.note_untrusted("q" * 600, "paste")
        redact._tls.in_strict = True
        try:
            red.run_strict(Config(workdir=self.tmp))
        finally:
            redact._tls.in_strict = False
        self.assertEqual(len(redact._pending_strict), 1)

    def test_cache_recomputes_after_new_value(self):
        red = self.red()
        text = "log: hunter2xyz used"
        self.assertEqual(red.mask_text(text), text)        # cached, nothing known yet
        red.mask_text('password = "hunter2xyz"')
        self.assertEqual(red.mask_text(text), "log: [REDACTED_PASSWORD_1] used")

    def test_stream_context_bounded_and_escapes_in_tool_call(self):
        red = self.red()
        red.mask_text(self.KEY)
        ph = redact.PLACEHOLDER_RE.search(red.mask_text(self.KEY)).group(0)
        un = redact.StreamUnmasker(red)
        out = un.feed("x" * 100000)
        out += un.feed('<tool_') + un.feed('call>{"a": "' + ph[:5]) + un.feed(ph[5:] + '"}</tool_call>')
        self.assertLess(len(un.raw), 100)
        body = out[100000:]
        call = json.loads(body[len("<tool_call>"):-len("</tool_call>")])
        self.assertIn("MIIfakeKEYbodyLINEone\nMIIfakeKEYbodyLINEtwo", call["a"])


if __name__ == "__main__":
    unittest.main()
