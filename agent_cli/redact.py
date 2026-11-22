"""Mask secrets and personal data before anything is sent to the LLM.

The one choke point is llm.py: every outgoing message is passed through
`Redactor.mask_messages` right before the request is built, and every reply
(content, reasoning, tool-call arguments) through `unmask_*` right after it
arrives. Nothing else calls the model server, so nothing else needs to care.

Masking is reversible. Only the sensitive value is replaced, with a placeholder
like ``[REDACTED_PASSWORD_1]``::

    password = "hunter2"   ->   password = "[REDACTED_PASSWORD_1]"

The same value always gets the same placeholder for the whole session, so
prompts stay byte-identical between requests (prefix caching keeps working) and
the model can refer to a value it never saw. Replies are unmasked, so a
SEARCH/REPLACE block written around a placeholder applies to the real file. The
value <-> placeholder mapping lives in memory only and is never written to disk.

Two detector families:

- Secrets (always on): private keys, cloud/SaaS/LLM tokens, JWTs, passwords in
  URLs, Authorization headers, values assigned to secret-like names.
- Personal data (opt-in: AGENT_REDACT_PII=1 or "pii": true in the config):
  pattern + validation (Luhn, IBAN mod-97, tax-ID checksum, ...) producing a
  score that context words raise or lower; only matches at or above the
  threshold are masked. Optional strict mode additionally asks a local model.

Config: <project>/.agent/redact.json, overridden by AGENT_REDACT (0 = off),
AGENT_REDACT_PII and AGENT_REDACT_STRICT. Standard library only.
"""
from __future__ import annotations

import argparse
import codecs
import fnmatch
import glob
import hashlib
import ipaddress
import json
import os
import re
import sys
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ---- placeholders -------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\[REDACTED_([A-Z][A-Z0-9_]*?)_(\d+)\]")
# A trailing piece of text that could still grow into a placeholder.
_PARTIAL_RE = re.compile(r"\[(?:R(?:E(?:D(?:A(?:C(?:T(?:E(?:D(?:_[A-Z0-9_]*)?)?)?)?)?)?)?)?)?")
_MAX_PLACEHOLDER = 64

# kind -> (singular, plural) for the "masked before sending" line.
_LABELS = {
    "PASSWORD": ("password", "passwords"),
    "SECRET": ("secret", "secrets"),
    "TOKEN": ("token", "tokens"),
    "API_KEY": ("API key", "API keys"),
    "PRIVATE_KEY": ("private key", "private keys"),
    "AWS_KEY": ("AWS key", "AWS keys"),
    "AWS_SECRET": ("AWS secret", "AWS secrets"),
    "GITHUB_TOKEN": ("GitHub token", "GitHub tokens"),
    "GITLAB_TOKEN": ("GitLab token", "GitLab tokens"),
    "SLACK_TOKEN": ("Slack token", "Slack tokens"),
    "SLACK_WEBHOOK": ("Slack webhook", "Slack webhooks"),
    "GOOGLE_KEY": ("Google API key", "Google API keys"),
    "STRIPE_KEY": ("Stripe key", "Stripe keys"),
    "ATLASSIAN_TOKEN": ("Atlassian token", "Atlassian tokens"),
    "JWT": ("JWT", "JWTs"),
    "EMAIL": ("email", "emails"),
    "CARD": ("card number", "card numbers"),
    "IBAN": ("IBAN", "IBANs"),
    "PHONE": ("phone number", "phone numbers"),
    "TAX_ID": ("tax ID", "tax IDs"),
    "IP": ("IP address", "IP addresses"),
    "NAME": ("name", "names"),
    "ADDRESS": ("address", "addresses"),
    "BIRTH_DATE": ("birth date", "birth dates"),
    "PII": ("personal data item", "personal data items"),
}

# ---- reporting ----------------------------------------------------------------------

Reporter = Callable[[str, str], None]  # (message, level: "info" | "warn")


def _default_reporter(msg: str, level: str) -> None:
    print(msg, file=sys.stderr)


_reporter: Reporter = _default_reporter


def set_reporter(fn: Optional[Reporter]) -> None:
    """Route masking notices (counts, file warnings) to the app's console."""
    global _reporter
    _reporter = fn or _default_reporter


def rich_reporter(console) -> Reporter:
    """A reporter that prints to a rich Console: dim info, yellow warnings."""
    from .term import safe_markup

    def report(msg: str, level: str) -> None:
        style = "yellow" if level == "warn" else "dim"
        console.print(f"[{style}]{safe_markup(msg)}[/{style}]")
    return report


def _report(msg: str, level: str = "info") -> None:
    try:
        _reporter(msg, level)
    except Exception:
        pass  # a notice must never break a request


def describe_counts(counts: Dict[str, int]) -> str:
    """'2 passwords, 1 email' from {'PASSWORD': 2, 'EMAIL': 1}."""
    parts = []
    for kind, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        one, many = _LABELS.get(kind, (kind.lower().replace("_", " "),
                                       kind.lower().replace("_", " ") + "s"))
        parts.append(f"{n} {one if n == 1 else many}")
    return ", ".join(parts)


def report_counts(counts: Dict[str, int]) -> None:
    """Print the one dim transparency line after a request (never the values)."""
    if counts and not getattr(_tls, "quiet", False):
        _report(f"masked before sending: {describe_counts(counts)}")


# ---- findings -----------------------------------------------------------------------

@dataclass
class Finding:
    start: int
    end: int
    kind: str
    rule: str
    score: float = 1.0
    secret: bool = True     # secrets are always masked; PII only above the threshold
    priority: int = 50      # lower wins when two findings cover the same span

    def value(self, text: str) -> str:
        return text[self.start:self.end]


# ---- secret detection ---------------------------------------------------------------

# Names that hold secrets. The name must END with one of these words, so
# `authToken`, `DB_PASSWORD` and `client_secret` match but `tokenType`,
# `passwordHint` and `tokenizer` do not.
_SECRET_WORD = (
    r"(?:password|passwd|passwort|pwd|passphrase|secret|token|"
    r"api[_\-.]?key|apikey|access[_\-.]?key(?:[_\-.]?id)?|private[_\-.]?key|secret[_\-.]?key|"
    r"signing[_\-.]?key|encryption[_\-.]?key|auth[_\-.]?key|master[_\-.]?key|"
    r"client[_\-.]?secret|credentials?|conn(?:ection)?[_\-.]?string)"
)
# A secret word that ends an identifier. The name rules are anchored on these
# hits (scan back for the rest of the name, then match the separator and value
# forward), which keeps them linear even on megabyte-long minified lines.
_SECRET_WORD_RX = re.compile(_SECRET_WORD + r"(?![A-Za-z0-9])", re.I)
_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")

# After the name: optional closing quote of a JSON/YAML key, optional `: Type`,
# separator, quoted value.
_QUOTED_TAIL_RE = re.compile(
    r"[\"']?[ \t]*(?::[ \t]*[A-Za-z_][\w.]*(?:<[^>\n]*>)?\??[ \t]*)?"
    r"(?:=|:=|=>|:|\+=)[ \t]*([\"'`])((?:\\.|(?!\1)[^\\\n])*)\1")
# Unquoted value with the name at line start: .env, .properties, YAML, ini.
# A leading "+"/"-" is a diff marker: added lines in /review diffs count too.
_LINE_HEAD_RE = re.compile(r"[+\-]?[ \t]*(?:export[ \t]+)?(?:-[ \t]+)?[\"']?")
_LINE_TAIL_RE = re.compile(
    r"[\"']?[ \t]*[=:][ \t]*(?![\"'`])([^\s#;][^\n]*?)[ \t]*(?:[ \t]#[^\n]*)?\r?$")
# Inline in logs and query strings: ...failed (password=abc123) / ?token=abc&x=1
_INLINE_TAIL_RE = re.compile(r"=(?![\"'`])([^\s,;&)\]}'\"<>]{3,})")


def _secret_names(line: str):
    """(name_start, name_end, name) for each identifier in `line` ending in a
    secret word: `DB_PASSWORD`, `authToken`, `spring.datasource.password`."""
    for m in _SECRET_WORD_RX.finditer(line):
        start = m.start()
        while start > 0 and m.start() - start < 60 and line[start - 1] in _NAME_CHARS:
            start -= 1
        while start < m.start() and start > 0 and line[start - 1].isalnum():
            start += 1   # cut at 60 chars mid-word: start after the next separator
        yield start, m.end(), line[start:m.end()]
# Groovy: storePassword 'x' (no '=') ; Kotlin DSL form is caught by the quoted rule.
_ANDROID_SIGNING_RE = re.compile(
    r"\b(?:storePassword|keyPassword)\b[ \t]*\(?[ \t]*([\"'])((?:\\.|(?!\1)[^\\\n])*)\1")
# buildConfigField("String", "API_KEY", "\"abc\"")  /  buildConfigField 'String', 'API_KEY', '"abc"'
_BUILD_CONFIG_RE = re.compile(
    r"buildConfigField[ \t]*\(?[ \t]*[\"'][\w.<>]+[\"'][ \t]*,[ \t]*[\"'](\w+)[\"'][ \t]*,[ \t]*"
    r"([\"'])((?:\\.|(?!\2)[^\\\n])*)\2")
_BUILD_CONFIG_SECRET_NAME_RE = re.compile(r"(?i)" + _SECRET_WORD + r"$|_KEY$|^KEY$")

_URL_PASSWORD_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.\-]{1,15}://[^\s:/@'\"<>]+:([^\s@/'\"<>]+)@[^\s/'\"<>]+")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(?:proxy-)?authorization\b[\"']?[ \t]*[:=][ \t]*[\"']?"
    r"(?:bearer|basic|token|bot|digest|apikey|sso-key)[ \t]+([A-Za-z0-9._~+/=\-:]{6,})")
_BEARER_RE = re.compile(r"\bBearer[ \t]+([A-Za-z0-9._~+/\-]{16,}=*)")
# Newlines may be real or JSON-escaped ("\\n"): a key inside a JSON string or a
# tool call written back into the history is still a key.
_NL = r"(?:\r?\n|(?:\\r)?\\n)"
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN ((?:[A-Z0-9]+ )*)PRIVATE KEY(?: BLOCK)?-----[ \t]*" + _NL
    + r"([\s\S]*?)" + _NL + r"?[ \t]*-----END \1PRIVATE KEY(?: BLOCK)?-----")

# (kind, regex, group) — whole-token secrets with a recognisable shape.
_TOKEN_PATTERNS: List[Tuple[str, "re.Pattern", int]] = [
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), 0),
    ("AWS_SECRET", re.compile(
        r"(?i)aws.{0,20}?(?:secret|sk).{0,20}?[\"'\s:=]+([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])"), 1),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b"), 0),
    ("GITHUB_TOKEN", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"), 0),
    ("GITLAB_TOKEN", re.compile(
        r"\bgl(?:pat|dt|rt|ptt|soat|cbt|ft|imt|agent)-[A-Za-z0-9_\-]{20,}"), 0),
    ("SLACK_TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), 0),
    ("SLACK_WEBHOOK", re.compile(r"https://hooks\.slack\.com/services/([A-Za-z0-9/_\-]{20,})"), 1),
    ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])"), 0),
    ("STRIPE_KEY", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b"), 0),
    ("STRIPE_KEY", re.compile(r"\bwhsec_[A-Za-z0-9]{24,}\b"), 0),
    ("ATLASSIAN_TOKEN", re.compile(r"\bATATT[A-Za-z0-9_\-=]{20,}"), 0),
    ("API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), 0),
    ("API_KEY", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("API_KEY", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0),
    ("API_KEY", re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"), 0),
    ("API_KEY", re.compile(r"\bpplx-[A-Za-z0-9]{40,}\b"), 0),
    ("API_KEY", re.compile(r"\bxai-[A-Za-z0-9]{40,}\b"), 0),
    ("API_KEY", re.compile(r"\br8_[A-Za-z0-9]{30,}\b"), 0),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
]

# Values that are obviously not real secrets.
_DUMMY_VALUES = {
    "changeme", "change_me", "change-me", "changeit", "password", "passwort", "secret",
    "token", "null", "none", "nil", "undefined", "true", "false", "todo", "fixme",
    "dummy", "placeholder", "redacted", "example", "sample", "xxx", "bearer", "basic",
    "string", "required", "optional", "env", "your_password", "yourpassword",
    "not-needed", "not_needed", "unused", "empty", "n/a", "tbd",
}
# Kotlin/Java/Swift/TS type names that follow `password:` in declarations.
_TYPE_NAMES = {
    "String", "CharSequence", "CharArray", "ByteArray", "Int", "Long", "Boolean", "Any",
    "Char", "Byte", "Short", "Double", "Float", "Unit", "Nothing", "Object", "Integer",
    "char[]", "byte[]", "string", "number", "boolean", "str", "bytes", "int",
    "SecretKey", "Credentials", "Password", "Token", "Secret", "PasswordAuthentication",
}
_CODE_EXTS = {
    ".kt", ".kts", ".java", ".swift", ".m", ".mm", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".py", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".dart", ".scala",
    ".groovy", ".gradle", ".rb", ".php",
}


def _lines_matching(text: str, rx: "re.Pattern"):
    """(offset, line) for each line of text that rx finds something in."""
    pos = 0
    for line in text.split("\n"):
        if rx.search(line):
            yield pos, line
        pos += len(line) + 1


def _code_path(path: Optional[str]) -> bool:
    return bool(path) and Path(path).suffix.lower() in _CODE_EXTS


def _looks_dummy(value: str) -> bool:
    v = value.strip()
    if len(v) < 3:
        return True
    if PLACEHOLDER_RE.search(v):
        return True
    low = v.lower()
    if low in _DUMMY_VALUES:
        return True
    if "${" in v or "{{" in v or "%(" in v:
        return True
    if re.fullmatch(r"\$[A-Za-z_][\w.]*", v) or re.fullmatch(r"%[A-Za-z_]\w*%", v):
        return True
    if re.fullmatch(r"<[^<>]*>", v) or re.fullmatch(r"__[A-Z0-9_]+__", v):
        return True   # <your-token>, __SESSION_TOKEN__: template slots
    if re.fullmatch(r"[*xX•.#\-_]{3,}", v):
        return True
    if low.startswith(("your_", "your-", "your ", "<your", "my_secret", "insert_", "replace_")):
        return True
    return False


def _looks_like_code_value(value: str, name: str) -> bool:
    """For unquoted values: is this an expression or a type rather than a secret?"""
    v = value.strip().rstrip(",;")
    if not v:
        return True
    if v in _TYPE_NAMES or v.rstrip("?") in _TYPE_NAMES:
        return True
    if v[0] in "{[&*!|>(":     # YAML block/anchor/alias/flow, or code
        return True
    if any(c in v for c in "()"):
        return True
    if re.fullmatch(r"[a-z_][\w]*(?:\.[A-Za-z_]\w*)+", v):   # it.token, viewModel.password
        return True
    if v.lower().replace("_", "").replace("-", "") == name.lower().replace("_", "").replace("-", ""):
        return True
    if v in ("=", "==", "->", "=>"):
        return True
    if re.match(r"[A-Z][\w.]*(?:<[^>]*>)?\??[ \t]*(?:=|,|\)|\{)", v):   # `String = ""`, `String,`
        return True
    if re.fullmatch(r"[a-z]+(?:[A-Z][a-z]+)+", v):   # newPassword: a variable, not a value
        return True
    return False


def _is_key_name_constant(value: str) -> bool:
    """'pref_auth_token', 'key_password': the name of a key, not its value."""
    return bool(re.fullmatch(r"[a-z][a-z0-9]*(?:[_.\-][a-z0-9]+)*", value)) and bool(
        re.search(r"pass|pwd|secret|token|key|credential", value))


def _kind_for_name(name: str) -> str:
    n = name.lower()
    if any(w in n for w in ("pass", "pwd", "passwort")):
        return "PASSWORD"
    if "token" in n:
        return "TOKEN"
    if "key" in n:
        return "API_KEY"
    return "SECRET"


# ---- personal data detection ----------------------------------------------------------

_EMAIL_RE = re.compile(r"(?<![\w.+\-])[A-Za-z0-9._%+\-]+@(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}(?![\w\-])")
_EMAIL_SKIP_DOMAINS = {"example.com", "example.org", "example.net", "example.de", "test.com",
                       "test.de", "localhost", "email.com", "domain.com", "users.noreply.github.com"}
_EMAIL_SKIP_TLDS = {"test", "example", "invalid", "localhost", "local",
                    "png", "jpg", "jpeg", "gif", "svg", "webp", "kt", "java", "js", "ts"}
_CARD_RE = re.compile(r"(?<![\d\-.])(?:\d[ \-]?){12,18}\d(?![\d\-])")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_PHONE_INTL_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ \-./]?\(?\d{1,5}\)?){1,6}(?![\w])")
_PHONE_DE_RE = re.compile(r"(?<![\w+./\-])0[1-9]\d{1,4}(?:[ \-/]?\d{2,8}){1,4}(?![\w])")
_TAX_ID_RE = re.compile(r"(?<![\d])[1-9]\d(?: ?\d{3}){3}(?![\d])")
_IPV4_RE = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
_IPV6_RE = re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")

_IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BR": 29, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20,
    "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26, "IT": 27, "JO": 30,
    "KW": 30, "KZ": 20, "LB": 28, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24,
    "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28,
    "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SE": 24, "SI": 19, "SK": 24,
    "SM": 27, "TN": 24, "TR": 26, "UA": 29, "VG": 24, "XK": 20,
}

_POSITIVE_CONTEXT = {
    "PHONE": r"phone|tel|telefon|mobil|mobile|handy|fax|cell|rufnummer|call|kontakt|contact",
    "IBAN": r"iban|kontonummer|konto|bank|account|bic",
    "EMAIL": r"e-?mail|mail|contact|kontakt",
    "CARD": r"card|karte|kreditkarte|visa|mastercard|amex|cc|pan|payment",
    "TAX_ID": r"steuer-?id(?:nr)?|steuer|idnr|tax-?id|tax|tin|identifikationsnummer",
    "IP": r"ip|addr|host|client|remote|from|source",
}
_NEGATIVE_CONTEXT = re.compile(
    r"(?i)(?<![a-z])(?:version|ver|v|build|timestamp|time|ts|millis|epoch|id|uuid|guid|sha|hash|"
    r"commit|port|count|size|bytes|seq|sequence|offset|line|checksum|crc|order)(?![a-z])")

_PII_FIELDS = {
    "email": "EMAIL", "emailaddress": "EMAIL", "mail": "EMAIL", "eMail": "EMAIL",
    "phone": "PHONE", "phonenumber": "PHONE", "mobile": "PHONE", "mobilenumber": "PHONE",
    "telephone": "PHONE", "tel": "PHONE", "telefon": "PHONE", "handy": "PHONE", "fax": "PHONE",
    "iban": "IBAN",
    "birthdate": "BIRTH_DATE", "dateofbirth": "BIRTH_DATE", "dob": "BIRTH_DATE",
    "birthday": "BIRTH_DATE", "geburtsdatum": "BIRTH_DATE",
    "address": "ADDRESS", "streetaddress": "ADDRESS", "street": "ADDRESS",
    "addressline1": "ADDRESS", "addressline2": "ADDRESS", "strasse": "ADDRESS",
    "firstname": "NAME", "lastname": "NAME", "surname": "NAME", "givenname": "NAME",
    "familyname": "NAME", "fullname": "NAME", "vorname": "NAME", "nachname": "NAME",
}
_FIELD_KEY = r"[A-Za-z][A-Za-z0-9_\-]{0,40}"
# "first_name": "Max"  /  first_name: 'Max'  /  email = "x@y.de"
_FIELD_QUOTED_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])[\"']?(" + _FIELD_KEY + r")[\"']?[ \t]*(?:=|:|=>)[ \t]*([\"'])((?:\\.|(?!\2)[^\\\n])*)\2")
# YAML / properties at line start:  first_name: Max
_FIELD_LINE_RE = re.compile(
    r"(?m)^[+\-]?[ \t]*(?:-[ \t]+)?(" + _FIELD_KEY + r")[ \t]*[:=][ \t]*(?![\"'])([^\s#][^\n#]*?)[ \t]*\r?$")
# log style key=value anywhere: first_name=Max phone=+49...
_FIELD_KV_RE = re.compile(r"(?<![\w.])(" + _FIELD_KEY + r")=([^\s,;&)}\"'(]+)(?![(.\w])")
# JSON number value: "phone": 491701234567
_FIELD_JSON_NUM_RE = re.compile(r"\"(" + _FIELD_KEY + r")\"[ \t]*:[ \t]*(\+?\d[\d \-/]{5,}\d)")
_SQL_INSERT_RE = re.compile(
    r"(?is)INSERT\s+INTO\s+[`\"\[]?[\w.]+[`\"\]]?\s*\(([^)]*)\)\s*VALUES\s*")


def _norm_field(name: str) -> str:
    return re.sub(r"[_\-\s.]", "", name).lower()


def _field_kind(name: str) -> Optional[str]:
    n = _norm_field(name)
    if n in _PII_FIELDS:
        return _PII_FIELDS[n]
    for suffix in ("email", "phone", "iban", "firstname", "lastname", "birthdate", "dateofbirth"):
        if n.endswith(suffix) and len(n) <= len(suffix) + 10:
            return _PII_FIELDS[suffix]
    return None


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "").upper()
    if len(s) < 15 or _IBAN_LENGTHS.get(s[:2]) != len(s):
        return False
    rearranged = s[4:] + s[:4]
    try:
        number = int("".join(str(int(c, 36)) for c in rearranged))
    except ValueError:
        return False
    return number % 97 == 1


def de_tax_id_ok(raw: str) -> bool:
    """German Steuer-Identifikationsnummer: 11 digits, ISO 7064 MOD 11,10 check digit,
    and in the first ten digits exactly one digit repeats (2 or 3 times)."""
    s = raw.replace(" ", "")
    if len(s) != 11 or not s.isdigit() or s[0] == "0":
        return False
    counts = Counter(s[:10])
    repeated = [d for d, c in counts.items() if c > 1]
    if len(repeated) != 1 or counts[repeated[0]] not in (2, 3):
        return False
    product = 10
    for ch in s[:10]:
        total = (int(ch) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    check = 11 - product
    if check == 10:
        check = 0
    return check == int(s[10])


def _card_prefix_ok(d: str) -> bool:
    n = len(d)
    if d.startswith("4") and n in (13, 16, 19):
        return True
    two, four = int(d[:2]), int(d[:4])
    if 51 <= two <= 55 and n == 16:
        return True
    if 2221 <= four <= 2720 and n == 16:
        return True
    if two in (34, 37) and n == 15:
        return True
    if (d.startswith("6011") or d.startswith("65")) and n in (16, 19):
        return True
    if 3528 <= four <= 3589 and n in (16, 19):
        return True
    return False


# ---- file policy -------------------------------------------------------------------

BLOCKED_SUFFIXES = {".jks", ".keystore", ".p12", ".pfx", ".pem", ".key", ".ppk"}
SSH_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk",
                 "identity"}
SENSITIVE_NAMES = {"local.properties", "gradle.properties", "keystore.properties",
                   "secrets.properties", "signing.properties", "google-services.json",
                   "googleservice-info.plist", ".netrc", ".npmrc", ".pypirc"}
TEST_DATA_GLOBS = ["**/src/test/**", "**/src/androidTest/**", "**/fixtures/**",
                   "**/__fixtures__/**", "**/testFixtures/**"]


@dataclass
class FilePolicy:
    action: str   # "block", "warn" or "ok"
    reason: str = ""


def _gitignore_regex(pat: str) -> Tuple[Optional["re.Pattern"], bool]:
    """Compile one .gitignore line. Returns (regex, negated) or (None, False)."""
    pat = pat.rstrip("\n").rstrip()
    if not pat or pat.startswith("#"):
        return None, False
    neg = pat.startswith("!")
    if neg:
        pat = pat[1:]
    if pat.startswith("\\"):
        pat = pat[1:]
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    anchored = "/" in pat
    pat = pat.lstrip("/")
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("/**", i) and i + 3 == len(pat):
            out.append("(?:/.*)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pat[i + 1:j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    core = "".join(out)
    head = "^" if anchored else "^(?:.*/)?"
    tail = "/.*$" if dir_only else "(?:/.*)?$"
    return re.compile(head + core + tail), neg


class IgnoreFile:
    """A .agentignore (gitignore syntax). Later lines win; `!` re-includes."""

    def __init__(self, lines: Iterable[str]):
        self.rules = [r for r in (_gitignore_regex(line) for line in lines) if r[0] is not None]

    def matches(self, rel: str) -> bool:
        hit = False
        for rx, neg in self.rules:
            if rx.match(rel):
                hit = not neg
        return hit


# ---- the redactor ------------------------------------------------------------------

def _env_flag(name: str) -> Optional[bool]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    return v.strip().lower() not in ("0", "false", "no", "off")


_tls = threading.local()
_pending_strict: List[str] = []
_pending_lock = threading.Lock()


def note_untrusted(text: str, source: str = "") -> None:
    """Mark text (a log, a Jira ticket, a big paste) for the strict PII pass.

    Cheap no-op unless strict mode is on; the text is checked lazily, right
    before the next request, by whichever session sends it.
    """
    if not text or len(text) <= 500:
        return
    with _pending_lock:
        _pending_strict.append(text)
        del _pending_strict[:-50]  # never grow without bound


class Redactor:
    """Session state: settings, detectors and the value <-> placeholder mapping."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root).resolve() if root else None
        self._lock = threading.RLock()
        self._by_value: Dict[str, str] = {}
        self._by_ph: Dict[str, str] = {}
        self._counters: Counter = Counter()
        self._known_rx: Optional["re.Pattern"] = None
        self._known_forms: Dict[str, str] = {}   # value or its JSON-escaped form -> placeholder
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._strict_retry_at = 0.0
        self._strict_same_model_warned = False
        self._warned: set = set()
        self._strict_done: set = set()
        self._strict_failed = False
        self.load_settings()

    # -- settings --

    def load_settings(self) -> None:
        cfg: dict = {}
        self.config_error = ""
        if self.root is not None:
            p = self.root / ".agent" / "redact.json"
            if p.is_file():
                try:
                    cfg = json.loads(p.read_text(encoding="utf-8")) or {}
                    if not isinstance(cfg, dict):
                        raise ValueError("top level must be an object")
                except (OSError, ValueError) as e:
                    cfg = {}
                    self.config_error = f"{p}: {e}"
        env_on = _env_flag("AGENT_REDACT")
        self.enabled = env_on if env_on is not None else bool(cfg.get("enabled", True))
        env_strict = _env_flag("AGENT_REDACT_STRICT")
        self.strict = env_strict if env_strict is not None else bool(cfg.get("strict", False))
        env_pii = _env_flag("AGENT_REDACT_PII")
        self.pii = env_pii if env_pii is not None else bool(cfg.get("pii", False))
        self.pii = self.pii or self.strict   # strict implies PII
        try:
            self.threshold = float(cfg.get("threshold", 0.5))
        except (TypeError, ValueError):
            self.threshold = 0.5
        self.allow = {str(v) for v in cfg.get("allow", []) or [] if str(v)}
        self.allow_paths = [str(p) for p in cfg.get("allow_paths", []) or []]
        self.pii_base_url = str(cfg.get("pii_base_url", "") or "")
        self.patterns: List[Tuple[str, "re.Pattern", int]] = []
        for item in cfg.get("patterns", []) or []:
            try:
                kind = re.sub(r"[^A-Z0-9]+", "_", str(item.get("name", "custom")).upper()).strip("_")
                self.patterns.append((kind or "CUSTOM", re.compile(item["regex"]),
                                      int(item.get("group", 0))))
            except (KeyError, TypeError, ValueError, re.error) as e:
                self.config_error = f"redact.json pattern {item!r}: {e}"
        self.names_rx: Optional["re.Pattern"] = None
        if self.root is not None:
            names_file = self.root / ".agent" / "redact-names.txt"
            if names_file.is_file():
                try:
                    names = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines()]
                except OSError:
                    names = []
                names = sorted({n for n in names if n and not n.startswith("#") and len(n) >= 2},
                               key=len, reverse=True)
                if names:
                    self.names_rx = re.compile(
                        r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)")
        self.ignore = IgnoreFile([])
        if self.root is not None:
            ig = self.root / ".agentignore"
            if ig.is_file():
                try:
                    self.ignore = IgnoreFile(ig.read_text(encoding="utf-8").splitlines())
                except OSError:
                    pass

    # -- file policy --

    def _rel(self, path) -> str:
        p = Path(path)
        if self.root is not None:
            try:
                full = p if p.is_absolute() else (self.root / p)
                return full.resolve().relative_to(self.root).as_posix()
            except (ValueError, OSError):
                pass
        return p.as_posix()

    def file_policy(self, path, sniff: bool = True) -> FilePolicy:
        """Policy for one file. `sniff` also opens it to look for a private-key
        header; callers checking many paths at once (the repo map) skip that."""
        rel = self._rel(path)
        name = Path(rel).name
        low = name.lower()
        if self.ignore.matches(rel):
            return FilePolicy("block", "listed in .agentignore")
        if Path(low).suffix in BLOCKED_SUFFIXES:
            return FilePolicy("block", f"{Path(low).suffix} files hold keys or certificates")
        if low in SSH_KEY_NAMES or (low.startswith("id_") and not low.endswith(".pub")
                                    and "/.ssh/" in "/" + rel):
            return FilePolicy("block", "SSH private key")
        full = (self.root / rel) if self.root is not None and not Path(rel).is_absolute() else Path(rel)
        try:
            if sniff and full.is_file():
                with open(full, "rb") as fh:
                    head = fh.read(200)
                if re.search(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY", head):
                    return FilePolicy("block", "contains a private key")
        except OSError:
            pass
        if low.startswith(".env") or low in SENSITIVE_NAMES:
            return FilePolicy("warn", "may contain secrets; sent masked")
        return FilePolicy("ok")

    def check_file(self, path) -> Optional[str]:
        """None if the file may be sent; else a refusal message. Warns once for
        sensitive files (sent masked)."""
        pol = self.file_policy(path)
        rel = self._rel(path)
        if pol.action == "block":
            return f"{rel}: refused, never sent to the model ({pol.reason})."
        if pol.action == "warn" and rel not in self._warned:
            self._warned.add(rel)
            if self.enabled:
                _report(f"{rel} may contain secrets: it is sent with secret values masked.", "warn")
            else:
                _report(f"{rel} may contain secrets and masking is OFF (AGENT_REDACT=0).", "warn")
        return None

    def is_test_data(self, path: Optional[str]) -> bool:
        if not path:
            return False
        rel = self._rel(path)
        for g in TEST_DATA_GLOBS + self.allow_paths:
            if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch("/" + rel, g) or fnmatch.fnmatch(rel, g.lstrip("*/")):
                return True
        return False

    # -- detection --

    def find(self, text: str, path: Optional[str] = None, pii: Optional[bool] = None) -> List[Finding]:
        """All findings in `text` (secrets + PII above the threshold), non-overlapping."""
        use_pii = self.pii if pii is None else pii
        found = self._find_secrets(text, path)
        if use_pii and not self.is_test_data(path):
            found += [f for f in self._find_pii(text, path) if f.score >= self.threshold]
        return self._resolve(text, found)

    def _resolve(self, text: str, found: List[Finding]) -> List[Finding]:
        found.sort(key=lambda f: (f.start, f.priority, -(f.end - f.start)))
        out: List[Finding] = []
        for f in found:
            if f.end <= f.start:
                continue
            v = f.value(text)
            if v in self.allow or PLACEHOLDER_RE.search(v):
                continue   # allowlisted, or already masked
            if f.rule not in ("private_key", "token", "custom") and _looks_dummy(v):
                continue
            if out and f.start < out[-1].end:
                prev = out[-1]
                # keep the higher-priority one; on a tie keep the longer
                if (f.priority, -(f.end - f.start)) < (prev.priority, -(prev.end - prev.start)):
                    out[-1] = f
                continue
            out.append(f)
        return out

    def _find_secrets(self, text: str, path: Optional[str]) -> List[Finding]:
        out: List[Finding] = []
        add = out.append
        for m in _PRIVATE_KEY_RE.finditer(text):
            if m.group(2).strip():
                add(Finding(m.start(2), m.end(2), "PRIVATE_KEY", "private_key", priority=0))
        for kind, rx, grp in _TOKEN_PATTERNS:
            for m in rx.finditer(text):
                add(Finding(m.start(grp), m.end(grp), kind, "token", priority=10))
        for kind, rx, grp in self.patterns:
            for m in rx.finditer(text):
                try:
                    s, e = m.span(grp)
                except IndexError:
                    s, e = m.span()
                add(Finding(s, e, kind, "custom", priority=5))
        for m in _URL_PASSWORD_RE.finditer(text):
            add(Finding(m.start(1), m.end(1), "PASSWORD", "url_password", priority=20))
        for m in _AUTH_HEADER_RE.finditer(text):
            add(Finding(m.start(1), m.end(1), "TOKEN", "auth_header", priority=20))
        for m in _BEARER_RE.finditer(text):
            add(Finding(m.start(1), m.end(1), "TOKEN", "bearer", priority=25))
        for m in _BUILD_CONFIG_RE.finditer(text):
            if not _BUILD_CONFIG_SECRET_NAME_RE.search(m.group(1)):
                continue
            s, e = m.start(3), m.end(3)
            inner = text[s:e]
            # strip the escaped quotes Gradle needs around a String value
            for q in ('\\"', '"'):
                if inner.startswith(q) and inner.endswith(q) and len(inner) >= 2 * len(q):
                    s, e = s + len(q), e - len(q)
                    break
            v = text[s:e]
            if any(c in v for c in "$+(") or not v:
                continue
            add(Finding(s, e, _kind_for_name(m.group(1)), "build_config", priority=30))
        for m in _ANDROID_SIGNING_RE.finditer(text):
            add(Finding(m.start(2), m.end(2), "PASSWORD", "android_signing", priority=30))
        # Name-based rules are slow on long punctuation-heavy lines (minified
        # code), so they only run on the lines that mention a secret word.
        code = _code_path(path)
        for off, line in _lines_matching(text, _SECRET_WORD_RX):
            for ns, ne, name in _secret_names(line):
                m = _QUOTED_TAIL_RE.match(line, ne)
                if m:
                    # const val PREF_TOKEN = "pref_token": a key name, not a secret
                    if not _is_key_name_constant(m.group(2)):
                        add(Finding(off + m.start(2), off + m.end(2), _kind_for_name(name),
                                    "assignment", priority=40))
                    continue
                if code:
                    continue
                m = _LINE_TAIL_RE.match(line, ne)
                if m and _LINE_HEAD_RE.fullmatch(line, 0, ns):
                    if not _looks_like_code_value(m.group(1), name):
                        add(Finding(off + m.start(1), off + m.end(1), _kind_for_name(name),
                                    "assignment_line", priority=45))
                    continue
                m = _INLINE_TAIL_RE.match(line, ne)
                if m and not _looks_like_code_value(m.group(1), name):
                    add(Finding(off + m.start(1), off + m.end(1), _kind_for_name(name),
                                "assignment_inline", priority=46))
        return out

    def _context_score(self, text: str, start: int, end: int, kind: str) -> float:
        """+0.3 for a nearby word that fits the kind, -0.4 for one that suggests a
        technical number (version, timestamp, id...). The nearer word wins."""
        line_start = text.rfind("\n", 0, start) + 1
        before = text[max(line_start, start - 40):start].lower()
        after = text[end:end + 20].split("\n", 1)[0].lower()
        pos = _POSITIVE_CONTEXT.get(kind)
        pos_rx = re.compile(r"(?<![a-z])(?:" + pos + r")") if pos else None
        last_pos = max((m.end() for m in pos_rx.finditer(before)), default=-1) if pos_rx else -1
        last_neg = max((m.end() for m in _NEGATIVE_CONTEXT.finditer(before)), default=-1)
        if last_pos >= 0 or last_neg >= 0:
            # ties go to the positive word: "Steuer-ID" is not a technical "id"
            return 0.3 if last_pos >= last_neg else -0.4
        if pos_rx and pos_rx.search(after):
            return 0.3
        return 0.0

    def _find_pii(self, text: str, path: Optional[str]) -> List[Finding]:
        out: List[Finding] = []

        def add(s: int, e: int, kind: str, rule: str, base: float, context: bool = True) -> None:
            score = base + (self._context_score(text, s, e, kind) if context else 0.0)
            out.append(Finding(s, e, kind, rule, score=score, secret=False, priority=60))

        for m in _EMAIL_RE.finditer(text):
            local, domain = m.group(0).rsplit("@", 1)
            domain = domain.lower()
            if local in ("this", "return", "super", "break", "continue"):
                continue   # Kotlin labels: this@MainActivity.finish()
            tld = domain.rsplit(".", 1)[-1]
            if domain in _EMAIL_SKIP_DOMAINS or tld in _EMAIL_SKIP_TLDS \
                    or domain.endswith((".example.com", ".example.org", ".test", ".example", ".local")):
                continue
            add(m.start(), m.end(), "EMAIL", "email", 0.8)
        for m in _IBAN_RE.finditer(text):
            if iban_ok(m.group(0)):
                add(m.start(), m.end(), "IBAN", "iban", 0.95)
        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and luhn_ok(digits) and len(set(digits)) > 2:
                add(m.start(), m.end(), "CARD", "card", 0.7 if _card_prefix_ok(digits) else 0.35)
        for m in _PHONE_INTL_RE.finditer(text):
            n = len(re.sub(r"\D", "", m.group(0)))
            if 8 <= n <= 15:
                add(m.start(), m.end(), "PHONE", "phone_intl", 0.6)
        for m in _PHONE_DE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 7 <= len(digits) <= 13:
                base = 0.5 if re.match(r"01[5-7]\d", digits) else 0.35
                add(m.start(), m.end(), "PHONE", "phone_de", base)
        for m in _TAX_ID_RE.finditer(text):
            if de_tax_id_ok(m.group(0)):
                add(m.start(), m.end(), "TAX_ID", "de_tax_id", 0.5)
        for m in _IPV4_RE.finditer(text):
            try:
                ip = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue
            base = 0.2 if (ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_multicast
                           or ip.is_unspecified or ip.is_link_local) else 0.55
            add(m.start(), m.end(), "IP", "ipv4", base)
        for m in _IPV6_RE.finditer(text):
            raw = m.group(0)
            groups = [g for g in raw.split(":") if g]
            # `::add` (a Kotlin function reference) is valid IPv6 syntax: require
            # real groups and a digit before calling it an address.
            if len(groups) < 2 or not re.search(r"\d", raw):
                continue
            try:
                ip = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_unspecified or ip.is_link_local or ip.is_multicast:
                continue
            add(m.start(), m.end(), "IP", "ipv6", 0.6)
        if self.names_rx is not None:
            for m in self.names_rx.finditer(text):
                add(m.start(), m.end(), "NAME", "name_list", 0.8, context=False)
        self._find_structured(text, path, out)
        return out

    def _structured_value_ok(self, v: str, key: str = "") -> bool:
        s = v.strip()
        if len(s) < 2 or _looks_dummy(s) or s in _TYPE_NAMES:
            return False
        if key and _norm_field(s) == _norm_field(key):
            return False   # "email": "EMAIL" is a lookup table, not data
        if s.lower() in ("null", "none", "true", "false", "n/a", "-", "''", '""'):
            return False
        return True

    def _find_structured(self, text: str, path: Optional[str], out: List[Finding]) -> None:
        def add(s: int, e: int, kind: str, rule: str, key: str = "") -> None:
            while s < e and text[s] in " \t":
                s += 1
            while e > s and text[e - 1] in " \t\r":
                e -= 1
            if self._structured_value_ok(text[s:e], key):
                out.append(Finding(s, e, kind, rule, score=0.9, secret=False, priority=55))

        for m in _FIELD_QUOTED_RE.finditer(text):
            kind = _field_kind(m.group(1))
            if kind:
                add(m.start(3), m.end(3), kind, "field", m.group(1))
        for m in _FIELD_JSON_NUM_RE.finditer(text):
            kind = _field_kind(m.group(1))
            if kind:
                add(m.start(2), m.end(2), kind, "field")
        if not _code_path(path):
            for m in _FIELD_LINE_RE.finditer(text):
                kind = _field_kind(m.group(1))
                v = m.group(2)
                if kind and not _looks_like_code_value(v, m.group(1)):
                    add(m.start(2), m.end(2), kind, "field")
        for m in _FIELD_KV_RE.finditer(text):
            kind = _field_kind(m.group(1))
            v = m.group(2)
            if kind and (v[0].isupper() or v[0].isdigit() or v[0] == "+" or "@" in v):
                add(m.start(2), m.end(2), kind, "field")
        self._find_csv(text, add)
        self._find_sql(text, add)

    @staticmethod
    def _split_cells(line: str, delim: str) -> List[Tuple[int, int]]:
        """Cell spans of one CSV line, honouring double quotes."""
        spans, i, start, quoted = [], 0, 0, False
        while i < len(line):
            c = line[i]
            if c == '"':
                quoted = not quoted
            elif c == delim and not quoted:
                spans.append((start, i))
                start = i + 1
            i += 1
        spans.append((start, len(line)))
        return spans

    def _find_csv(self, text: str, add) -> None:
        lines = text.split("\n")
        offsets, pos = [], 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln) + 1
        i = 0
        while i < len(lines):
            header = lines[i].rstrip("\r")
            delim = next((d for d in (",", ";", "\t", "|") if header.count(d) >= 1), None)
            if delim is None:
                i += 1
                continue
            cols = [header[a:b].strip().strip('"') for a, b in self._split_cells(header, delim)]
            if len(cols) < 2 or not all(re.fullmatch(r"[A-Za-z][\w \-.]{0,40}", c or "x") for c in cols):
                i += 1
                continue
            pii_cols = {j: _field_kind(c) for j, c in enumerate(cols) if _field_kind(c)}
            if not pii_cols:
                i += 1
                continue
            j = i + 1
            while j < len(lines):
                row = lines[j].rstrip("\r")
                if not row.strip():
                    break
                cells = self._split_cells(row, delim)
                if len(cells) != len(cols):
                    break
                for col, kind in pii_cols.items():
                    a, b = cells[col]
                    cell = row[a:b]
                    if cell.strip().startswith('"') and cell.strip().endswith('"') and len(cell.strip()) >= 2:
                        lead = len(cell) - len(cell.lstrip())
                        a, b = a + lead + 1, a + lead + len(cell.strip()) - 1
                    add(offsets[j] + a, offsets[j] + b, kind, "csv")
                j += 1
            i = j

    def _find_sql(self, text: str, add) -> None:
        for m in _SQL_INSERT_RE.finditer(text):
            cols = [c.strip().strip('`"[]') for c in m.group(1).split(",")]
            kinds = {j: _field_kind(c) for j, c in enumerate(cols) if _field_kind(c)}
            if not kinds:
                continue
            i = m.end()
            # walk the tuples: (v1, 'v2', ...), (...) ;
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            while i < len(text) and text[i] == "(":
                i += 1
                col, cell_start, quoted = 0, i, False
                while i < len(text):
                    c = text[i]
                    if quoted:
                        if c == "'" and text[i + 1:i + 2] == "'":
                            i += 2
                            continue
                        if c == "'":
                            quoted = False
                            if col in kinds:
                                add(cell_start, i, kinds[col], "sql")
                    elif c == "'":
                        quoted = True
                        cell_start = i + 1
                    elif c in ",)":
                        if col in kinds and text[cell_start:i].strip() and "'" not in text[cell_start:i]:
                            add(cell_start, i, kinds[col], "sql")
                        col += 1
                        cell_start = i + 1
                        if c == ")":
                            i += 1
                            break
                    i += 1
                while i < len(text) and text[i] in " \t\r\n,":
                    i += 1

    # -- mapping --

    def placeholder_for(self, value: str, kind: str) -> str:
        with self._lock:
            ph = self._by_value.get(value)
            if ph is None:
                self._counters[kind] += 1
                ph = f"[REDACTED_{kind}_{self._counters[kind]}]"
                self._by_value[value] = ph
                self._by_ph[ph] = value
                self._known_rx = None
            return ph

    def learn(self, value: str, kind: str = "PII") -> None:
        """Remember a value found by the strict pass so it is masked everywhere."""
        value = value.strip()
        if len(value) >= 3 and value not in self.allow and not PLACEHOLDER_RE.search(value):
            self.placeholder_for(value, kind)

    def _known_values_rx(self) -> Optional["re.Pattern"]:
        """One regex for every known value, also in its JSON-escaped forms: the
        agent writes tool calls back into the history as JSON, so a key with
        newlines or a name like Müller sits there as "\\n" / "\\u00fc"."""
        with self._lock:
            if self._known_rx is None and self._by_value:
                forms: Dict[str, str] = {}
                for v, ph in self._by_value.items():
                    if len(v) < 3:
                        continue
                    forms.setdefault(v, ph)
                    for ascii_only in (False, True):
                        esc = json.dumps(v, ensure_ascii=ascii_only)[1:-1]
                        if esc != v:
                            forms.setdefault(esc, ph)
                self._known_forms = forms
                vals = sorted(forms, key=len, reverse=True)
                if vals:
                    parts = []
                    for v in vals:
                        # `_` counts as a word char so a value never matches
                        # inside a placeholder like [REDACTED_PASSWORD_1].
                        p = re.escape(v)
                        if v[0].isalnum() or v[0] == "_":
                            # ...but right after an escape like "\n" is a start
                            p = r"(?:(?<![A-Za-z0-9_])|(?<=\\[nrt]))" + p
                        if v[-1].isalnum() or v[-1] == "_":
                            p = p + r"(?![A-Za-z0-9_])"
                        parts.append(p)
                    self._known_rx = re.compile("|".join(parts))
            return self._known_rx

    # -- masking --

    def mask_text(self, text: str, path: Optional[str] = None,
                  seen: Optional[Dict[str, str]] = None) -> str:
        """Mask one string. `seen` collects placeholder -> kind for the report.

        Embedded files whose path is blocked are withheld first, even with
        masking off. Results are cached per (text, path): the history is sent
        again with every request, and masking is deterministic until a new
        value is learned (the cache entry is then recomputed)."""
        if not text:
            return text
        if path is None and "\n" in text:
            text = self.withhold_blocked(text)
        if not self.enabled:
            return text
        key = (text, path)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[2] == len(self._by_value):
                self._cache.move_to_end(key)
                if seen is not None:
                    seen.update(hit[1])
                return hit[0]
        local: Dict[str, str] = {}
        if path is None and "\n" in text:
            masked = self._mask_segments(text, local)
        else:
            masked = self._mask_plain(text, path, local)
        with self._lock:
            self._cache[key] = (masked, local, len(self._by_value))
            while len(self._cache) > 512:
                self._cache.popitem(last=False)
        if seen is not None:
            seen.update(local)
        return masked

    def withhold_blocked(self, text: str) -> str:
        """Replace the body of any embedded file (coder file block, @mention,
        diff section) whose path is blocked. A safety net behind the checks at
        each read site."""
        if "```" not in text and "~~~" not in text and "--- " not in text \
                and "diff --git " not in text:
            return text
        spans = []
        for s, e, p in self._file_spans(text):
            pol = self.file_policy(p, sniff=False)
            if pol.action == "block":
                spans.append((s, e, p, pol.reason))
        if not spans:
            return text
        out, last = [], 0
        for s, e, p, reason in sorted(spans):
            if s < last:
                continue
            out.append(text[last:s])
            out.append(f"(content of {p} withheld: never sent to the model, {reason})")
            last = e
        out.append(text[last:])
        return "".join(out)

    def _file_spans(self, text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for m in self._FILE_BLOCK_RE.finditer(text):
            spans.append((m.start(3), m.end(3), m.group(1)))
        for m in self._MENTION_BLOCK_RE.finditer(text):
            spans.append((m.start(2), m.end(2), m.group(1)))
        for m in self._DIFF_FILE_RE.finditer(text):
            paths = diff_paths(m.group(0))
            if paths:
                spans.append((m.end(1), m.end(), paths[-1]))
        return spans

    def _mask_plain(self, text: str, path: Optional[str], seen: Optional[Dict[str, str]]) -> str:
        out, last = [], 0
        for f in self.find(text, path):
            out.append(text[last:f.start])
            ph = self.placeholder_for(f.value(text), f.kind)
            if seen is not None:
                seen[ph] = f.kind
            out.append(ph)
            last = f.end
        out.append(text[last:])
        masked = "".join(out)
        rx = self._known_values_rx()
        if rx is not None:
            def sub(m: "re.Match") -> str:
                ph = self._known_forms[m.group(0)]
                if seen is not None:
                    seen[ph] = PLACEHOLDER_RE.match(ph).group(1)
                return ph
            masked = rx.sub(sub, masked)
        return masked

    # A file embedded in a prompt: coder's "path\n```lang\n...\n```", an @mention
    # "--- path ---\n...\n--- end path ---", or one file's section of a git diff.
    _FILE_BLOCK_RE = re.compile(
        r"(?m)^([\w./\\\-]+\.[\w]+|[\w./\\\-]*/[\w.\-]+)\n(`{3,}|~{3,})[^\n]*\n([\s\S]*?)\n?^\2[ \t]*$")
    _MENTION_BLOCK_RE = re.compile(r"(?m)^--- (\S+) ---\n([\s\S]*?)\n--- end \1 ---$")
    _DIFF_FILE_RE = re.compile(r"(?m)^(diff --git [^\n]*\n)[\s\S]*?(?=^diff --git |\Z)")

    def _mask_segments(self, text: str, seen: Optional[Dict[str, str]]) -> str:
        """Mask text that may embed whole files, giving each file its path so
        path rules apply (test-data allowlist, code vs config files)."""
        spans = self._file_spans(text)
        if not spans:
            return self._mask_plain(text, None, seen)
        spans.sort()
        out, last = [], 0
        for s, e, p in spans:
            if s < last:
                continue
            out.append(self._mask_plain(text[last:s], None, seen))
            out.append(self._mask_plain(text[s:e], p, seen))
            last = e
        out.append(self._mask_plain(text[last:], None, seen))
        return "".join(out)

    def mask_messages(self, messages: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
        """Masked copies of chat messages (all roles, string content and text
        parts, tool-call arguments) plus {kind: distinct values masked}.
        With masking off, blocked embedded files are still withheld."""
        seen: Dict[str, str] = {}
        out = []
        for msg in messages:
            m = dict(msg)
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = self.mask_text(content, seen=seen)
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part = dict(part)
                        part["text"] = self.mask_text(part["text"], seen=seen)
                    parts.append(part)
                m["content"] = parts
            if isinstance(m.get("tool_calls"), list):
                calls = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = self.mask_text(fn["arguments"], seen=seen)
                    calls.append(tc)
                m["tool_calls"] = calls
            out.append(m)
        return out, dict(Counter(seen.values()))

    # -- unmasking --

    def unmask_text(self, text: str, prefix: str = "", json_mode: bool = False) -> str:
        """Put real values back. Inside a JSON tool call (`json_mode`, or a
        `<tool_call>{...` the text is in) values are JSON-escaped so the JSON
        stays valid even for multi-line keys or values with quotes."""
        if not text or "[REDACTED_" not in text:
            return text
        # Only the part of the prefix since its last open <tool_call> matters.
        i = prefix.rfind("<tool_call>")
        open_prefix = prefix[i:] if i != -1 and prefix.rfind("</tool_call>") < i else ""
        has_tags = "<tool_call>" in text or "</tool_call>" in text
        base = bool(open_prefix) and _in_json_tool_call(open_prefix)

        def in_json(pos: int) -> bool:
            if json_mode:
                return True
            if not has_tags:
                return base
            return _in_json_tool_call(open_prefix + text[:pos])

        def sub(m: "re.Match") -> str:
            value = self._by_ph.get(m.group(0))
            if value is None:
                return m.group(0)
            if in_json(m.start()):
                return json.dumps(value, ensure_ascii=False)[1:-1]
            return value
        return PLACEHOLDER_RE.sub(sub, text)

    def unmask_value(self, value):
        """Unmask strings inside dicts/lists (already-parsed tool arguments)."""
        if isinstance(value, str):
            return self.unmask_text(value)
        if isinstance(value, list):
            return [self.unmask_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.unmask_value(v) for k, v in value.items()}
        return value

    def unmask_arguments(self, args):
        """Tool-call arguments: a JSON string (escape-aware) or a parsed dict."""
        if isinstance(args, str):
            return self.unmask_text(args, json_mode=_looks_json(args))
        return self.unmask_value(args)

    # -- strict mode --

    def _pii_target(self, cfg) -> Tuple[str, str]:
        """(model, base_url) for the strict check: the "pii" entry of the role
        map, else its "default", else the main model; AGENT_REDACT_PII_URL or
        "pii_base_url" points it at a separate (local) server."""
        from .micro.modelpick import DEFAULT_KEY, parse_map
        mapping = parse_map(getattr(cfg, "model_map", ""))
        model = mapping.get("PII") or mapping.get(DEFAULT_KEY) or cfg.model
        url = os.environ.get("AGENT_REDACT_PII_URL", "").strip() or self.pii_base_url or cfg.base_url
        if model == cfg.model and url.rstrip("/") == cfg.base_url.rstrip("/") \
                and not self._strict_same_model_warned:
            self._strict_same_model_warned = True
            _report("strict PII check runs on the main model and server: set a \"pii\" entry in "
                    "AGENT_MODEL_MAP (and AGENT_REDACT_PII_URL for a local server) to keep it local.",
                    "warn")
        return model, url

    def run_strict(self, cfg) -> None:
        """Ask the local "PII" role model about pending logs/tickets/pastes and
        learn what it finds. Runs in addition to the rules, never instead.

        A text counts as checked only once every chunk got an answer. If the
        model fails, the unchecked texts go back in the queue and the check is
        paused for a minute, so a dead model doesn't slow every request."""
        if getattr(_tls, "in_strict", False):
            return   # the PII model's own request: leave the queue alone
        with _pending_lock:
            texts = list(_pending_strict)
            _pending_strict.clear()
        if not (self.enabled and self.strict) or not texts:
            return
        if time.monotonic() < self._strict_retry_at:
            _requeue(texts)
            return
        from . import llm
        import dataclasses
        model, url = self._pii_target(cfg)
        sub_cfg = dataclasses.replace(cfg, model=model, base_url=url, stream=False,
                                      temperature=0.0, retries=1)
        for idx, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if digest in self._strict_done:
                continue
            masked = self.mask_text(text)   # the model only sees rule-masked text
            found: List[str] = []
            for i in range(0, len(masked), 6000):
                chunk = masked[i:i + 6000]
                _tls.in_strict = True
                _tls.quiet = True
                try:
                    reply = llm.chat(sub_cfg, [
                        {"role": "system", "content": STRICT_SYSTEM},
                        {"role": "user", "content": chunk},
                    ])
                except Exception as e:
                    self._strict_retry_at = time.monotonic() + 60
                    _requeue(texts[idx:])
                    if not self._strict_failed:
                        self._strict_failed = True
                        _report(f"strict PII check failed ({str(e)[:120]}); "
                                "rule-based masking still applied, retrying later.", "warn")
                    return
                finally:
                    _tls.in_strict = False
                    _tls.quiet = False
                found += parse_strict_reply(reply.content)
            self._strict_failed = False
            for item in found:
                if item in text:
                    self.learn(item, "PII")
            self._strict_done.add(digest)


def _requeue(texts: List[str]) -> None:
    with _pending_lock:
        _pending_strict[:0] = texts
        del _pending_strict[:-50]


STRICT_SYSTEM = """You find personal data in text. List every item of personal data in the
text: names of people, email addresses, phone numbers, postal addresses, birth
dates, bank account numbers, customer or patient identifiers, licence plates.
Ignore placeholders like [REDACTED_...], code identifiers and technical values.
Reply with one item per line, copied EXACTLY as it appears in the text, nothing
else. If there is none, reply NONE."""


def parse_strict_reply(text: str) -> List[str]:
    items = []
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    for line in text.splitlines():
        s = line.strip().lstrip("-*•0123456789.) ").strip().strip("`\"'")
        if not s or s.upper() == "NONE" or len(s) < 3 or len(s) > 200:
            continue
        if PLACEHOLDER_RE.search(s):
            continue
        items.append(s)
    return items


def _looks_json(s: str) -> bool:
    t = s.lstrip()
    return t.startswith("{") or t.startswith("[")


def _in_json_tool_call(before: str) -> bool:
    i = before.rfind("<tool_call>")
    if i == -1 or before.rfind("</tool_call>") > i:
        return False
    return before[i + len("<tool_call>"):].lstrip().startswith("{")


class StreamUnmasker:
    """Unmask a streamed reply delta by delta. A trailing partial placeholder
    ("[REDAC") is held back until the next delta completes it (or proves it
    isn't one)."""

    def __init__(self, red: "Redactor", json_mode: bool = False):
        self.red = red
        self.json_mode = json_mode
        self.buf = ""
        self.raw = ""   # recent masked text, for tool-call context (bounded)

    def _cut(self) -> int:
        i = self.buf.rfind("[")
        if i == -1:
            return len(self.buf)
        tail = self.buf[i:]
        if len(tail) <= _MAX_PLACEHOLDER and "]" not in tail and _PARTIAL_RE.fullmatch(tail):
            return i
        return len(self.buf)

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self.buf += delta
        cut = self._cut()
        raw, self.buf = self.buf[:cut], self.buf[cut:]
        return self._emit(raw)

    def flush(self) -> str:
        raw, self.buf = self.buf, ""
        return self._emit(raw)

    def _emit(self, raw: str) -> str:
        if not raw:
            return ""
        prefix = self.raw
        out = self.red.unmask_text(raw, prefix=prefix, json_mode=self.json_mode)
        # Keep only what can still matter: an open <tool_call> and its text,
        # or a short tail in case a tag is split across deltas.
        ctx = prefix + raw
        i = ctx.rfind("<tool_call>")
        self.raw = ctx[i:] if i != -1 and ctx.rfind("</tool_call>") < i else ctx[-12:]
        return out


# ---- session access -------------------------------------------------------------------

_sessions: Dict[str, Redactor] = {}
_sessions_lock = threading.Lock()


def get(root=None) -> Redactor:
    """The session Redactor for a project root (one mapping per root per process)."""
    key = str(Path(root).resolve()) if root else ""
    with _sessions_lock:
        red = _sessions.get(key)
        if red is None:
            red = Redactor(Path(key) if key else None)
            _sessions[key] = red
        return red


def for_config(cfg) -> Redactor:
    return get(getattr(cfg, "workdir", None))


def reset_sessions() -> None:
    """Forget every mapping (tests)."""
    with _sessions_lock:
        _sessions.clear()
    with _pending_lock:
        _pending_strict.clear()


def check_file(root, path) -> Optional[str]:
    return get(root).check_file(path)


def mask_for_log(root, text: str) -> str:
    """Mask text before it is written to a run log or printed for debugging."""
    try:
        return get(root).mask_text(text or "")
    except Exception:
        return text


def startup_notice(cfg) -> None:
    """Print a warning when masking is off, and a dim line for PII/strict mode."""
    red = for_config(cfg)
    if red.config_error:
        _report(f"redact config error: {red.config_error}", "warn")
    if not red.enabled:
        _report("Masking is OFF (AGENT_REDACT=0): secrets and personal data are sent "
                "to the model as-is.", "warn")
    elif red.strict:
        _report("Masking: secrets + personal data (strict: local PII model also checks "
                "logs, Jira tickets and pastes).")
    elif red.pii:
        _report("Masking: secrets + personal data.")


def _git_unquote(p: str) -> str:
    """Undo git's C-style quoting: "caf\\303\\251.key" -> café.key."""
    p = p.strip()
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        try:
            return codecs.escape_decode(p[1:-1].encode("utf-8"))[0].decode("utf-8", "replace")
        except Exception:
            return p[1:-1]
    return p


def _strip_ab(p: str) -> Optional[str]:
    p = _git_unquote(p.split("\t", 1)[0])
    if p == "/dev/null":
        return None
    if p[:2] in ("a/", "b/"):
        p = p[2:]
    return p or None


_QUOTED_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def diff_paths(section: str) -> List[str]:
    """Every path one diff section touches (old and new), from the header,
    ---/+++ lines and rename/copy lines. Handles spaces and git quoting."""
    paths: List[str] = []

    def add(p: Optional[str]) -> None:
        if p and p not in paths:
            paths.append(p)

    lines = section.split("\n")
    head = lines[0][len("diff --git "):] if lines and lines[0].startswith("diff --git ") else ""
    if head:
        if head.startswith('"'):
            m = _QUOTED_TOKEN_RE.match(head)
            if m:
                add(_strip_ab(m.group(0)))
                add(_strip_ab(head[m.end():].strip()))
        elif head.endswith('"') and ' "' in head:
            a, b = head.rsplit(' "', 1)
            add(_strip_ab(a))
            add(_strip_ab('"' + b))
        else:
            # "a/my keys/x.pem b/my keys/x.pem": split where both halves agree
            cuts = [m.start() for m in re.finditer(r" b/", head)]
            cut = next((c for c in cuts if head[2:c] == head[c + 3:]), cuts[-1] if cuts else -1)
            if cut != -1:
                add(_strip_ab(head[:cut]))
                add(_strip_ab(head[cut + 1:]))
    for line in lines[1:]:
        if line.startswith("@@"):
            break
        for tag in ("--- ", "+++ ", "rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(tag):
                p = line[len(tag):]
                add(_strip_ab(p) if tag in ("--- ", "+++ ") else _git_unquote(p))
        m = re.match(r"Binary files (.+) and (.+) differ$", line)
        if m:
            add(_strip_ab(m.group(1)))
            add(_strip_ab(m.group(2)))
    return paths


def filter_diff(diff: str, root) -> Tuple[str, List[str]]:
    """Drop the sections of a unified diff that touch blocked files."""
    red = get(root)
    out, refused = [], []
    for part in re.split(r"(?m)^(?=diff --git )", diff):
        if not part.startswith("diff --git "):
            out.append(part)
            continue
        paths = diff_paths(part)
        blocked = [p for p in paths if red.file_policy(p).action == "block"]
        if blocked:
            refused.extend(p for p in blocked if p not in refused)
            out.append(part.split("\n", 1)[0] + "\n"
                       "(changes to this file are not shown: it is never sent to the model)\n")
            continue
        for p in paths:
            red.check_file(p)
        out.append(part)
    return "".join(out), refused


# ---- `agent scan` ---------------------------------------------------------------------

def scan_main(argv: List[str], root: Optional[Path] = None) -> int:
    """`agent scan <files|globs>`: list what would be masked, per line, showing
    only the first 4 characters of each value, plus each file's policy."""
    ap = argparse.ArgumentParser(prog="agent scan", description=scan_main.__doc__)
    ap.add_argument("paths", nargs="+", help="files or globs (** allowed)")
    ap.add_argument("--pii", action="store_true", help="include personal data even if off in config")
    ap.add_argument("--dir", "-d", default=None, help="project root (default: current directory)")
    args = ap.parse_args(argv)
    root = Path(args.dir or root or Path.cwd()).resolve()
    red = get(root)
    files: List[Path] = []
    for pat in args.paths:
        p = Path(pat)
        base = p if p.is_absolute() else root / p
        matches = sorted(glob.glob(str(base), recursive=True)) or ([str(base)] if base.exists() else [])
        for m in matches:
            mp = Path(m)
            if mp.is_dir():
                files.extend(sorted(x for x in mp.rglob("*") if x.is_file()
                                    and ".git" not in x.parts))
            elif mp.is_file():
                files.append(mp)
        if not matches:
            print(f"{pat}: no such file", file=sys.stderr)
    total = 0
    for f in dict.fromkeys(files):
        rel = red._rel(f)
        pol = red.file_policy(f)
        label = {"block": "BLOCKED", "warn": "sent masked (warning)", "ok": "sent masked"}[pol.action]
        extra = f" ({pol.reason})" if pol.reason else ""
        if red.is_test_data(rel):
            extra += " (test data: personal data not masked)"
        print(f"{rel}: {label}{extra}")
        if pol.action == "block":
            continue
        try:
            raw = f.read_bytes()
        except OSError as e:
            print(f"  could not read: {e}")
            continue
        if b"\x00" in raw[:8192]:
            print("  binary, skipped")
            continue
        text = raw.decode("utf-8", errors="replace")
        findings = red.find(text, rel, pii=True if args.pii else None)
        for fd in findings:
            line = text.count("\n", 0, fd.start) + 1
            v = fd.value(text)
            shown = v[:4].replace("\n", "\\n") + "…"
            score = "" if fd.secret else f" score {fd.score:.2f}"
            print(f"  {line:>5}  {fd.kind:<16} {shown:<8} ({fd.rule}{score})")
        total += len(findings)
    if not red.enabled:
        print("note: masking is OFF (AGENT_REDACT=0); these values would be sent as-is.")
    print(f"{total} finding(s) in {len(dict.fromkeys(files))} file(s)")
    return 0
