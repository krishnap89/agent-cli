# agent-cli

A minimal coding agent for your terminal that works with a local
OpenAI-compatible server (vLLM, llama.cpp, Ollama, LM Studio...) running Qwen.
It talks to the server over plain HTTP (stdlib urllib, no openai SDK) and
does NOT need native tool/function calling: tools are described in the
system prompt and the agent parses `<tool_call>` blocks from plain text.

## Setup
    pip install -e .
    export AGENT_BASE_URL=http://localhost:8000/v1
    export AGENT_MODEL=Qwen2.5-Coder-32B-Instruct   # name your server reports at /v1/models
    export AGENT_API_KEY=not-needed                 # set if your server requires one

## Use
    agent --check                           # test server, list model names
    agent                                   # interactive REPL
    agent "add type hints to utils.py"
    agent --chat                            # chat only: no tools, no file access
    agent -c "explain Python decorators"
    agent -u http://gpu-box:8000/v1 -m qwen -y "run the tests and fix failures"

In the REPL: /chat and /agent switch modes, /clear resets, /exit quits.
Multi-line: pasted text is kept together (press Enter on an empty line to send),
or type """ on its own line to start and end a multi-line message.

Env: AGENT_BASE_URL, AGENT_MODEL, AGENT_API_KEY, AGENT_MAX_TOKENS,
AGENT_STYLE (full|lite|off, default full: see "Change style"),
AGENT_TEMPERATURE, AGENT_TIMEOUT, AGENT_MAX_TURNS,
AGENT_DEBUG=1 (print raw server responses),
AGENT_HTTP_LOG=<path> (append every HTTP request and reply to that file, bearer
token redacted, secrets masked; unset = off), AGENT_STREAM (default 1),
AGENT_RETRIES (default 3, retries 429/502/503/504).
AGENT_KEEP_SPECIAL_TOKENS (default 1: sends vLLM `skip_special_tokens: false`, so
Qwen's `<tool_call>` markup isn't stripped; set 0 for servers that reject it),
AGENT_THINKING (unset = server default, 0 = Qwen3 thinking off, 1 = on),
AGENT_EXTRA_BODY (JSON merged into every request, e.g. '{"top_p": 0.8}').
AGENT_AUTO_MODEL (default 0: when 1, the micro-agent edit loop runs each role on
the model mapped to its job by AGENT_MODEL_MAP instead of the single AGENT_MODEL),
AGENT_MODEL_MAP (JSON {role: model}; roles are Locator, Planner, Editor, Fixer,
Reviewer, Summarizer, e.g.
'{"Editor":"code-model","Fixer":"code-model","Planner":"reason-model","Locator":"fast-model","Reviewer":"review-model"}';
a "default" key sets the model for any unnamed role, else such roles use
AGENT_MODEL; a "pii" key names the local model used by strict masking).
AGENT_REDACT (default 1; 0 turns masking off), AGENT_REDACT_PII (default 0),
AGENT_REDACT_STRICT (default 0): see "Masking secrets and personal data".
See sections below for AGENT_LINT_*, AGENT_TEST_*, AGENT_NOTES_*,
AGENT_DOC_*, AGENT_JIRA_* (read-only Jira), and
AGENT_CHECK_OUTPUT_CHARS / AGENT_MAX_FIX_ATTEMPTS.

## How tool calling works without server support

Tool calls are read from three places, so they work whatever the server does:
the text (`<tool_call>{json}</tool_call>`, or the XML form newer Qwen models use:
`<tool_call><function=search><parameter=pattern>Auth</parameter></function></tool_call>`),
and native `tool_calls` if the server's tool parser is enabled.

If the agent says "Model returned an empty reply twice" and the model "only
produced thinking text", the tool call was generated but removed by the server.
Keep AGENT_KEEP_SPECIAL_TOKENS=1 (the default).

1. System prompt lists tools and the <tool_call>{json}</tool_call> format.
2. Model replies with text; `stop=["<tool_response>"]` prevents it from
   inventing results.
3. agent.py extracts the JSON, runs the tool, and sends output back as a
   user message wrapped in <tool_response>.
4. Malformed JSON is reported back to the model so it can retry.
5. A reply with no tool call ends the turn. <think> blocks are stripped.

## Extending
Add a schema to TOOL_SCHEMAS in tools.py and a matching `t_<name>` method.

## coder: lightweight Aider-style assistant
A second command with Aider's workflow, built on the same HTTP client,
repo map and settings:

    coder src/billing/tax.py            # start with a file in the chat
    coder --ask                         # questions only, no edits
    coder -y --message "add docstrings" app/util.py   # one-shot

- You choose the files: /add puts their full contents in every request,
  /read adds reference-only files, /drop removes them.
- The repo map summarises everything else (files in the chat are excluded).
- The model edits with SEARCH/REPLACE blocks in plain text. Matching tolerates
  whitespace and indentation mistakes; failed blocks are sent back with the
  closest real lines so the model can fix them (AGENT_MAX_REFLECTIONS, default 2).
- If the model names a repo file that isn't in the chat, coder offers to add
  it and re-send your request.
- /micro <request> runs the request through the micro-agent loop (see below).
- /undo, /diff, /run <cmd> (optionally share output), /tokens, /map, /clear, /reset.
- /lint runs syntax checks; /test runs AGENT_TEST_CMD; /autotest on|off toggles auto-test after edits.
- /review [target] reviews code changes (see below).
- /debt lists deliberate shortcuts (see "Change style" below).
- /notes shows the project notes file.
- AGENT_CONTEXT_CHARS (default: no limit): set a char budget to cap context —
  then the oldest history is dropped to stay under it, you're warned if the
  files alone are too big, and /review splits per file past 50% of the budget.
  Left unset or set to 0/inf/infinite/none/unlimited, nothing is trimmed.

## Change style: smallest complete change
coder and the `/micro` Editor add a short set of rules to their system prompt,
adapted from the MIT-licensed [ponytail](https://github.com/DietrichGebert/ponytail)
skill by Dietrich Gebert: make the smallest change that fully solves the request,
reuse existing helpers, no speculative options or reformatting, prefer deleting,
add a test with new branching logic, mark shortcuts, and end with a `Risk:` line.

    AGENT_STYLE=full   # default: all rules (8 lines)
    AGENT_STYLE=lite   # 3 lines: smallest change, don't touch unrelated code, Risk line
    AGENT_STYLE=off    # no rules (the baseline for measuring diff size)

**Risk note.** Every coder or `/micro` reply that edits files ends with one line
built from facts, plus the model's own `Risk:` line (which is removed from the
reply above so it isn't shown twice):

    Risk: lint ok, compile ok · tests not run · 1 caller not updated: MainActivity.onCreate (app/src/main/java/com/example/app/MainActivity.kt:23) · model: callers must pass force

"not compiled" means only lint ran (no Gradle compile check for these files).
"callers not updated" lists calls, from files the edit didn't touch, into
functions whose declaration line changed, plus calls to a function the edit
removed or renamed (found by name, so only names no other function in the project
has). Kotlin, Swift and TS/JS only: the code graph's languages, and the graph is
not loaded when no changed file is one of them. The model's line is the latest
`Risk:` line from the reply or its lint/test fix rounds; `Risk:` lines inside code
blocks are ignored. Replies that edit nothing (a question back to you) get no note.

**Shortcuts.** When the model knowingly takes a shortcut it is asked to leave a
comment `shortcut: <limit>, <when to upgrade>` (`;` also separates the parts).
`/debt` (coder and agent, or `agent debt [dir]`) lists them by file, scanning the
repo map's source files; a marker inside a string literal or docstring is ignored.
For example:

    2 shortcuts in 1 file:
    core/data/src/main/java/com/example/data/UserRepository.kt
         25  in-memory only  ->  upgrade: when the app runs more than one process
         31  no eviction  ->  upgrade: (no upgrade trigger given)

**Measuring diff size.** `python -m agent_cli.evaldiff` runs each task in
`evals/diffsize/tasks.json` (8 small edits on the Kotlin, TS/JS and Swift test
fixtures, each in a temp copy) once per style against your model server and prints
changed lines (+/-) per task, with the total change against the first style:

    python -m agent_cli.evaldiff                         # off (before) vs full (after), coder
    python -m agent_cli.evaldiff --styles off,lite,full --micro --repeat 3

## Automatic checks after edits
After applying edits, coder runs built-in lint (Python syntax, JSON validity,
XML well-formedness) and optionally a custom lint command and tests. Errors
are fed back to the model for automatic fixing (up to AGENT_MAX_FIX_ATTEMPTS
rounds, default 2).

    AGENT_LINT=1            # enable/disable built-in lint (default: on)
    AGENT_LINT_CMD="ruff check {files}"  # custom lint; {files} is replaced
    AGENT_LINT_CMD_KT="ktlint {files}"   # per-extension lint command
    AGENT_LINT_CMD_XML="xmllint {files}"  # (AGENT_LINT_CMD_<EXT>)
    AGENT_TEST_CMD="python -m pytest"    # test command for /test and auto-test
    AGENT_AUTO_TEST=1       # run tests after every edit (default: off)
    AGENT_TEST_TIMEOUT=600  # seconds before test command is killed
    AGENT_CHECK_OUTPUT_CHARS=4000  # max chars of check output sent to model
    AGENT_MAX_FIX_ATTEMPTS=2      # auto-fix rounds for lint/test failures

Per-extension lint commands (`AGENT_LINT_CMD_<EXT>`, e.g. `AGENT_LINT_CMD_KT`)
run on the changed files of that extension only. Lookup order per file:
`AGENT_LINT_CMD_<EXT>`, then `AGENT_LINT_CMD`, then built-in check.

## Micro-agent loop (`/micro`)
`/micro <request>` (and `coder --micro "<request>"`) runs an edit request
through a small pipeline of focused roles instead of one long chat. Each role
sees only its own inputs — there is no shared history to drift — and produces a
strict format that is parsed in Python; a bad-format reply is retried once, then
the run stops with a clear message.

    Locator -> Planner -> for each step:
        Editor -> Checker -> (Fixer -> Checker)* -> Reviewer
    -> Summarizer

- **Locator** picks the files to edit from the repo map (it may name a new file
  to create, as long as the path is sane — a known source extension inside an
  existing directory; wild paths are ignored). **Planner** breaks the request
  into small ordered steps, and each step's files are validated against the
  Locator's set: a step referencing a path that was never located has it
  dropped (and the plan is re-requested once if nothing valid is left), so steps
  can't send the Editor after an invented file.
- For each step, **Editor** writes SEARCH/REPLACE blocks, **Checker** (no LLM:
  built-in lint + the Gradle compile check already used after normal edits)
  verifies them, and **Fixer** repairs failures for up to `AGENT_FIX_ROUNDS`
  rounds. The step stops early if the *same* check error comes back twice.
  Edits are scoped to the step's files: a block aimed at any other path (an
  invented path, or the format example) is rejected and sent back, never written.
- Before writing, the Editor also gets **code graph facts** (Kotlin, Swift,
  TS/JS): callers outside the step's files of the functions it is changing (those
  named in the step, else all in the files), the tests `agent graph impact` says
  are affected, and up to 5 existing helpers elsewhere whose names resemble the
  step ("reuse candidates"). `AGENT_GRAPH_FACTS=0` turns this off.
- **Reviewer** judges each step's diff against its intent. A CHANGES verdict
  feeds its note back to the Fixer (so "compiles but doesn't match the step" is
  caught too), up to `AGENT_REVIEW_ROUNDS` extra passes; the edits are kept
  either way. **Summarizer** writes the final one-paragraph summary from the
  actual diff.
- **Fast path:** if the request names exactly one file already in the chat, the
  Locator and Planner are skipped and the request runs as a single step.
- The active role and round are shown in the terminal. Every role call is logged
  to `.agent-cache/runs/<timestamp>.jsonl` (input size, output, duration).
- The whole run is one undo snapshot: a single `/undo` reverts everything.

```
AGENT_FIX_ROUNDS=2      # Fixer rounds per step for a failed check (default: AGENT_MAX_FIX_ATTEMPTS)
AGENT_REVIEW_ROUNDS=1   # extra Fixer passes driven by a Reviewer CHANGES verdict (0 disables)
```

## Kotlin / Android support
For Kotlin Android projects built with Gradle, coder detects modules
automatically and runs compile checks after edits.

**Detection.** `AGENT_ANDROID=auto|on|off` (default `auto`). In auto mode, the
project is detected as Android if it has `gradlew`, `settings.gradle(.kts)`,
and at least one `build.gradle(.kts)` applying an Android plugin (including
version catalog aliases in `libs.versions.toml`).

**Module mapping.** Each changed file is mapped to its nearest Gradle module
(the closest ancestor with `build.gradle(.kts)`). Android modules use
`compile<Variant>Kotlin`; plain Kotlin modules use `compileKotlin`; XML-only
changes in Android modules use `process<Variant>Resources`.

**Compile check.** After lint passes, `./gradlew <tasks> <AGENT_GRADLE_ARGS>`
runs. Kotlin compiler errors are parsed, de-duplicated, and sent to the model
with code context. Errors only in files you didn't change are not sent.

    AGENT_COMPILE=on|off     # default: on when Android detected
    AGENT_COMPILE_TIMEOUT=600
    AGENT_ANDROID_VARIANT=Debug
    AGENT_GRADLE_ARGS="--offline --console=plain -q"

**ktlint (optional).** If `ktlint` is on PATH and `AGENT_KTLINT` is not `off`,
it runs before the compile check. `syntax` (default) only reports parse
failures; `full` reports all violations.

    AGENT_KTLINT=syntax|full|off

**Test command examples for Android:**

    AGENT_TEST_CMD="./gradlew :app:testDebugUnitTest --offline --console=plain -q"

**New commands:**

    /compile [all]    run compile check on modules of files in chat
    /modules          list detected modules and file-to-module mapping

## Gradle and the corporate proxy
Gradle (a Java program) ignores `HTTP_PROXY` / `HTTPS_PROXY`. Proxy settings
go in `~/.gradle/gradle.properties`:

    systemProp.http.proxyHost=proxy.corp.example.com
    systemProp.http.proxyPort=8080
    systemProp.http.proxyUser=user
    systemProp.http.proxyPassword=pass
    systemProp.http.nonProxyHosts=localhost|127.0.0.1
    systemProp.https.proxyHost=proxy.corp.example.com
    systemProp.https.proxyPort=8080
    systemProp.https.proxyUser=user
    systemProp.https.proxyPassword=pass

Recommendation: keep `--offline` in `AGENT_GRADLE_ARGS` once all dependencies
are cached by a normal Android Studio build. If `gradlew` isn't executable,
run `chmod +x gradlew`.

## Jira (read-only)
Give the agent read access to Jira Cloud issues so you can say "implement
PROJ-123" and it pulls the ticket, or ask it to look over open bugs. Set three
env vars (the token is an API token from id.atlassian.com → Security → API
tokens, not your password):

    AGENT_JIRA_URL=https://you.atlassian.net
    AGENT_JIRA_EMAIL=you@company.com
    AGENT_JIRA_TOKEN=<api-token>
    AGENT_JIRA_TIMEOUT=30     # seconds per Jira request (default 30)

With all three set, two tools appear:

- `jira_issue(key)` — read one issue (summary, status, assignee, description,
  comments), e.g. `jira_issue(key="PROJ-123")`.
- `jira_search(jql, max_results=20)` — run a JQL query, e.g.
  `jira_search(jql='project = PROJ AND status = "In Progress"')`.

Both are read-only: the agent never creates, comments on, or transitions
issues. If any of the three vars is unset the tools are hidden from the model.

### `/jira-plan <ISSUE-KEY>` — ticket → subtasks
A read-only pipeline (in `coder`, code mode) that turns one ticket into an
ordered plan:

    /jira-plan PROJ-123

It fetches the issue, locates the code it concerns (the same keyword-search +
Locator step the `/micro` analysis pipeline uses), then runs the Planner to
break the work into small, ordered subtasks, each naming the files it would
touch. It prints the plan and stops — no source is edited, no document is
written, and nothing is sent back to Jira. Implement a step afterwards with
`/micro`.

## Masking secrets and personal data
Before anything is sent to the model, secrets (and optionally personal data)
are replaced with placeholders; the reply is restored before you or the code
see it. This happens in one place (llm.py), so it covers every request: agent,
coder, `/micro` roles, `/review`, `/doc`, `/test-gen`, Jira tickets and the web UI.

    password = "hunter2"   is sent as   password = "[REDACTED_PASSWORD_1]"

- Only the value is replaced. The same value always gets the same placeholder
  for the whole session, so prompts stay stable (prefix caching still works).
- Replies are unmasked, including streamed text (a placeholder split across
  chunks is held back until it is complete), reasoning, and tool-call
  arguments. A SEARCH/REPLACE block or `edit_file` call written around a
  placeholder therefore edits the real file with the real value.
- The value/placeholder mapping lives in memory only. Real values are never
  written to disk: the HTTP log (`AGENT_HTTP_LOG`), `AGENT_DEBUG` output and the
  `/micro` run logs in `.agent-cache/runs/` contain placeholders.
- After each request a dim line says what was masked, never the values:
  `masked before sending: 2 passwords, 1 email`.

**Secrets (always on):** private key blocks; AWS, GitHub, GitLab, Slack,
Google (`AIza…`), Stripe, Atlassian (`ATATT…`) and LLM API keys (OpenAI,
Anthropic, Hugging Face, Groq, …); JWTs; passwords in URLs; `Authorization`
headers; quoted values assigned to secret-like names (`password`, `secret`,
`token`, `api_key`, `client_secret`, `credentials`, …, also `dbPassword`,
`AUTH_TOKEN`); unquoted values in `.env`, `.properties`, YAML and `key=value`
log lines; Android `storePassword`/`keyPassword` and secret `buildConfigField`s.
Not masked: type names (`password: String`), `${VAR}`, `$VAR`, `<...>`,
`****`, `changeme` and other obvious dummies, and allowlisted values.

**Personal data (off by default; `AGENT_REDACT_PII=1` or `"pii": true`):**
each match is validated and scored, and only matches at or above the threshold
(default 0.5) are masked.
- email (not example.com/.org/.net or test domains), card numbers (Luhn),
  IBAN (country length + mod-97), phone numbers (international and German
  formats, plausible length), German tax ID (checksum), IPv4/IPv6 (not
  private/loopback).
- Words near a match raise or lower its score: `Tel`, `IBAN`, `Kontonummer`,
  `email` raise it; `version`, `timestamp`, `id` lower it.
- Structured data: values of fields named email, phone, iban, birth_date,
  address, first_name, last_name (and variants) in JSON, YAML, CSV, SQL
  `INSERT` dumps and `key=value` logs.
- Names and cities listed one per line in `.agent/redact-names.txt`.
- Test data is left alone: files under `src/test`, `src/androidTest`,
  `fixtures` (plus `allow_paths`) keep their personal data; secrets in them are
  still masked.
- **Strict mode** (`AGENT_REDACT_STRICT=1` or `"strict": true`, implies PII):
  logs (`/run` output, check failures, `run_command`), Jira tickets and pasted
  text over 500 characters are also shown, already rule-masked, to a local
  model, which lists the personal data it sees; those values are masked too.
  The model is the `pii` entry of `AGENT_MODEL_MAP` (e.g.
  `'{"pii": "qwen2.5-3b-instruct"}'`), else its `default`, else `AGENT_MODEL`
  (with a warning, since that is the main model). `AGENT_REDACT_PII_URL` (or
  `"pii_base_url"` in the config) sends the check to a separate local server.
  It only adds to the rule-based detection: if the model fails you get a
  warning, the rules still apply, and the unchecked texts are retried a
  minute later.

**Files.** Some files are never sent. `/add`, `/read`, `read_file`, `search`
results, `run_command` (when the command names one), `@mentions`, `/review`
(including diffs), `/doc`, `/test-gen` sources, the repo map and the web UI
refuse them with a clear message, and llm.py withholds any that still turn up
embedded in a prompt (even with `AGENT_REDACT=0`):
- `*.jks`, `*.keystore`, `*.p12`, `*.pfx`, `*.pem`, `*.key`, `*.ppk`, SSH
  private keys (`id_rsa`, `id_ed25519`, …), and any file that starts with a
  private key block;
- anything matched by `.agentignore` in the project root (gitignore syntax:
  `*`, `**`, `dir/`, `!negation`).

These are sent masked, with a one-time warning: `.env*`, `local.properties`,
`gradle.properties`, `keystore.properties`, `secrets.properties`,
`signing.properties`, `google-services.json`, `GoogleService-Info.plist`,
`.netrc`, `.npmrc`, `.pypirc`.

**Config** in `<project>/.agent/redact.json` (all keys optional):

    {
      "pii": true,
      "strict": false,
      "threshold": 0.5,
      "allow": ["demo-password-for-docs"],
      "allow_paths": ["**/sampledata/**"],
      "pii_base_url": "http://localhost:11434/v1",
      "patterns": [{"name": "internal_token", "regex": "itk_([A-Za-z0-9]{24})", "group": 1}]
    }

`allow` values are never masked; `allow_paths` are extra test-data paths;
`patterns` add secret rules (`group` picks the part to mask, default the whole
match). The env vars override the file. `AGENT_REDACT=0` turns masking off and
prints a warning at startup; blocked files stay blocked.

**Check before you send:** `agent scan <files|globs|dirs>` lists, per line,
what would be masked (kind, rule, score, and only the first 4 characters of
each value) and each file's policy (blocked, sent masked, test data):

    agent scan .env app/src/main/**/*.kt --pii
    .env: sent masked (warning) (may contain secrets; sent masked)
          3  PASSWORD         supe…    (assignment_line)

## Project notes
Create an AGENT.md (or CONVENTIONS.md, .agent/notes.md) in your project root
to give the model persistent instructions (coding style, architecture notes,
etc.). Notes are included in every request.

    agent --init-notes      # generate AGENT.md from your codebase
    coder --init-notes
    /notes                  # show the current notes in the REPL

    AGENT_NOTES=1           # enable/disable (default: on)
    AGENT_NOTES_FILE=path   # override the default file search
    AGENT_NOTES_CHARS=3000  # max chars of notes included in context

## Skills
Skills are reusable instruction sets that live in your project at
`.agent/skills/<name>/SKILL.md`. Each SKILL.md has front matter:

    ---
    name: kotlin-style
    description: Kotlin coding conventions
    triggers:
      - kotlin
      - style
      - convention
    ---
    Your markdown instructions here.

Skills are loaded at startup in both `agent` and `coder`. For each user
message, the top 1–2 skills whose trigger/description words match are
auto-selected and injected as context (the model doesn't choose skills).

    /skills              list loaded skills
    /skill <name>        pin a skill (stays active until /skill off)
    /skill off           unpin the current skill
    /tokens              shows skill size in the context budget

    AGENT_SKILL_CHARS=3000  # max chars per skill (default 3000)

## Code review
Review code changes with a structured checklist. Works in both coder and agent.

    /review                 # uncommitted changes (git diff HEAD + untracked)
    /review staged          # staged changes (git diff --cached)
    /review main            # changes vs a branch (git diff main...HEAD)
    /review abc123..def456  # between two commits (git diff abc123..def456)
    /review abc123 def456   # same, space-separated
    /review abc123..        # from a commit up to HEAD
    /review main...feature  # feature vs its merge-base with main
    /review app.py utils.py # specific files (no git needed); globs ok
    coder --review          # one-shot from command line
    coder --review staged
    agent --review main

Reviews check for: correctness, edge cases, error handling, security, resource
leaks, performance, readability, test coverage, and unnecessary code (anything
that could be deleted or re-implements an existing helper). Output is a numbered list
of findings with severity, location, problem, and fix suggestion. Large diffs
are split per file and findings merged automatically.

After a review, coder offers to add the reviewed files so you can follow up
with "fix 1 and 3" in code mode.

If the diff adds branching logic (`if`/`elif`/`else if`, `when`/`switch`/`case`/
`match`, `guard`, `catch`/`except`, ` ? : `) to source files and changes no test
file, the reviewer is told so (it reports it as a finding) and the terminal shows
`New branching logic without a test change: Foo.kt (+3 branches).` On an Android
project with a Kotlin file flagged, coder offers to run `/test-gen <Class>`
(with `-y` it only prints the command).

## Test generation (`/test-gen`)
Generate JVM unit tests for a Kotlin class that compile, run, and follow your
project's existing test style. Requires an Android/Gradle project.

    /test-gen LoginViewModel                class by simple name
    /test-gen com.example.LoginViewModel    class by fully qualified name
    /test-gen app/src/main/.../Foo.kt       file path
    /test-gen LoginViewModel login          only tests for the method `login`
    coder --test-gen "LoginViewModel"       one-shot from command line
    coder -y --test-gen "LoginViewModel"    auto-accept plan

Steps: resolve the target, detect test libraries from Gradle files, classify the
class (ViewModel, Repository, UseCase, Plain), show a test plan for approval,
write tests with SEARCH/REPLACE blocks, compile, run, and report. Production code
is never changed. One `/undo` reverts everything.

Unsupported: DAO (needs instrumented tests), Activity/Fragment/Composable (needs
UI tests). Detected libraries include JUnit 4/5, MockK, Mockito-Kotlin,
coroutines-test, Turbine, Truth, assertk, Robolectric, and arch-testing.

    AGENT_TESTGEN_EXAMPLES=2            existing test files used as style examples
    AGENT_TESTGEN_MAX_FIX=3             compile + test fix rounds
    AGENT_TESTGEN_METHODS_PER_STEP=4    methods per generation batch
    AGENT_TESTGEN_PLAN=1                ask to approve the plan (0 = auto-accept)
    AGENT_TESTGEN_ALLOW_GRADLE_EDIT=0   allow adding missing test dependencies

## KDoc / Javadoc generation (`/doc`)
Add KDoc to undocumented Kotlin declarations and Javadoc to undocumented Java
declarations. Reads function bodies and generates accurate descriptions using
SEARCH/REPLACE blocks, then verifies that only comments changed (no code
modifications). The language is picked per file from its extension (`.kt` or `.java`).

    /doc ClassName                 class by simple name (Kotlin or Java)
    /doc path/to/File.kt           specific file
    /doc path/to/File.java         specific Java file
    /doc feature/                  all Kotlin and Java files in a folder (>10 asks confirmation)
    /doc --update ClassName        fix outdated @param/@return in existing KDoc/Javadoc
    coder --doc "ClassName"        one-shot from command line
    coder --doc-update --doc "ClassName"  update mode from CLI

What gets documented: classes, interfaces, objects, enums, functions (Kotlin);
classes, interfaces, enums, records, annotation types, methods, constructors (Java).
Skipped: private declarations, overrides (Kotlin `override`, Java `@Override`), test
files, generated code, already-documented (unless `--update`). Properties (Kotlin
val/var, Java fields) are skipped unless `AGENT_DOC_PROPERTIES=1`.

Style detection: samples up to 20 existing KDoc blocks (Javadoc blocks for Java
files) to detect tag usage (@param/@return vs prose), summary style (third-person vs
imperative), and line width.

`AGENT_DOC_VISIBILITY` for Java: `public` documents public and protected members;
`internal` also documents package-private ones. Interface members count as public.

    AGENT_DOC_PROPERTIES=0        # include val/var / Java fields (default: off)
    AGENT_DOC_VISIBILITY=internal # public|internal (default: internal)
    AGENT_DOC_PER_STEP=8          # declarations per LLM batch
    AGENT_DOC_MAX_FIX=2           # fix rounds for failed edit blocks

## Web UI (`agent-ui`)
A local web interface for coder. Same features, in a browser — no external
dependencies, no CDN, works offline.

    agent-ui                        # start on a random port, opens browser
    agent-ui --port 8080            # fixed port
    agent-ui --no-open              # don't open browser
    agent-ui --workdir /path/to/project

Features:
- **Chat**: ask and code modes, streamed replies, stop button, markdown rendering
- **File management**: searchable project tree, add files as editable or read-only,
  remove from chat
- **Diffs**: each edit shown as a diff with an undo button
- **Settings**: server URL, model, temperature, max tokens, context budget,
  API key (never saved to disk), test connection button
- **Token meter**: shows context usage as a percentage bar
- Light and dark mode (follows system), works on narrow screens

Security: binds to 127.0.0.1 only, uses a random session token in the URL
to prevent other websites from calling the API.

## Playground (`agent-playground`)
A local page for checking what the model returns, using the same HTTP client,
masking and reply parsing as the CLI. It never runs tools or edits files.

    agent-playground                        # uses your AGENT_* settings / .env
    agent-playground -u http://gpu-box:8000/v1 -m qwen
    python -m agent_cli.playground          # without re-running pip install

- Edit the system prompt (or load the chat / agent preset), user prompt, sampling
  options, stop sequences, streaming, thinking and extra JSON.
- Send one prompt to up to 4 models side by side; repeat a name to check consistency.
- For each reply: time to first output, total time, token usage (when the server
  reports it), finish reason, thinking, the text the agent would keep, the tool
  calls it would parse, the request body as sent and the raw response.
- Warnings for empty replies, truncation and tool calls that don't parse.

Requests are not retried, so you see the server's first answer or error. The
agent preset leaves out the repo map and project notes. Messages are masked as
usual; the extra JSON field is sent as-is. When streaming, add
`{"stream_options": {"include_usage": true}}` if token usage is missing.
Same security model as agent-ui: 127.0.0.1 only, random token, and the API key
stays in the Python process.

## Repo map
Both commands index the codebase into a ranked outline (files, classes,
functions, line numbers). See it with `agent --map "query"` or /map in either
REPL. Cached in .agent-cache/ (add it to .gitignore). AGENT_MAP=0 disables it,
AGENT_MAP_CHARS (default 4000) sets its size.

Kotlin, Swift and Java are parsed with tree-sitter when the optional grammars are
installed (`pip install ".[treesitter]"`), which gives exact classes, methods,
constructors and fields; without them a regex fallback is used, which can mistake
calls or enum constants for definitions. Installing or removing a grammar
re-indexes the cache automatically.

TypeScript and JavaScript use the code graph's own parser (no extra install). The
map lists classes, interfaces, enums, type aliases, namespaces, functions,
arrow-function `const`s and object-literal modules (`export const api = { … }`), plus
their methods, accessors and constructors, with `export` / `export default`
shown. Local functions and member fields are left out. Only the names a file
exports (or, for a script without imports/exports, all of its top-level names) count
as things other files can depend on. `.vue` files still use the regex patterns.

## Code graph (`agent graph`, Kotlin / Android, Swift / iOS, TypeScript / JavaScript)

A resolved graph of a Kotlin Android project: classes, functions, properties,
Gradle modules and resources, linked by `contains`, `imports`, `calls`,
`inherits`, `injects` (@Inject constructors/fields, `by viewModels()`/`inject()`),
`uses_type`, `overrides`, `depends_on` (Gradle `project(...)`) and `references`
(`R.layout`/`R.id`/... and ViewBinding classes → res XML; layouts and the manifest
→ classes). Pure Python: its own Kotlin and Swift tokenizers/parsers, no tree-sitter.

Swift / iOS works the same way. Modules come from Xcode targets
(`project.pbxproj`, including Xcode 16 synchronized folders) and `Package.swift`
targets, with `depends_on` from target dependencies and package products; without
either, the whole project is one module. `extension Foo` adds members and
conformances to `Foo` (extensions of `String`, `UIView`, … behave like extension
functions), overloads are told apart by argument labels (`select(_:)` vs
`select(id:)` — ids carry them), and methods that implement a protocol requirement
get an inferred `overrides`, so `callers` follows calls made through the protocol.
`injects` covers `@EnvironmentObject`/`@StateObject`/`@ObservedObject`,
Factory/Resolver `@Injected`/`@Inject` and Swinject's `resolve(Foo.self)`.
`references` links `UIImage(named:)`, `Image(_:)`, `UIColor(named:)`,
`NSLocalizedString`, `Text("key")`, storyboard ids and R.swift to asset-catalog
entries, `.strings`/`.xcstrings` keys and storyboards; storyboards/xibs link to their
`customClass` classes and `@IBAction`s, and Info.plist to `$(PRODUCT_MODULE_NAME).X`
classes. Pods, Carthage, DerivedData and `.build` are skipped.

TypeScript and JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`, `.d.ts`)
work the same way, JSX included. Every file is its own namespace, so ids start with
its path without the extension (`src/api/user.UserApi.load`), and imports are
resolved the way the compiler does it: relative paths (a `./x.js` import also finds
`x.ts`), tsconfig/jsconfig `paths` and `baseUrl` (with comments, trailing commas and
`extends` handled), and workspace packages by their `package.json` name. For those,
`main`/`types`/`exports` paths that point at `dist/` or `lib/` are mapped back to
`src/`. An import is then followed through barrel files (`export * from`,
`export { a as b } from`, `export * as ns`), default exports and CommonJS
(`require`, `module.exports = { … }`, `exports.x = …`) to the declaration it names.
`import * as ns` makes `ns.fn()` and `ns.sub.fn()` resolvable. Workspace packages
become modules, with `depends_on` taken from `dependencies` on each other.
Besides classes, interfaces, enums, type aliases, namespaces and functions, the
graph has arrow functions assigned to a `const` or a class field, object-literal
modules (`export const api = { load() {…} }`) and top-level code (`<module>`, which
covers Express routes and Jest `describe`/`it` bodies). `<UserCard … />` is a call of
`UserCard`, and `onClick={handleClick}` is a reference to `handleClick`. A bare call
never resolves to a class member (that takes `this.`), and a name that is neither
local nor imported only matches globals declared in scripts such as `.d.ts` files.
`injects` covers constructor parameters of `@Injectable`/`@Component`/`@Controller`/…
classes (Angular, NestJS, InversifyJS, tsyringe) and Angular's `inject(Foo)`.
Methods that implement an interface or redefine a base-class method get an
inferred `overrides`, with or without the `override` keyword. `node_modules`,
`dist`, `build`, `.next`, `.turbo` and other build outputs are skipped, as are
`*.min.js` files. Vue and Svelte single-file components are not parsed.

    agent graph build [--force]          # index + stats (load time, cache size, call-resolution rate)
    agent graph callers UserRepository.load
    agent graph callees MainActivity.onCreate
    agent graph path MainActivity UserCache [--max-depth 6] [--kinds calls,injects]
    agent graph explain UserRepository
    agent graph impact core/data/src/main/java/com/example/data/UserRepository.kt
    agent graph query "user cache"
    agent graph callers "UserListViewModel.select(_:)"   # Swift: labels pick an overload
    agent graph impact packages/core/src/cache.ts        # TS: through barrels into other workspace packages
    agent graph audit [--limit 20] [--no-explain]       # dead code, near-duplicates, untested risky files

`audit` ranks three kinds of finding on one score, roughly "lines at stake":
`dead` functions with no callers, references or overrides (framework entry
points such as `override`/`open` members, `onCreate`, `viewDidLoad`, `@Test`,
`@IBAction`, `@Preview` and interface requirements are skipped; score = lines);
`dup` pairs of function bodies at least 85% alike by tokens (score = smaller body's
lines x similarity x 1.5); and `untested` code files no test reaches through the
graph, weighted by how many other files depend on them and how much they branch
(score = 3 x dependents + branches / 2 + lines / 100). Only the top 5 go to the
model, with their code, to judge whether each is real and name the smallest fix;
`--no-explain` skips the model.

Symbols can be full ids, `Class.member` or bare names. Output is plain text with
`path:line` and is capped by `--budget` (default 4000 chars, `AGENT_GRAPH_CHARS`).
Each edge says how it was found: `extracted` (scope, imports), `inferred` (via a
receiver's declared type, or a unique project-wide name) or `ambiguous` (several
candidates, all kept), with a confidence. When the receiver type is unknown, calls to
names the standard library uses everywhere (`map`, `filter`, `getString`, ...) are left
unresolved rather than guessed.

Per-file parse results are cached in `.agent-cache/codegraph.json`, re-parsed by
mtime/size; the graph itself is resolved in memory on each load (about 0.1 s
for ~700 files). Storage sits behind `graph.store.FactStore`, so it can be swapped
for SQLite. On Kotlin, Swift and TypeScript/JavaScript repos the repo map is ranked with PageRank over the graph's
call/inherit/inject edges (files without such edges keep the identifier-based
ranking); `AGENT_GRAPH_RANK=0` turns that off.

## Credits
The change-style rules, risk note, `shortcut:` markers, the "what could be deleted"
review item and `agent graph audit` adapt ideas from
[ponytail](https://github.com/DietrichGebert/ponytail) by Dietrich Gebert (MIT
license). The rules text here is a rewrite, not a copy.
