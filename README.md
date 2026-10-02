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
AGENT_TEMPERATURE, AGENT_TIMEOUT, AGENT_MAX_TURNS,
AGENT_DEBUG=1 (print raw server responses), AGENT_STREAM (default 1),
AGENT_RETRIES (default 3, retries 429/502/503/504).
See sections below for AGENT_LINT_*, AGENT_TEST_*, AGENT_NOTES_*,
AGENT_DOC_*, and AGENT_CHECK_OUTPUT_CHARS / AGENT_MAX_FIX_ATTEMPTS.

## How tool calling works without server support
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
- /undo, /diff, /run <cmd> (optionally share output), /tokens, /map, /clear, /reset.
- /lint runs syntax checks; /test runs AGENT_TEST_CMD; /autotest on|off toggles auto-test after edits.
- /review [target] reviews code changes (see below).
- /notes shows the project notes file.
- AGENT_CONTEXT_CHARS (default 48000, ~12k tokens): oldest history is dropped
  to stay under it, and you're warned if the files alone are too big.

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

## Code review
Review code changes with a structured checklist. Works in both coder and agent.

    /review                 # uncommitted changes (git diff HEAD + untracked)
    /review staged          # staged changes (git diff --cached)
    /review main            # changes vs a branch (git diff main...HEAD)
    /review app.py utils.py # specific files (no git needed); globs ok
    coder --review          # one-shot from command line
    coder --review staged
    agent --review main

Reviews check for: correctness, edge cases, error handling, security, resource
leaks, performance, readability, and test coverage. Output is a numbered list
of findings with severity, location, problem, and fix suggestion. Large diffs
are split per file and findings merged automatically.

After a review, coder offers to add the reviewed files so you can follow up
with "fix 1 and 3" in code mode.

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

## KDoc generation (`/doc`)
Add KDoc to undocumented Kotlin declarations. Reads function bodies and generates
accurate descriptions using SEARCH/REPLACE blocks, then verifies that only comments
changed (no code modifications).

    /doc ClassName                 class by simple name
    /doc path/to/File.kt           specific file
    /doc feature/                  all Kotlin files in a folder (>10 asks confirmation)
    /doc --update ClassName        fix outdated @param/@return in existing KDoc
    coder --doc "ClassName"        one-shot from command line
    coder --doc-update --doc "ClassName"  update mode from CLI

What gets documented: classes, interfaces, objects, enums, functions. Skipped:
private declarations, overrides, test files, generated code, already-documented
(unless `--update`). Properties are skipped unless `AGENT_DOC_PROPERTIES=1`.

Style detection: samples up to 20 existing KDoc blocks to detect tag usage
(@param/@return vs prose), summary style (third-person vs imperative), and line width.

    AGENT_DOC_PROPERTIES=0        # include val/var (default: off)
    AGENT_DOC_VISIBILITY=internal # public|internal (default: internal)
    AGENT_DOC_PER_STEP=8          # declarations per LLM batch
    AGENT_DOC_MAX_FIX=2           # fix rounds for failed edit blocks

## Repo map
Both commands index the codebase into a ranked outline (files, classes,
functions, line numbers). See it with `agent --map "query"` or /map in either
REPL. Cached in .agent-cache/ (add it to .gitignore). AGENT_MAP=0 disables it,
AGENT_MAP_CHARS (default 4000) sets its size.
