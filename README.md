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
See sections below for AGENT_LINT_*, AGENT_TEST_*, AGENT_NOTES_*, and
AGENT_CHECK_OUTPUT_CHARS / AGENT_MAX_FIX_ATTEMPTS.

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
After applying edits, coder runs built-in lint (Python syntax, JSON validity)
and optionally a custom lint command and tests. Errors are fed back to the
model for automatic fixing (up to AGENT_MAX_FIX_ATTEMPTS rounds, default 2).

    AGENT_LINT=1            # enable/disable built-in lint (default: on)
    AGENT_LINT_CMD="ruff check {files}"  # custom lint; {files} is replaced
    AGENT_TEST_CMD="python -m pytest"    # test command for /test and auto-test
    AGENT_AUTO_TEST=1       # run tests after every edit (default: off)
    AGENT_TEST_TIMEOUT=600  # seconds before test command is killed
    AGENT_CHECK_OUTPUT_CHARS=4000  # max chars of check output sent to model
    AGENT_MAX_FIX_ATTEMPTS=2      # auto-fix rounds for lint/test failures

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

## Repo map
Both commands index the codebase into a ranked outline (files, classes,
functions, line numbers). See it with `agent --map "query"` or /map in either
REPL. Cached in .agent-cache/ (add it to .gitignore). AGENT_MAP=0 disables it,
AGENT_MAP_CHARS (default 4000) sets its size.
