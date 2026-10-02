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
- AGENT_CONTEXT_CHARS (default 48000, ~12k tokens): oldest history is dropped
  to stay under it, and you're warned if the files alone are too big.

## Repo map
Both commands index the codebase into a ranked outline (files, classes,
functions, line numbers). See it with `agent --map "query"` or /map in either
REPL. Cached in .agent-cache/ (add it to .gitignore). AGENT_MAP=0 disables it,
AGENT_MAP_CHARS (default 4000) sets its size.
