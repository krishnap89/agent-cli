import os
from dataclasses import dataclass, field
from pathlib import Path


# Sentinel for an unbounded context budget. A large int (not float("inf")) so
# that the existing integer arithmetic on context_chars keeps working: `// 4`
# and `// 2` stay ints, comparisons never trim, and it still serializes as
# plain JSON instead of the invalid `Infinity` literal.
UNLIMITED_CHARS = 1 << 62


def parse_context_chars(value: str) -> int:
    """Parse AGENT_CONTEXT_CHARS. Empty, 0, or inf/infinite/none/unlimited
    (any case) mean no limit; anything else is an int char budget."""
    v = value.strip().lower()
    if v in ("", "0", "inf", "infinite", "infinity", "none", "unlimited", "-1"):
        return UNLIMITED_CHARS
    return int(v)


@dataclass
class Config:
    base_url: str = os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("AGENT_API_KEY", "not-needed")
    model: str = os.getenv("AGENT_MODEL", "qwen")
    # How the model is asked to call tools:
    #   "xml"    -> <tool_call>{json}</tool_call> (Qwen/Gemma; the default)
    #   "action" -> plain-prose "ACTION name key=value" lines, parsed by the
    #               agent. Use it for models whose tool-call markup comes back
    #               empty (e.g. Nemotron returns null content for the XML form
    #               but follows the ACTION protocol reliably).
    tool_format: str = os.getenv("AGENT_TOOL_FORMAT", "xml")
    # Automatic model switching in the micro-agent loop: when on, each role runs
    # on the model mapped to its job instead of the single cfg.model. Off by
    # default, so nothing changes unless you set the map below.
    auto_model: bool = os.getenv("AGENT_AUTO_MODEL", "0") not in ("0", "false")
    # JSON object mapping a role name to the model that role should run on, e.g.
    #   AGENT_MODEL_MAP='{"Editor":"code-model","Planner":"reason-model",
    #                     "Locator":"fast-model","Reviewer":"review-model"}'
    # Roles: Locator, Planner, Editor, Fixer, Reviewer, Summarizer. Any role not
    # named here (or an invalid/empty map) falls back to cfg.model. A "default"
    # key sets the fallback for all unnamed roles.
    model_map: str = os.getenv("AGENT_MODEL_MAP", "")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
    timeout: int = int(os.getenv("AGENT_TIMEOUT", "300"))  # seconds per LLM request
    stream: bool = os.getenv("AGENT_STREAM", "1") not in ("0", "false")  # avoids 504s
    retries: int = int(os.getenv("AGENT_RETRIES", "3"))  # on 429/502/503/504
    use_map: bool = os.getenv("AGENT_MAP", "1") not in ("0", "false")
    map_chars: int = int(os.getenv("AGENT_MAP_CHARS", "4000"))  # repo map size in the prompt
    context_chars: int = parse_context_chars(os.getenv("AGENT_CONTEXT_CHARS", ""))  # default no limit; set chars to cap
    max_reflections: int = int(os.getenv("AGENT_MAX_REFLECTIONS", "2"))  # coder: retries for failed edits
    max_fix_attempts: int = int(os.getenv("AGENT_MAX_FIX_ATTEMPTS", "2"))
    max_turns: int = int(os.getenv("AGENT_MAX_TURNS", "30"))
    lint_enabled: bool = os.getenv("AGENT_LINT", "1") not in ("0", "false")
    lint_cmd: str = os.getenv("AGENT_LINT_CMD", "")
    test_cmd: str = os.getenv("AGENT_TEST_CMD", "")
    auto_test: bool = os.getenv("AGENT_AUTO_TEST", "0") not in ("0", "false")
    test_timeout: int = int(os.getenv("AGENT_TEST_TIMEOUT", "600"))
    check_output_chars: int = int(os.getenv("AGENT_CHECK_OUTPUT_CHARS", "4000"))
    android: str = os.getenv("AGENT_ANDROID", "auto")  # auto|on|off
    android_variant: str = os.getenv("AGENT_ANDROID_VARIANT", "Debug")
    compile_enabled: str = os.getenv("AGENT_COMPILE", "")  # on|off, default on when android detected
    compile_timeout: int = int(os.getenv("AGENT_COMPILE_TIMEOUT", "600"))
    gradle_args: str = os.getenv("AGENT_GRADLE_ARGS", "--offline --console=plain -q")
    ktlint_mode: str = os.getenv("AGENT_KTLINT", "syntax")  # syntax|full|off
    testgen_examples: int = int(os.getenv("AGENT_TESTGEN_EXAMPLES", "2"))
    testgen_max_fix: int = int(os.getenv("AGENT_TESTGEN_MAX_FIX", "3"))
    testgen_methods_per_step: int = int(os.getenv("AGENT_TESTGEN_METHODS_PER_STEP", "4"))
    testgen_plan: bool = os.getenv("AGENT_TESTGEN_PLAN", "1") not in ("0", "false")
    testgen_allow_gradle_edit: bool = os.getenv("AGENT_TESTGEN_ALLOW_GRADLE_EDIT", "0") not in ("0", "false")
    doc_properties: bool = os.getenv("AGENT_DOC_PROPERTIES", "0") not in ("0", "false")
    doc_visibility: str = os.getenv("AGENT_DOC_VISIBILITY", "internal")  # public|internal
    doc_per_step: int = int(os.getenv("AGENT_DOC_PER_STEP", "8"))
    doc_max_fix: int = int(os.getenv("AGENT_DOC_MAX_FIX", "2"))
    # Route plain-language messages like "add tests for Foo" to /test-gen or
    # /doc (after a y/n confirm) via one cheap classifier call. Set to 0 to send
    # every non-slash message straight to the normal loop.
    intent_routing: bool = os.getenv("AGENT_INTENT_ROUTING", "1") not in ("0", "false")
    notes_enabled: bool = os.getenv("AGENT_NOTES", "1") not in ("0", "false")
    notes_file: str = os.getenv("AGENT_NOTES_FILE", "")
    notes_chars: int = int(os.getenv("AGENT_NOTES_CHARS", "3000"))
    skill_chars: int = int(os.getenv("AGENT_SKILL_CHARS", "3000"))
    # Jira Cloud (read-only). When all three are set, the jira_issue and
    # jira_search tools are offered to the model; otherwise they stay hidden.
    # The token is an API token from id.atlassian.com -> Security -> API tokens.
    jira_url: str = os.getenv("AGENT_JIRA_URL", "")      # e.g. https://you.atlassian.net
    jira_email: str = os.getenv("AGENT_JIRA_EMAIL", "")  # the account the token belongs to
    jira_token: str = os.getenv("AGENT_JIRA_TOKEN", "")
    jira_timeout: int = int(os.getenv("AGENT_JIRA_TIMEOUT", "30"))  # seconds per Jira request
    debug: bool = os.getenv("AGENT_DEBUG", "") not in ("", "0", "false")
    # Append every outgoing HTTP request and incoming reply to this file.
    # Quieter than AGENT_DEBUG (which prints to the console); the Authorization
    # bearer token is redacted. Empty = off.
    http_log: str = os.getenv("AGENT_HTTP_LOG", "")
    # vLLM: keep special tokens such as <tool_call> in the output. Qwen's tool-call
    # markup is made of special tokens, which vLLM strips by default, so the
    # model's tool call silently vanishes. Set to 0 for servers that reject it.
    keep_special_tokens: bool = os.getenv("AGENT_KEEP_SPECIAL_TOKENS", "1") not in ("0", "false")
    # Qwen3 thinking: "" = server default, "0" = off (faster), "1" = on.
    thinking: str = os.getenv("AGENT_THINKING", "")
    # Show a one-line dim gist of the model's reasoning before each reply.
    # Set to 0 to hide it (the full reasoning is still discarded either way).
    show_thinking: bool = os.getenv("AGENT_SHOW_THINKING", "1") not in ("0", "false")
    # Extra JSON merged into every request body, e.g. '{"top_p": 0.8}'.
    extra_body: str = os.getenv("AGENT_EXTRA_BODY", "")
    workdir: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False


# The server has no native tool calling, so tools are described in the prompt
# and the model emits calls as text. Qwen models are trained on this
# <tool_call> format, so they follow it reliably.
SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
Working directory: {workdir}

# Tools
You can call these tools:

{tool_docs}

# How to call a tool
Write a tool call on its own, in exactly this format, with valid JSON:

<tool_call>
{{"name": "search", "arguments": {{"pattern": "LoginViewModel"}}}}
</tool_call>

Rules:
- You may make one or more tool calls in a reply, then STOP and wait.
- Results come back in the next message inside <tool_response> tags.
- Never write a <tool_response> yourself or guess what a tool will return.
- When the task is done, reply normally with NO tool call. That ends your turn.

{repo_map_section}# Guidelines
- To find where a class, function or text is defined or used, call search with a plain
  name first, e.g. search(pattern="LoginViewModel"). Don't list every folder.
- To see what a file depends on and what depends on it, call dependency_graph(path).
- Use repo_map(query) when you only know the topic, not a name.
- For big files, use outline first, then read_file with start_line/end_line.
- Read only what the task needs: your context is small.
- Prefer small edits with edit_file over rewriting whole files.
- After changing code, run tests or the program to verify when possible.
- When asked how something works or is implemented, use tools to find and read the code first. Never answer from memory alone.
- Keep the final answer short: what you changed and why.
"""


# Used when cfg.tool_format == "action": a plain-prose tool protocol for models
# that return empty content for the <tool_call> XML form (e.g. Nemotron). The
# model names a tool and its arguments on an ACTION line; the agent parses the
# line and runs the tool, so no JSON or special-token markup is needed.
ACTION_SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
Working directory: {workdir}

# Tools
You can use these tools:

{tool_docs}

# How to use a tool
You cannot run anything yourself. To use a tool, write an ACTION line: the word
ACTION, then the tool name, then each argument as key=value, all on one line.

ACTION list_dir path=.
ACTION read_file path=app/build.gradle.kts start_line=1 end_line=80
ACTION search pattern=LoginViewModel
ACTION run_command command=./gradlew :app:assembleDebug

For edit_file, put the exact text to replace and its replacement in OLD/NEW
blocks on the lines right after the ACTION line (path= is the only key=value):

ACTION edit_file path=app/src/main/java/com/example/Login.kt
<<<OLD
val timeout = 30
OLD
<<<NEW
val timeout = 60
NEW

For tools whose value is a list (apply_patch, todo), pass a single arg= holding
inline JSON for that one argument:

ACTION todo arg={{"action": "set", "items": [{{"text": "read login flow", "status": "in_progress"}}]}}
ACTION apply_patch arg=[{{"path": "a.kt", "old_str": "x", "new_str": "y"}}]

Rules:
- Put ACTION lines and their OLD/NEW blocks by themselves. You may emit several
  ACTION lines in one reply; each runs and its result comes back next message.
- Values with spaces are fine: everything after the first = up to the end of the
  line (or the next key= for simple tools) is the value. Do not add quotes.
- Never write a tool result yourself or guess what a tool returns.
- When the task is done, reply in plain prose with NO ACTION line. That ends
  your turn.

{repo_map_section}# Guidelines
- To find where a class, function or text is defined or used, use search with a
  plain name first, e.g. ACTION search pattern=LoginViewModel. Don't list every folder.
- To see what a file depends on and what depends on it, use dependency_graph.
- Use repo_map with a query when you only know the topic, not a name.
- For big files, use outline first, then read_file with start_line/end_line.
- Prefer small edits with edit_file over rewriting whole files.
- After changing code, run tests or the program to verify when possible.
- When asked how something works, use tools to find and read the code first.
  Never answer from memory alone.
- Keep the final answer short: what you changed and why.
"""


# Used in chat-only mode: no tools, no file access, just conversation.
CHAT_SYSTEM_PROMPT = """You are a helpful assistant for software developers.
Answer clearly and concisely. Use markdown code blocks for code.
You cannot see the user's files in this mode; if you need code, ask them to paste it.
"""


REPO_MAP_SECTION = """# Repo map
Most important files and symbols in this codebase (line numbers on the left).
It is not complete: call repo_map with a query to find other code.

{repo_map}

"""
