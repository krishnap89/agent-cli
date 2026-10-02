import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    base_url: str = os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("AGENT_API_KEY", "not-needed")
    model: str = os.getenv("AGENT_MODEL", "qwen")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
    timeout: int = int(os.getenv("AGENT_TIMEOUT", "300"))  # seconds per LLM request
    stream: bool = os.getenv("AGENT_STREAM", "1") not in ("0", "false")  # avoids 504s
    retries: int = int(os.getenv("AGENT_RETRIES", "3"))  # on 429/502/503/504
    use_map: bool = os.getenv("AGENT_MAP", "1") not in ("0", "false")
    map_chars: int = int(os.getenv("AGENT_MAP_CHARS", "4000"))  # repo map size in the prompt
    context_chars: int = int(os.getenv("AGENT_CONTEXT_CHARS", "48000"))  # ~12k tokens, coder
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
    notes_enabled: bool = os.getenv("AGENT_NOTES", "1") not in ("0", "false")
    notes_file: str = os.getenv("AGENT_NOTES_FILE", "")
    notes_chars: int = int(os.getenv("AGENT_NOTES_CHARS", "3000"))
    debug: bool = os.getenv("AGENT_DEBUG", "") not in ("", "0", "false")
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
{{"name": "read_file", "arguments": {{"path": "main.py"}}}}
</tool_call>

Rules:
- You may make one or more tool calls in a reply, then STOP and wait.
- Results come back in the next message inside <tool_response> tags.
- Never write a <tool_response> yourself or guess what a tool will return.
- When the task is done, reply normally with NO tool call. That ends your turn.

{repo_map_section}# Guidelines
- Find code with the repo map, repo_map(query) and search. Don't list every folder.
- For big files, use outline first, then read_file with start_line/end_line.
- Read only what the task needs: your context is small.
- Prefer small edits with edit_file over rewriting whole files.
- After changing code, run tests or the program to verify when possible.
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
