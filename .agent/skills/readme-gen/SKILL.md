---
name: readme-gen
description: Generate or refresh the project's README.md from the actual code — install, usage, config, and layout, grounded in what the repo really contains
triggers:
  - readme
  - read me
  - readme.md
  - project documentation
---

## Goal

Produce a `README.md` that reflects what the project **actually** is — read the
code first, never invent features, commands, or dependencies. A README full of
plausible-but-wrong boilerplate (fake install steps, imaginary flags) is worse
than none.

## Output contract

The README is created ONLY by a `write_file` tool call to `README.md` (repo
root, unless the user names another path). The whole Markdown document goes in
the tool call's `content` field — NOT in a ```markdown code block, NOT in prose.
A code block in your reply writes nothing. So your reply is one `write_file`
tool call and no README text anywhere else.

Emit it in this exact format, as your main action, before any report:

    <tool_call>
    {"name": "write_file", "arguments": {"path": "README.md", "content": "# Project Name\n\n...the full README, every newline and quote JSON-escaped...\n"}}
    </tool_call>

The `content` string must be valid JSON: escape every `"` as `\"` and every
newline as `\n`. Only after the `<tool_response>` confirms the write do you
reply — one short line. Never claim the README was written without a confirming
tool response; if none came back, emit the tool call again.

**If `README.md` already exists, read it first** and preserve anything still
accurate (badges, license, project description, hand-written notes) rather than
discarding the user's work. Update stale parts, fill gaps, keep the rest.

## Ground every section in the repo

Before writing, inspect the project and base each section on real evidence:

- **Name & one-line description** — from the package metadata
  (`pyproject.toml` / `package.json` / `Cargo.toml` / `go.mod`) or the top of
  the main module. Don't guess the name.
- **Install** — from the real dependency/build files: `pip install -e .` or
  `pip install -r requirements.txt` only if that file exists; `npm install`,
  `cargo build`, etc. Match the actual package manager and entry point.
- **Usage** — from the real CLI/API surface: the console-script name in
  `pyproject.toml`, `argparse`/`click` options, exported functions, the `main`.
  Show commands that actually exist, with real flag names.
- **Configuration** — environment variables, config files, and defaults that
  the code actually reads (grep for `os.environ` / `getenv` / config loaders).
- **Project layout** — the real top-level modules/dirs and a one-line purpose
  each, taken from the tree, not assumed.
- **Tests** — the real test command (`pytest`, `npm test`, …) if a test
  suite exists; omit the section if it doesn't.
- **License** — FIRST check whether a `LICENSE`/`LICENSE.md`/`LICENSE.txt` file
  exists (or a `license` field in the package metadata). If NONE exists, OMIT
  the License section entirely — do not write one. Never write "MIT" (or any
  license) as a guess or a default; an unlicensed project must have no License
  section. Only when a license is actually present, name exactly that one.

## Workflow

1. **Read the repo** — package metadata, entry points, config reads, the dir
   tree, and the existing `README.md` if present. Do this with the read/search
   tools before writing a word.
2. **Draft the README** from that evidence — concise, skimmable, with a short
   intro, fenced command blocks for install/usage, and only the sections the
   project warrants. Omit sections you have no evidence for rather than padding.
3. **Emit the `write_file` tool call** (format above) with the full document in
   `content`. Do not print the README outside the tool call.
4. **Wait for the `<tool_response>`**, then reply with one line: the path
   written and a one-sentence note of what you based it on or changed.

## Style

- Lead with what the project is and does, in one or two sentences.
- Prefer copy-pasteable commands in fenced blocks over prose descriptions.
- Keep it honest and minimal: no "Contributing"/"Roadmap"/"Acknowledgements"
  filler unless the user asks or the repo already has that content.
- Match the tone and formatting of the existing README when one is present.
