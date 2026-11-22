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

Create `README.md` (repo root, unless the user names another path) with
`write_file`, in the format your system prompt gives. Text printed in your reply
or in a code block is not saved. Report in one line only after the result
confirms the write.

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
   tree, and the existing `README.md` if present. Read and search before
   writing a word.
2. **Draft the README** from that evidence — concise, skimmable, with a short
   intro, fenced command blocks for install/usage, and only the sections the
   project warrants. Omit sections you have no evidence for rather than padding.
3. **Write the file** with `write_file`, passing the full document.
4. **Wait for the result**, then reply with one line: the path written and a
   one-sentence note of what you based it on or changed.

## Style

- Lead with what the project is and does, in one or two sentences.
- Prefer copy-pasteable commands in fenced blocks over prose descriptions.
- Keep it honest and minimal: no "Contributing"/"Roadmap"/"Acknowledgements"
  filler unless the user asks or the repo already has that content.
- Match the tone and formatting of the existing README when one is present.
