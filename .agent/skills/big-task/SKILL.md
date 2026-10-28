---
name: big-task
description: Split a large task into small subtasks tracked in a plan file, so each step fits in context and work can resume after /clear
triggers:
  - subtask
  - subtasks
  - breakdown
  - migrate
  - migration
  - resume
---

## Core idea

Your memory is the plan file, not the chat. History gets trimmed and these
instructions may not be shown again, so everything needed to continue must be
written to `.agent-cache/plans/<slug>.md`.

## Phase 1 — Plan (no edits yet)

1. If a plan file for this task already exists, read it and go to Phase 2.
2. Survey cheaply: `repo_map`, `search`, `outline`. Do NOT read whole files.
3. Write the plan file in this format:

   ```
   # Goal: <one sentence>
   Verify: <command that checks everything, e.g. ./gradlew test>
   ## Subtasks
   - [ ] 1. <goal> | files: a.kt, b.kt | check: <command>
   - [ ] 2. ...
   ## Notes
   <decisions, new names/signatures later steps need>
   ```
4. Sizing rules for each subtask:
   - touches at most 3 files and roughly 150 changed lines
   - leaves the build green on its own (no "fix it in step 5")
   - has its own check command
   - order by dependency: shared types/interfaces first, callers after
5. Mirror the subtasks with the `todo` tool, show the plan, and wait for the
   user to confirm if there are more than 5 subtasks.

## Phase 2 — Execute one subtask at a time

For the first unchecked subtask:
1. Re-read the plan file (the Notes may have changed what to do).
2. Read only what this subtask needs: `outline` first, then `read_file` with
   `start_line`/`end_line`. Never re-read files from finished subtasks.
3. Make the edit, run the subtask's check.
4. If the check fails, fix it (at most 2 attempts). Still failing: mark the line
   `- [!]` with the error summary in Notes, then stop and ask the user.
5. On success: mark it `- [x]` and add 1–2 lines to Notes ONLY for facts a later
   subtask needs (renamed symbol, new function signature, chosen approach).
6. Update `todo`, then report in one line: `Done 3/8: <goal>. Next: <goal>.`

After 2–3 subtasks, or when tool output has been large, stop and tell the user:
"Type /clear then 'resume <slug>' to continue with a fresh context."

## Context hygiene

- Trim command output: `... 2>&1 | tail -40`.
- Don't paste file contents back in replies; refer to path:line.
- Search with a narrow `path`, not the whole repo.
- Never keep a subtask's details in your head — if it matters later, it goes
  in Notes.

## Finish

When all subtasks are `[x]`: run the top-level Verify command, summarise the
whole change in a few lines, and leave the plan file in place (the user may
delete it).
