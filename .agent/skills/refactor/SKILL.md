---
name: refactor
description: Safe code refactoring workflow — verify tests, make incremental changes, preserve public API
triggers:
  - refactor
  - refactoring
  - extract method
  - extract function
  - extract class
  - inline
  - rename
  - move function
  - restructure
  - simplify code
  - clean up code
  - reduce duplication
---

## Rules

1. **Tests gate everything** — run the full suite before starting (abort if
   failing) and after every single transformation (revert if failing).
2. **One transformation per step** — extract, inline, rename, move,
   de-duplicate, or simplify conditionals. Never bundle unrelated changes.
3. **Behavior-preserving only** — no features, no bug fixes, no logic changes.
   Structure changes, not behavior.
4. **Preserve public API** — do not change exported signatures unless the user
   explicitly asks. Check callers (grep) before changing anything.
5. **Don't fix tests** — if tests fail after a refactoring, the refactoring is
   wrong. Exception: updating imports for moved/renamed symbols.
6. **No speculative abstraction** — no new interfaces, patterns, or
   dependencies unless existing duplication justifies it.
7. **Delete dead code** — don't comment it out or preserve it.
8. **No unrelated formatting** — don't touch files you aren't refactoring.

## Workflow

1. Run tests → identify public API and callers → plan
2. For 3+ transformations, summarize the plan and confirm with the user
3. Loop: apply one change → update imports/references → run tests → next
4. Final diff review — verify no behavior changed, no dead code remains
5. Report: what changed and why
