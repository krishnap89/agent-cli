---
name: click-dummy
description: Generate a clickable low-fi HTML prototype (click dummy) of a feature — one self-contained wireframe file with linked screens and fake data
triggers:
  - click dummy
  - clickdummy
  - clickable prototype
  - clickable mockup
  - clickable wireframe
  - html prototype
  - html mockup
  - html wireframe
  - ui wireframe
  - ui mockup
  - screen flow
  - screen mockup
  - throwaway ui
  - proof of concept ui
  - poc ui
---

## What a click dummy is

A throwaway, clickable HTML prototype that lets a stakeholder *feel* a feature's
flow before any real code is written. It is NOT production code: no backend, no
build step, no real data, no persistence. Just screens you can click through.

## Output contract

Produce **one single self-contained `.html` file** at
`click-dummy/<feature-slug>.html`, created with `write_file` in the format your
system prompt gives. HTML printed in your reply or in a code block is not saved.
Write the file first, report in one line only after the result confirms it. The
file must open directly in a browser with no server and no build.

- **Styling:** Tailwind via CDN — `<script src="https://cdn.tailwindcss.com"></script>`.
  No other dependencies, no local assets, no external images.
- **Fidelity:** LOW-FI WIREFRAME. Grey boxes, placeholder text, visible borders,
  a monospace or plain font. It must look deliberately unfinished so no one
  mistakes it for final visual design. Use `bg-gray-*`, `border-dashed`,
  "Lorem"-style filler, and `[ image ]` / `▭` placeholders for media. Do not
  pick a brand palette or polish the visuals.
- **Navigation:** all screens live in the one file as sibling `<section>`
  blocks; only one is visible at a time. Clicking a nav element or button calls
  a tiny vanilla-JS `show(id)` that toggles which section is visible. No page
  reloads, no router, no frameworks.
- **Fake data:** hard-code plausible sample rows/items inline. Never fetch.
- Add a thin persistent top bar labelling the prototype, e.g.
  `CLICK DUMMY · <feature> · not functional` and a small screen-name indicator.

## Workflow

**Never stop to ask the user questions — always produce the file in one turn.**
This skill runs one-shot, so a clarifying question ends the turn and nothing
gets built. Infer the feature name, the screens, and the click path from the
request; when a detail is missing, pick a sensible default and proceed. Only
state your assumptions in the final report, after the file is written.

1. **Infer the flow silently.** From the request, decide the feature's name, the
   screens, and the click path. Fill any gap with an obvious default — do not ask
   and do not narrate a plan first.
2. **Write the file** with `write_file`: all screens as sibling `<section>`s, the
   top bar, the `show(id)` switcher, every button wired to a real `show(...)` so
   nothing dead-ends.
3. **Wait for the result.** Once it confirms the write, reply with one short line:
   the file path, the screen list, "open it in a browser." Without that
   confirmation the file was NOT written: write it again.

## The switcher (use this pattern)

```html
<nav><!-- buttons call show('screen-id') --></nav>
<section id="screen-list" class="screen">...</section>
<section id="screen-detail" class="screen hidden">...</section>
<script>
  function show(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.add('hidden'));
    document.getElementById(id).classList.remove('hidden');
    document.getElementById('screen-name').textContent = id.replace('screen-', '');
    window.scrollTo(0, 0);
  }
</script>
```

## Rules

- One file, no dependencies beyond the Tailwind CDN tag. Everything inline.
- Keep it low-fi on purpose — do not drift into polished UI.
- Every interactive element must either navigate (`show`) or be an explicit
  labelled stub. No silent dead ends.
- Don't build backend logic, forms that submit, auth, or state that persists.
  Fake it visually.
- Prefer fewer, clearer screens over an exhaustive one. Cover the happy path
  plus the one or two branches the user cares about.
