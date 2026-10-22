---
name: click-dummy
description: Generate a clickable HTML prototype (click dummy) of a feature — one self-contained low-fi wireframe file with linked screens and fake data
triggers:
  - click dummy
  - clickdummy
  - clickable prototype
  - clickable mockup
  - html prototype
  - html mockup
  - wireframe
  - wireframes
  - mockup
  - prototype
  - screen flow
  - user flow
  - proof of concept ui
  - poc ui
  - throwaway ui
---

## What a click dummy is

A throwaway, clickable HTML prototype that lets a stakeholder *feel* a feature's
flow before any real code is written. It is NOT production code: no backend, no
build step, no real data, no persistence. Just screens you can click through.

## Output contract

Produce **one single self-contained `.html` file**. The ONLY way to create it
is to emit a `write_file` tool call. The whole HTML document goes inside the
tool call's `content` field — NOT in a ```html code block, NOT in prose. A code
block in your reply does nothing: the file is written only when the agent runs
the tool call below. So your reply contains exactly one such tool call and no
HTML anywhere else.

Emit it in this exact format (this is the first and main thing you do — do it
before writing any report, and never say the file was created unless you have
actually emitted this call):

    <tool_call>
    {"name": "write_file", "arguments": {"path": "click-dummy/<feature-slug>.html", "content": "<!DOCTYPE html>\n<html>...the full document, with every newline and quote JSON-escaped...\n</html>"}}
    </tool_call>

The `content` string must be valid JSON: escape every `"` as `\"` and every
newline as `\n`. After the tool call runs you will get a `<tool_response>`
confirming the write — only then write a one-line report. The file must open
directly in a browser with no server and no build.

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
2. **Emit the `write_file` tool call** (format above) as your main action. Build
   the full HTML inside its `content`: all screens as sibling `<section>`s, the
   top bar, the `show(id)` switcher, every button wired to a real `show(...)` so
   nothing dead-ends. Do not print the HTML anywhere outside the tool call.
3. **Wait for the `<tool_response>`.** Only after it confirms the write do you
   reply — one short line: the file path, the screen list, "open it in a browser."
   If no tool response came back, the file was NOT written: emit the tool call
   again. Never report success without a confirming tool response.

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
