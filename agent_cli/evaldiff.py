"""Measure edit size per AGENT_STYLE on an eval set (needs a model server).

    python -m agent_cli.evaldiff                       # off vs full, coder, evals/diffsize/tasks.json
    python -m agent_cli.evaldiff --styles off,lite,full --micro --repeat 3

Each task copies its fixture to a temp dir, adds the task's files, sends the
request once per style and counts changed lines (+/-) of the resulting diff.
`off` is the baseline without the style rules (the "before").
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "evals" / "diffsize" / "tasks.json"


def diff_size(diffs: List[str]) -> int:
    """Changed lines (+ and -, headers excluded) in unified diffs."""
    return sum(1 for d in diffs for ln in d.splitlines()
               if ln[:1] in "+-" and not ln.startswith(("+++", "---")))


def run_task(task: dict, style: str, micro: bool = False, cfg_overrides: Optional[dict] = None,
             base: Path = REPO) -> int:
    """Changed lines for one task under one style; -1 when nothing changed."""
    from .coder import Coder
    from .config import Config
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        shutil.copytree(base / task["fixture"], root)
        cfg = Config(workdir=root, auto_approve=True, style=style, **(cfg_overrides or {}))
        # Never block on y/n prompts (mentioned files, test-gen offers).
        with mock.patch("agent_cli.coder.Confirm.ask", return_value=False):
            coder = Coder(cfg, mode="code")
            coder.add(task["files"])
            if micro:
                coder.micro(task["request"])
            else:
                coder.send(task["request"])
        return diff_size(coder.last_diffs) if coder.last_diffs else -1


def run_eval(tasks: List[dict], styles: List[str], micro: bool = False, repeat: int = 1,
             cfg_overrides: Optional[dict] = None, base: Path = REPO) -> Dict[str, Dict[str, List[int]]]:
    """{task id: {style: [changed lines per run]}}."""
    out: Dict[str, Dict[str, List[int]]] = {}
    for t in tasks:
        out[t["id"]] = {s: [run_task(t, s, micro, cfg_overrides, base) for _ in range(repeat)]
                        for s in styles}
    return out


def _mean(xs: List[int]) -> Optional[float]:
    ok = [x for x in xs if x >= 0]
    return sum(ok) / len(ok) if ok else None


def report(results: Dict[str, Dict[str, List[int]]], styles: List[str]) -> str:
    w = max([len(k) for k in results] + [4])
    lines = [f"{'task':<{w}}  " + "  ".join(f"{s:>7}" for s in styles)]
    totals = {s: 0.0 for s in styles}
    complete = []
    for tid, per in results.items():
        means = {s: _mean(per[s]) for s in styles}
        lines.append(f"{tid:<{w}}  " + "  ".join(
            f"{means[s]:>7.1f}" if means[s] is not None else f"{'-':>7}" for s in styles))
        if all(m is not None for m in means.values()):  # only tasks every style edited
            complete.append(tid)
            for s in styles:
                totals[s] += means[s]
    lines.append(f"{'total':<{w}}  " + "  ".join(f"{totals[s]:>7.1f}" for s in styles)
                 + f"   ({len(complete)}/{len(results)} tasks edited under every style)")
    base = styles[0]
    for s in styles[1:]:
        if totals[base]:
            lines.append(f"{s} vs {base}: {100.0 * (totals[s] - totals[base]) / totals[base]:+.1f}% "
                         "changed lines")
    lines.append("('-' = no edit made; changed lines = added + removed)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent_cli.evaldiff")
    p.add_argument("--tasks", default=str(DEFAULT_TASKS))
    p.add_argument("--styles", default="off,full", help="comma-separated; first is the baseline")
    p.add_argument("--micro", action="store_true", help="run through /micro instead of coder")
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args(argv)
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    results = run_eval(tasks, styles, args.micro, max(1, args.repeat))
    print(report(results, styles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
