"""`agent graph <command>`: build and query the Kotlin/Android + Swift/iOS code graph.

  agent graph build [--force]            index (incremental) and print stats
  agent graph callers <symbol>           who calls it
  agent graph callees <symbol>           what it calls
  agent graph path <from> <to>           shortest dependency path (BFS)
  agent graph explain <symbol>           everything the graph knows about it
  agent graph impact <file>...           what may break if these files change
  agent graph query "<text>"             ranked symbol search

Symbols are full ids (com.x.UserRepository.load, App.Repo.load(id:)), suffixes
(UserRepository.load, Repo.load(id:)) or simple names (load). Output is plain text capped by --budget characters.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import queries
from .build import load_graph
from .model import EDGE_KINDS


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", "-d", default=".", help="project root (default: current dir)")
    common.add_argument("--budget", "-b", type=int,
                        default=int(os.getenv("AGENT_GRAPH_CHARS", str(queries.DEFAULT_BUDGET))),
                        help="max output characters (default 4000, AGENT_GRAPH_CHARS)")
    p = argparse.ArgumentParser(prog="agent graph", description="Kotlin/Android and Swift/iOS code graph.",
                                parents=[common])
    sub = p.add_subparsers(dest="cmd", metavar="<command>")
    b = sub.add_parser("build", parents=[common], help="index and print stats")
    b.add_argument("--force", action="store_true", help="ignore the cache and re-parse everything")
    for name, help_ in (("callers", "who calls a symbol"), ("callees", "what a symbol calls"),
                        ("explain", "summarize a symbol")):
        s = sub.add_parser(name, parents=[common], help=help_)
        s.add_argument("symbol")
    s = sub.add_parser("path", parents=[common], help="shortest path between two symbols")
    s.add_argument("source")
    s.add_argument("target")
    s.add_argument("--max-depth", type=int, default=6)
    s.add_argument("--kinds", default=",".join(queries.PATH_KINDS),
                   help="comma-separated edge kinds to follow")
    s = sub.add_parser("impact", parents=[common], help="what depends on these files")
    s.add_argument("files", nargs="+")
    s.add_argument("--depth", type=int, default=3)
    s = sub.add_parser("query", parents=[common], help="search symbols")
    s.add_argument("text", nargs="+")
    s.add_argument("--limit", type=int, default=15)
    return p


def run(argv: List[str], out=None) -> int:
    out = out or sys.stdout
    p = _parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help(out)
        return 1
    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=out)
        return 2
    graph, _info = load_graph(root, force=getattr(args, "force", False))
    budget = max(200, args.budget)
    if args.cmd == "build":
        text = queries.build_summary(graph)
    elif args.cmd == "callers":
        text = queries.callers(graph, args.symbol, budget)
    elif args.cmd == "callees":
        text = queries.callees(graph, args.symbol, budget)
    elif args.cmd == "explain":
        text = queries.explain(graph, args.symbol, budget)
    elif args.cmd == "path":
        kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
        bad = [k for k in kinds if k not in EDGE_KINDS]
        if bad:
            print(f"Unknown edge kind(s): {', '.join(bad)}. Known: {', '.join(EDGE_KINDS)}", file=out)
            return 2
        text = queries.path(graph, args.source, args.target, args.max_depth, kinds, budget)
    elif args.cmd == "impact":
        files = []
        for f in args.files:
            fp = Path(f)
            if fp.is_absolute():
                try:
                    f = fp.resolve().relative_to(root).as_posix()
                except ValueError:
                    pass
            files.append(f)
        text = queries.impact(graph, files, args.depth, budget)
    else:
        text = queries.query(graph, " ".join(args.text), budget, args.limit)
    print(text, file=out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
