"""Code graph for Kotlin/Android, Swift/iOS and TypeScript/JavaScript projects.

Pipeline: sources.list_sources -> per-file facts (kotlin_parser, swift_parser,
ts_parser, Gradle, Android resource XML and manifests, Package.swift,
project.pbxproj, storyboards/xibs, .strings/.xcstrings, asset catalogs,
Info.plist, package.json, tsconfig.json) cached in a
FactStore -> resolver builds a CodeGraph in memory -> queries / rank / cli.

    from agent_cli.graph import load_graph
    graph, info = load_graph(Path("."))
"""
from .build import load_graph
from .model import CodeGraph, Edge, Node

__all__ = ["load_graph", "CodeGraph", "Edge", "Node"]
