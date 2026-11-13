"""Code graph for Kotlin Android projects.

Pipeline: sources.list_sources -> per-file facts (kotlin_parser, Gradle,
resource XML, manifest) cached in a FactStore -> resolver builds a CodeGraph
in memory -> queries / rank / cli.

    from agent_cli.graph import load_graph
    graph, info = load_graph(Path("."))
"""
from .build import load_graph
from .model import CodeGraph, Edge, Node

__all__ = ["load_graph", "CodeGraph", "Edge", "Node"]
