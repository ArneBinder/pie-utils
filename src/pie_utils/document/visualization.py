from __future__ import annotations

import logging
from collections import defaultdict
from typing import TypeVar

from asciidag.graph import Graph
from asciidag.node import Node
from pytorch_ie.core import Document

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def _construct_nodes(name: str, ancestor_graph: dict[str, list[str]], store: dict[str, Node]):
    if name in store:
        return store[name]
    deps = ancestor_graph.get(name, [])
    dep_nodes = [_construct_nodes(dep_name, ancestor_graph, store=store) for dep_name in deps]
    store[name] = Node(name, parents=dep_nodes)
    return store[name]


def _revert_edges(edges: dict[str, list[str]]) -> dict[str, list[str]]:
    reverted_edges = defaultdict(list)
    for source, targets in edges.items():
        for target in targets:
            reverted_edges[target].append(source)
    return dict(reverted_edges)


def print_document_annotation_graph(
    annotation_graph: dict[str, list[str]],
    add_root_node: str | None = None,
    remove_node: str | None = None,
    swap_edges: bool = True,
):
    """example call:

    print_document_annotation_graph(
        annotation_graph=document._annotation_graph, remove_node="_artificial_root",
    )

    Args:
        annotation_graph: the annotation graph
        add_root_node: if available, add an artificial root node with this name that connects to all original roots
        remove_node: if available, remove the node with this name from the annotation_graph
        swap_edges: iff True, swap the edges of the graph
    """
    dependency_graph = dict(annotation_graph)
    if remove_node is not None:
        del dependency_graph[remove_node]

    reverted_dependency_graph = _revert_edges(edges=dependency_graph)
    sources = set(dependency_graph) - set(reverted_dependency_graph)
    sinks = set(reverted_dependency_graph) - set(dependency_graph)

    if swap_edges:
        ancestor_graph = dict(reverted_dependency_graph)
        roots = sinks
    else:
        ancestor_graph = dict(dependency_graph)
        roots = sources

    if add_root_node is not None:
        ancestor_graph[add_root_node] = list(roots)
        roots = {add_root_node}

    graph = Graph()
    node_store: dict[str, Node] = dict()
    root_nodes = [
        _construct_nodes(root, ancestor_graph=ancestor_graph, store=node_store) for root in roots
    ]
    graph.show_nodes(tips=root_nodes)
