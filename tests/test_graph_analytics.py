"""Spot-check PageRank stability on a known toy graph (no Spark required)."""

from __future__ import annotations

import networkx as nx


def test_pagerank_top_node_on_known_graph():
    """Hub-spoke graph: node 0 is the hub; PageRank should rank it highest."""
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    # 4 spokes feeding the hub, hub feeds none (sink).
    for i in range(1, 5):
        G.add_edge(i, 0, weight=1.0)

    pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    top = max(pr.items(), key=lambda kv: kv[1])[0]
    assert top == 0, f"Expected hub (0) to dominate, got {top}: {pr}"


def test_shortest_paths_on_chain():
    """A → B → C → D: shortest-path lengths from D to (A, B, C) should be 3, 2, 1."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    # Cycle, distances from D forward.
    assert nx.shortest_path_length(G, source="D", target="A") == 1
    assert nx.shortest_path_length(G, source="D", target="B") == 2
    assert nx.shortest_path_length(G, source="D", target="C") == 3
