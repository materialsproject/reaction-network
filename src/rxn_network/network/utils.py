"""
Utility functions used in the ReactionNetwork class.
"""
from typing import List

from graph_tool import Graph, Vertex

from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.reactions.computed import ComputedReaction


def get_rxn_nodes_and_edges(rxns: List[ComputedReaction]):
    """
    Given a list of reactions, return a list of nodes and edges for constructing the
    reaction network.

    Args:
        rxns: a list of enumerated ComputedReaction objects to build a network from.

    Returns:
        A tuple consisting of (nodes, edges) where nodes is a list of NetworkEntry
        objects and edges is a list of tuples of the form (source_idx, target_idx).
    """
    nodes, edges = [], []

    for rxn in rxns:
        reactant_node = NetworkEntry(rxn.reactant_entries, NetworkEntryType.Reactants)
        product_node = NetworkEntry(rxn.product_entries, NetworkEntryType.Products)

        if reactant_node not in nodes:
            nodes.append(reactant_node)
            reactant_idx = len(nodes) - 1
        else:
            reactant_idx = nodes.index(reactant_node)

        if product_node not in nodes:
            nodes.append(product_node)
            product_idx = len(nodes) - 1
        else:
            product_idx = nodes.index(product_node)

        edges.append((reactant_idx, product_idx))

    return nodes, edges


def get_loopback_edges(g: Graph, nodes: List[Vertex]):
    """
    Given a graph and a list of nodes to check, this function finds and returns loopback edges
    (i.e., edges that connect a product node to its equivalent reactant node)

    Args:
        g: graph-tool Graph object
        nodes: List of vertices from which to find loopback edges

    Returns:
        A list of tuples of the form (source_idx, target_idx, cost=0, rxn=None, type="loopback")
    """
    edges = []
    for idx1, p in enumerate(nodes):
        if p.description.value != NetworkEntryType.Products.value:
            continue
        for idx2, r in enumerate(nodes):
            if r.description.value != NetworkEntryType.Reactants.value:
                continue
            if p.entries == r.entries:
                edges.append((g.vertex(idx1), g.vertex(idx2), 0.0, None, "loopback"))

    return edges
