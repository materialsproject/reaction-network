"""
Implementation of reaction network and graph classes.
"""
from __future__ import annotations

from dataclasses import field
from queue import Empty, PriorityQueue
from typing import TYPE_CHECKING, Iterable

import rustworkx as rx
from pymatgen.entries import Entry
from tqdm import tqdm

from rxn_network.costs.functions import Softplus
from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.network.base import Graph, Network
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.pathways.basic import BasicPathway
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from rxn_network.costs.base import CostFunction
    from rxn_network.reactions.base import Reaction

logger = get_logger(__name__)


class ReactionNetwork(Network):
    """
    Main reaction network class for building graph networks and performing
    pathfinding. Graphs are built using the rustworkx package (a NetworkX equivalent
    implemented in Rust).

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.

    """

    def __init__(
        self,
        rxns: ReactionSet,
        cost_function: CostFunction = field(default_factory=Softplus),
    ):
        """
        Initialize a ReactionNetwork object for a reaction set and cost function.

        To build the graph network, call the build() method in-place.

        Args:
            rxns: Set of reactions used to construct the network.
            cost_function: The function used to calculate the cost of each reaction
                edge. Defaults to a Softplus function with default settings (i.e.
                energy_per_atom only).
        """
        super().__init__(rxns=rxns, cost_function=cost_function)

    def build(self) -> None:
        """
        In-place method. Construct the reaction network graph object and store under the
        "graph" attribute.

        WARNING: This does NOT initialize the precursors or target attributes; you must
        call set_precursors() or set_target() to do so.

        Returns:
            None
        """
        logger.info("Building graph from reactions...")

        g = Graph()

        nodes, edges = get_rxn_nodes_and_edges(self.rxns)
        edges.extend(get_loopback_edges(nodes))  # type: ignore

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        logger.info(f"Built graph with {g.num_nodes()} nodes and {g.num_edges()} edges")

        self._g = g  # type: ignore

    def find_pathways(
        self, targets: list[Entry | str], k: int = 15
    ) -> list[BasicPathway]:
        """
        Find the k-shortest paths to a provided list of one or more targets.

        Args:
            targets: List of the formulas or entry objects of each target.
            k: Number of k-shortest paths to find for each target. Defaults to 15.

        Returns:
            List of BasicPathway objects to all provided targets.
        """
        if not self.precursors:
            raise AttributeError("Must call set_precursors() before pathfinding!")

        paths = []
        for target in targets:
            self.set_target(target)
            print(
                f"Paths to {self.target.composition.reduced_formula} \n"  # type: ignore
            )
            print("--------------------------------------- \n")
            pathways = self._k_shortest_paths(k=k)
            paths.extend(pathways)

        paths = PathwaySet.from_paths(paths)

        return paths

    def set_precursors(self, precursors: Iterable[Entry | str]):
        """
        In-place method. Sets the precursors of the network. Removes all references to
        previous precursors.

        If entries are provided, will use the entries to set the precursors. If strings
        are provided, will automatically find the lowest-energy entries with matching
        reduced_formula.

        Args:
            precursors: iterable of entries/formulas of precursor phases.

        Returns:
            None
        """
        g = self._g
        if not g:
            raise ValueError("Must call build() before setting precursors!")

        precursors = {
            p
            if isinstance(p, (Entry, ExperimentalReferenceEntry))
            else self.entries.get_min_entry_by_formula(p)
            for p in precursors
        }

        if precursors == self.precursors:
            return
        if not all(p in self.entries for p in precursors):
            raise ValueError("One or more precursors are not included in network!")

        precursors_entry = NetworkEntry(precursors, NetworkEntryType.Precursors)
        if self.precursors:  # remove old precursors
            for node in g.node_indices():
                if (
                    g.get_node_data(node).description.value
                    == NetworkEntryType.Precursors.value
                ):
                    g.remove_node(node)
                    break
            else:
                raise ValueError("Old precursors node not found in graph!")

        precursors_node = g.add_node(precursors_entry)

        edges_to_add = []
        for node in g.node_indices():
            entry = g.get_node_data(node)
            entry_type = entry.description.value

            if entry_type == NetworkEntryType.Reactants.value:
                if entry.entries.issubset(precursors):
                    edges_to_add.append((precursors_node, node, "precursor_edge"))
            elif entry.description.value == NetworkEntryType.Products.value:
                for node2 in g.node_indices():
                    entry2 = g.get_node_data(node2)
                    if entry2.description.value == NetworkEntryType.Reactants.value:
                        if precursors.issuperset(entry2.entries):
                            continue
                        if precursors.union(entry.entries).issuperset(entry2.entries):
                            edges_to_add.append((node, node2, "loopback_edge"))

        g.add_edges_from(edges_to_add)
        self._precursors = precursors

    def set_target(self, target: Entry | str) -> None:
        """
        In-place method. Can only provide one target entry or formula at a time.

        If entry is provided, will use that entry to set the target. If string is
        provided, will automatically find minimum-energy entry with
        matching reduced_formula.

        Args:
            target: Entry, or string of reduced formula, of target phase.

        Returns:
            None
        """
        g = self._g
        if not g:
            raise ValueError("Must call build() before setting target!")

        target = (
            target
            if isinstance(target, (Entry, ExperimentalReferenceEntry))
            else self.entries.get_min_entry_by_formula(target)
        )

        if target == self.target:
            return

        if target not in self.entries:
            raise ValueError("Target is not included in network!")

        if self.target:
            for node in g.node_indices():
                if (
                    g.get_node_data(node).description.value
                    == NetworkEntryType.Target.value
                ):
                    g.remove_node(node)
                    break
            else:
                raise ValueError("Old target node not found in graph!")

        target_entry = NetworkEntry([target], NetworkEntryType.Target)
        target_node = g.add_node(target_entry)

        edges_to_add = []
        for node in g.node_indices():
            entry = g.get_node_data(node)
            entry_type = entry.description.value

            if entry_type != NetworkEntryType.Products.value:
                continue
            if target in entry.entries:
                edges_to_add.append((node, target_node, "target_edge"))

        g.add_edges_from(edges_to_add)

        self._target = target

    def _k_shortest_paths(self, k: int):
        """Wrapper for finding the k shortest paths using Yen's algorithm. Returns
        BasicPathway objects"""
        g = self._g
        if not g:
            raise ValueError("Must call build() before pathfinding!")
        paths = []

        precursors_node = g.find_node_by_weight(
            NetworkEntry(self.precursors, NetworkEntryType.Precursors)  # type: ignore
        )
        target_node = g.find_node_by_weight(
            NetworkEntry([self.target], NetworkEntryType.Target)
        )
        for path in yens_ksp(g, self.cost_function, k, precursors_node, target_node):
            paths.append(self._path_from_graph(g, path, self.cost_function))

        for path in paths:
            print(path, "\n")

        return paths

    @staticmethod
    def _path_from_graph(g, path, cf: CostFunction):
        """Gets a BasicPathway object from a shortest path found in the network"""
        rxns = []
        costs = []

        for step, node in enumerate(path):
            if (
                g.get_node_data(node).description.value
                == NetworkEntryType.Products.value
            ):
                e = g.get_edge_data(path[step - 1], node)

                rxns.append(e)
                costs.append(get_edge_weight(e, cf))

        return BasicPathway(reactions=rxns, costs=costs)


def get_rxn_nodes_and_edges(
    rxns: ReactionSet,
) -> tuple[list[NetworkEntry], list[tuple[int, int, Reaction]]]:
    """
    Given a reaction set, return a list of nodes and edges for constructing the
    reaction network.

    Args:
        rxns: a list of enumerated ComputedReaction objects to build a network from.

    Returns:
        A tuple consisting of (nodes, edges) where nodes is a list of NetworkEntry
        objects and edges is a list of tuples of the form (source_idx, target_idx,
        reaction).
    """
    nodes, edges = [], []

    for rxn in tqdm(rxns):
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

        edges.append((reactant_idx, product_idx, rxn))

    return nodes, edges


def get_loopback_edges(
    nodes: list[NetworkEntry],
) -> list[tuple[int, int, str]]:
    """
    Given a list of nodes to check, this function finds and returns loopback
    edges (i.e., edges that connect a product node to its equivalent reactant node)

    Args:
        nodes: List of vertices from which to find loopback edges

    Returns:
        A list of tuples of the form (source_idx, target_idx, reaction)
    """
    edges = []
    for idx1, p in enumerate(nodes):
        if p.description.value != NetworkEntryType.Products.value:
            continue
        for idx2, r in enumerate(nodes):
            if r.description.value != NetworkEntryType.Reactants.value:
                continue
            if p.entries == r.entries:
                edges.append((idx1, idx2, "loopback_edge"))

    return edges


def get_edge_weight(edge_obj: object, cf: CostFunction):
    """
    Given an edge of a reaction network, calculates the cost/weight of that edge.
    Corresponds to zero for loopback & precursor/target edges. Evaluates cost function
    for all reaction edges.

    Args:
        edge_obj: An edge in the reaction network
    """
    if isinstance(edge_obj, str) and edge_obj in [
        "loopback_edge",
        "precursor_edge",
        "target_edge",
    ]:
        return 0.0
    if isinstance(edge_obj, (ComputedReaction, OpenComputedReaction)):
        return cf.evaluate(edge_obj)

    raise ValueError("Unknown edge type")


def yens_ksp(
    g: rx.PyGraph,
    cf: CostFunction,
    num_k: int,
    precursors_node: int,
    target_node: int,
) -> list[list[int]]:
    """
    Yen's Algorithm for k-shortest paths, adopted for rustworkx.

    This implementation was inspired by the igraph implementation by Antonin Lenfant.

    Reference (original Yen's KSP paper):
        Jin Y. Yen, "Finding the K Shortest Loopless Paths n a Network", Management
        Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.

    Args:
        g: the rustworkx PyGraph object.
        cf: A cost function for evaluating the edge weights.
        num_k: number of k shortest paths that should be found.
        precursors_node: the index of the node representing the precursors.
        target_node: the index of the node representing the targets.

    Returns:
        List of lists of graph vertices corresponding to each shortest path
            (sorted in increasing order by cost).
    """

    def path_cost(nodes):
        """Calculates path cost given a list of nodes"""
        cost = 0
        for j in range(len(nodes) - 1):
            cost += get_edge_weight(g.get_edge_data(nodes[j], nodes[j + 1]), cf)
        return cost

    def get_edge_weight_with_cf(edge_obj):
        """Includes user-specified cost function in function call"""
        return get_edge_weight(edge_obj, cf)

    g = g.copy()

    path = rx.dijkstra_shortest_paths(  # type: ignore
        g, precursors_node, target_node, weight_fn=get_edge_weight_with_cf
    )

    if not path:
        return []

    path = list(path[target_node])

    a = [path]
    a_costs = [path_cost(path)]

    b = PriorityQueue()  # type: ignore

    for k in range(1, num_k):
        try:
            prev_path = a[k - 1]
        except IndexError:
            logger.info(f"Identified only k={k-1} paths before exiting. \n")
            break

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i]

            removed_edges = []

            for path in a:
                if len(path) - 1 > i and root_path == path[:i]:
                    try:
                        e = g.get_edge_data(path[i], path[i + 1])
                    except rx.NoEdgeBetweenNodes:
                        continue

                    g.remove_edge(path[i], path[i + 1])
                    removed_edges.append((path[i], path[i + 1], e))

            spur_path = rx.dijkstra_shortest_paths(  # type: ignore
                g, spur_node, target_node, weight_fn=get_edge_weight_with_cf
            )

            g.add_edges_from(removed_edges)

            if spur_path:
                total_path = list(root_path) + list(spur_path[target_node])
                total_path_cost = path_cost(total_path)
                b.put((total_path_cost, total_path))

        while True:
            try:
                cost_, path_ = b.get(block=False)
            except Empty:
                break
            if path_ not in a:
                a.append(path_)
                a_costs.append(cost_)
                break

    return a
