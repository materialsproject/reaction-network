"""
Implementation of reaction network interface.
"""
from dataclasses import field
from typing import Iterable, List, Optional, Union

import rustworkx as rx
from pymatgen.entries import Entry
from tqdm import tqdm

from rxn_network.core.cost_function import CostFunction
from rxn_network.core.network import Network
from rxn_network.costs.softplus import Softplus
from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.network.gt import (
    initialize_graph,
    load_graph,
    save_graph,
    update_vertex_props,
    yens_ksp,
)
from rxn_network.pathways.basic import BasicPathway
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.reactions.reaction_set import ReactionSet


class ReactionNetwork(Network):
    """
    Main reaction network class for building graphs of reactions and performing
    pathfinding.
    """

    def __init__(
        self,
        rxns: ReactionSet,
        cost_function: CostFunction = field(default_factory=Softplus),
    ):
        """
        Initialize a ReactionNetwork object for a set of reactions.

        Note: the precursors and target must be set by calling set_precursors() and
        set_target() respectively.

        Args:
            rxns: Reaction set of reactions
            enumerators: iterable of enumerators which will be called during the
                build of the network
            cost_function: the function used to calculate the cost of each reaction edge
            open_elem: Optional name of an element that is kept open during reaction
            chempot: Optional associated chemical potential of open element
        """
        super().__init__(rxns=rxns, cost_function=cost_function)

    def build(self):
        """
        Construct the reaction network graph object and store under the "graph"
        attribute. Does NOT initialize precursors or target; you must call
        set_precursors() or set_target() to do so.

        Returns:
            None
        """
        self.logger.info("Building graph from reactions...")

        g = rx.PyDiGraph()

        nodes, edges = get_rxn_nodes_and_edges(self.rxns)
        edges.extend(get_loopback_edges(g, nodes))

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        self._g = g

    def find_pathways(self, targets: List[str], k: float = 15) -> List[BasicPathway]:
        """
        Find the k-shortest paths to a provided list of 1 or more targets.

        Args:
            targets: List of the formulas of each target
            k: Number of shortest paths to find for each target

        Returns:
            List of BasicPathway objects to all provided targets.
        """
        if not self.precursors:
            raise AttributeError("Must call set_precursors() before pathfinding!")

        paths = []
        for target in targets:
            self.set_target(target)
            print(f"PATHS to {self.target.composition.reduced_formula} \n")
            print("--------------------------------------- \n")
            pathways = self._shortest_paths(k=k)
            paths.extend(pathways)

        paths = PathwaySet.from_paths(paths)

        return paths

    def set_precursors(self, precursors: Iterable[Union[Entry, str]]):
        """
        Sets the precursors of the network. Removes all references to previous
        precursors.

        If entries are provided, will use the entries to set the precursors. If strings
        are provided, will automatically find minimum-energy entries with matching
        reduced_formula.

        Args:
            precursors: iterable of

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
            entry_type = node.description.value

            if entry_type == NetworkEntryType.Reactants.value:
                if node.entries.issubset(precursors):
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

    def set_target(self, target: Union[Entry, str]):
        """
        If entry is provided, will use that entry to set the target. If string is
        provided, will automatically find minimum-energy entry with matching
        reduced_formula.

        Args:
            target: Entry, or string of reduced formula, of target

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
        for v in g.node_indices():
            entry = g.get_node_data(v)

            if entry.description.value != NetworkEntryType.Products.value:
                continue
            if target in entry.entries:
                edges_to_add.append((v, target_v, "target_edge"))

        g.add_edges_from(edges_to_add)

        self._target = target

    def load_graph(self, filename: str):
        """
        Loads graph-tool graph from file.

        Args:
            filename: Filename of graph object to load (for example, .gt or .gt.gz
                format)

        Returns: None
        """
        self._g = load_graph(filename)

    def write_graph(self, filename: Optional[str] = None):
        """
        Writes graph to file. If filename is not provided, will write to CHEMSYS.gt.gz
        (where CHEMSYS is the chemical system of the network)

        Args:
            filename: Filename to write to. If None, writes to default filename.

        Returns: None
        """
        if not filename:
            filename = f"{self.chemsys}.gt.gz"

        save_graph(self._g, filename)

    def _shortest_paths(self, k):
        """Finds the k shortest paths using Yen's algorithm and returns BasicPathways"""
        g = self._g
        paths = []

        precursors_v = find_vertex(g, g.vp["type"], NetworkEntryType.Precursors.value)[
            0
        ]
        target_v = find_vertex(g, g.vp["type"], NetworkEntryType.Target.value)[0]

        for path in yens_ksp(g, k, precursors_v, target_v):
            paths.append(self._path_from_graph(g, path))

        for path in paths:
            print(path, "\n")

        return paths

    @staticmethod
    def _path_from_graph(g, path):
        """Gets a BasicPathway object from a shortest path found in the network"""
        rxns = []
        costs = []

        for step, v in enumerate(path):
            if g.vp["type"][v] == NetworkEntryType.Products.value:
                e = g.edge(path[step - 1], v)

                rxns.append(g.ep["rxn"][e])
                costs.append(g.ep["cost"][e])

        return BasicPathway(reactions=rxns, costs=costs)

    @classmethod
    def from_dict_and_file(cls, d: dict, filename: str):
        """
        Convenience constructor method that loads a ReactionNetwork object from a
        dictionary (MSONable version) and a filename (to load graph object in
        graph-tool).

        Args:
            d: Dictionary containing the ReactionNetwork object
            filename: Filename of graph object to load (for example, .gt or .gt.gz
                format)

        Returns:
            ReactionNetwork object with loaded graph
        """
        rn = cls.from_dict(d)
        rn.load_graph(filename)  # pylint: disable=no-member

        return rn

    @property
    def graph(self):
        """Returns the network object in graph-tool"""
        return self._g

    @property
    def chemsys(self) -> str:
        """Returns a string of the chemical system of the network"""
        return "-".join(sorted(self.entries.chemsys))

    def as_dict(self) -> dict:
        """Return MSONable dict"""
        d = super().as_dict()
        d["precursors"] = list(self.precursors) if self.precursors else None
        d["target"] = self.target
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate object from MSONable dict"""
        precursors = d.pop("precursors", None)
        target = d.pop("target", None)

        rn = super().from_dict(d)
        rn._precursors = precursors  # pylint: disable=protected-access
        rn._target = target  # pylint: disable=protected-access

        return rn

    def __repr__(self):
        return (
            "ReactionNetwork for chemical system: "
            f"{self.chemsys}, "
            f"with Graph: {str(self._g)}"
        )


def get_rxn_nodes_and_edges(rxns: ReactionSet):
    """
    Given a reaction set, return a list of nodes and edges for constructing the
    reaction network.

    Args:
        rxns: a list of enumerated ComputedReaction objects to build a network from.

    Returns:
        A tuple consisting of (nodes, edges) where nodes is a list of NetworkEntry
        objects and edges is a list of tuples of the form (source_idx, target_idx).
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


def get_loopback_edges(g, nodes):
    """
    Given a graph and a list of nodes to check, this function finds and returns loopback
    edges (i.e., edges that connect a product node to its equivalent reactant node)

    Args:
        g: graph-tool Graph object
        nodes: List of vertices from which to find loopback edges

    Returns:
        A list of tuples of the form (source_idx, target_idx, cost=0, rxn=None,
        type="loopback")
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


def get_edge_weight(edge_obj, cf):
    name = edge_obj.__clas__.__name__
    if name in ["ComputedReaction", "OpenComputedReaction"]:
        return cf.calculate(edge_obj)
    if name in ["loopback_edge", "precursor_edge", "target_edge"]:
        return 0.0

    raise ValueError(f"Unknown edge type: {name}")
