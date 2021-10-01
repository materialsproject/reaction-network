"""
Implementation of an actual reaction network interface.
"""

from typing import List, Optional

from graph_tool.util import find_edge, find_vertex

from rxn_network.core import CostFunction, Enumerator, Network
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.network.gt import (
    initialize_graph,
    update_vertex_props,
    yens_ksp,
)
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.network.utils import get_loopback_edges, get_rxn_nodes_and_edges
from rxn_network.pathways.basic import BasicPathway
from rxn_network.reactions.reaction_set import ReactionSet


class ReactionNetwork(Network):
    """
    Main reaction network class for building graphs of reactions and performing
    pathfinding.
    """

    def __init__(
        self,
        entries: GibbsEntrySet,
        enumerators: List[Enumerator],
        cost_function: CostFunction,
        open_elem: Optional[str] = None,
        chempot: Optional[float] = None,
    ):
        """
        Initialize a ReactionNetwork object for a set of entires, enumerator,
        and cost function. The network can be constructed by calling build().

        Args:
            entries: iterable of entry-like objects
            enumerators: iterable of enumerators which will be called during the
                build of the network
            cost_function: the function used to calculate the cost of each reaction edge
            open_elem: Optional name of an element that is kept open during reaction
            chempot: Optional associated chemical potential of open element
        """
        super().__init__(
            entries=entries, enumerators=enumerators, cost_function=cost_function
        )
        self.open_elem = open_elem
        self.chempot = chempot

    def build(self):
        """
        Construct the reaction network graph object and store under the "graph"
        attribute. Does NOT initialize precursors or target; you must call set_precursors()
        or set_target() to do so.

        Returns: None

        """
        rxn_set = self._get_rxns()
        costs = rxn_set.calculate_costs(self.cost_function)
        rxns = rxn_set.get_rxns(self.open_elem, self.chempot)

        self.logger.info("Building graph from reactions...")
        nodes, rxn_edges = get_rxn_nodes_and_edges(rxns)

        g = initialize_graph()
        g.add_vertex(len(nodes))
        for i, network_entry in enumerate(nodes):
            props = {"entry": network_entry, "type": network_entry.description.value}
            update_vertex_props(g, g.vertex(i), props)

        edge_list = []
        for edge, cost, rxn in zip(rxn_edges, costs, rxns):
            v1 = g.vertex(edge[0])
            v2 = g.vertex(edge[1])
            edge_list.append((v1, v2, cost, rxn, "reaction"))

        edge_list.extend(get_loopback_edges(g, nodes))

        g.add_edge_list(edge_list, eprops=[g.ep["cost"], g.ep["rxn"], g.ep["type"]])

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
            if type(target) == str:
                target = self.entries.get_min_entry_by_formula(target)
            self.set_target(target)
            print(f"PATHS to {target.composition.reduced_formula} \n")
            print("--------------------------------------- \n")
            pathways = self._shortest_paths(k=k)
            paths.extend(pathways)

        return paths

    def set_precursors(self, precursors):
        """

        Args:
            precursors:

        Returns:

        """
        g = self._g

        precursors = set(precursors)
        if precursors == self.precursors:
            return
        elif self.precursors:
            precursors_v = find_vertex(
                g, g.vp["type"], NetworkEntryType.Precursors.value
            )[0]
            g.remove_vertex(precursors_v)
            loopback_edges = find_edge(g, g.ep["type"], "loopback_precursors")
            for e in loopback_edges:
                g.remove_edge(e)
        elif not all([p in self.entries for p in precursors]):
            raise ValueError("One or more precursors are not included in network!")

        precursors_v = g.add_vertex()
        precursors_entry = NetworkEntry(precursors, NetworkEntryType.Precursors)
        props = {"entry": precursors_entry, "type": precursors_entry.description.value}
        update_vertex_props(g, precursors_v, props)

        add_edges = []
        for v in g.vertices():
            entry = g.vp["entry"][v]
            if not entry:
                continue
            if entry.description.value == NetworkEntryType.Reactants.value:
                if entry.entries.issubset(precursors):
                    add_edges.append((precursors_v, v, 0.0, None, "precursors"))
            elif entry.description.value == NetworkEntryType.Products.value:
                for v2 in g.vertices():
                    entry2 = g.vp["entry"][v2]
                    if entry2.description.value == NetworkEntryType.Reactants.value:
                        if precursors.issuperset(entry2.entries):
                            continue
                        if precursors.union(entry.entries).issuperset(entry2.entries):
                            add_edges.append((v, v2, 0.0, None, "loopback_precursors"))

        g.add_edge_list(add_edges, eprops=[g.ep["cost"], g.ep["rxn"], g.ep["type"]])

        self.precursors = precursors

    def set_target(self, target):
        """

        Args:
            target:

        Returns:

        """
        g = self._g
        if target == self.target:
            return
        elif self.target or target is None:
            target_v = find_vertex(g, g.vp["type"], NetworkEntryType.Target.value)[0]
            g.remove_vertex(target_v)

        target_v = g.add_vertex()
        target_entry = NetworkEntry([target], NetworkEntryType.Target)
        props = {"entry": target_entry, "type": target_entry.description.value}
        update_vertex_props(g, target_v, props)

        add_edges = []
        for v in g.vertices():
            entry = g.vp["entry"][v]
            if not entry:
                continue
            if entry.description.value != NetworkEntryType.Products.value:
                continue
            if target in entry.entries:
                add_edges.append([v, target_v, 0.0, None, "target"])

        g.add_edge_list(add_edges, eprops=[g.ep["cost"], g.ep["rxn"], g.ep["type"]])

        self.target = target

    def _shortest_paths(self, k=15):
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

    def _get_rxns(self) -> ReactionSet:
        "Gets reaction set by running all enumerators"
        rxns = []
        for enumerator in self.enumerators:
            rxns.extend(enumerator.enumerate(self.entries))

        rxns = ReactionSet.from_rxns(rxns, self.entries)
        return rxns

    @staticmethod
    def _path_from_graph(g, path):
        rxns = []
        costs = []

        for step, v in enumerate(path):
            if g.vp["type"][v] == NetworkEntryType.Products.value:
                e = g.edge(path[step - 1], v)

                rxns.append(g.ep["rxn"][e])
                costs.append(g.ep["cost"][e])

        return BasicPathway(reactions=rxns, costs=costs)

    @property
    def graph(self):
        return self._g

    @property
    def chemsys(self):
        return "-".join(sorted(self.entries.chemsys))

    def __repr__(self):
        return (
            f"ReactionNetwork for chemical system: "
            f"{self.chemsys}, "
            f"with Graph: {str(self._g)}"
        )
