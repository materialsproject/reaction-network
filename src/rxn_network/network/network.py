"""
Implementation of reaction network interface.
"""
from typing import Iterable, List, Optional, Union

from graph_tool.util import find_edge, find_vertex
from pymatgen.entries import Entry

from rxn_network.core import CostFunction, Enumerator, Network
from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.network.gt import (
    initialize_graph,
    load_graph,
    save_graph,
    update_vertex_props,
    yens_ksp,
)
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
        chempot: float = 0.0,
    ):
        """
        Initialize a ReactionNetwork object for a set of entires, enumerator,
        and cost function. The network can be constructed by calling build().

        Note: the precursors and target must be set by calling set_precursors() and
        set_target() respectively.

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

        Returns:
            None
        """
        rxn_set = self._get_rxns()
        costs = rxn_set.calculate_costs(self.cost_function)
        rxns = rxn_set.get_rxns()

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
            self.set_target(target)
            print(f"PATHS to {self.target.composition.reduced_formula} \n")
            print("--------------------------------------- \n")
            pathways = self._shortest_paths(k=k)
            paths.extend(pathways)

        return paths

    def set_precursors(self, precursors: Iterable[Union[Entry, str]]):
        """
        Sets the precursors of the network. Removes all references to previous
        precursors.

        If entries are provided, will use the entries to set the precursors. If strings
        are provided, will automatically find minimum-energy entries with matching reduced_formula.

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

        if self.precursors:
            precursors_v = find_vertex(
                g, g.vp["type"], NetworkEntryType.Precursors.value
            )[0]
            g.remove_vertex(precursors_v)
            loopback_edges = find_edge(g, g.ep["type"], "loopback_precursors")
            for e in loopback_edges:
                g.remove_edge(e)
        elif not all(p in self.entries for p in precursors):
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

        self._target = target

    def load_graph(self, filename: str):
        """
        Loads graph-tool graph from file.

        Args:
            filename: Filename of graph object to load (for example, .gt or .gt.gz format)

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
        """Gets reaction set by running all enumerators"""
        rxns = []
        for enumerator in self.enumerators:
            rxns.extend(enumerator.enumerate(self.entries))

        rxns = ReactionSet.from_rxns(
            rxns, self.entries, open_elem=self.open_elem, chempot=self.chempot
        )
        return rxns

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
        Convenience constructor method that loads a ReactionNetwork object from a dictionary (MSONable version) and a
        filename (to load graph object in graph-tool).

        Args:
            d: Dictionary containing the ReactionNetwork object
            filename: Filename of graph object to load (for example, .gt or .gt.gz format)

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
        d["precursors"] = list(self.precursors)
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
            f"ReactionNetwork for chemical system: "
            f"{self.chemsys}, "
            f"with Graph: {str(self._g)}"
        )
