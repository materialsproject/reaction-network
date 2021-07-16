"Implementation of reaction network interface"
from typing import List
import graph_tool.all as gt

from rxn_network.core import Entry
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.core import Network
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.pathways.utils import shortest_path_to_reaction_pathway
from rxn_network.network.utils import get_rxn_nodes_and_edges, get_loopback_edges
from rxn_network.network.adaptors.gt import initialize_graph, yens_ksp, \
    update_vertex_props


class ReactionNetwork(Network):
    def __init__(self,
                 entries: GibbsEntrySet,
                 enumerators,
                 cost_function,
                 open_elem=None,
                 chempot=None
    ):
        super().__init__(entries=entries, enumerators=enumerators, cost_function=cost_function)

        self.open_elem = open_elem
        self.chempot = chempot
        self.chemsys = "-".join(sorted(entries.chemsys))

        self._g = None
        self.precursors = None
        self.target = None

    def build(self):
        rxn_set = self._get_rxns()
        costs = rxn_set.calculate_costs(self.cost_function)
        rxns = rxn_set.get_rxns(self.open_elem, self.chempot)

        nodes, rxn_edges  = get_rxn_nodes_and_edges(rxns)

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

        loopback_edges = get_loopback_edges(g, nodes)
        edge_list.extend(loopback_edges)

        g.add_edge_list(
            edge_list, eprops=[g.ep["cost"], g.ep["rxn"], g.ep["type"]]
        )
        self._g = g
        self.nodes = nodes
        self.rxn_edges = rxn_edges

    def set_precursors(self, precursors):
        precursors = set(precursors)
        if precursors == self.precursors:
            return
        elif not all([p in self.entries for p in precursors]):
            raise ValueError("One or more precursors are not included in network!")

        g = self._g
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
                        if precursors.union(entry.entries).issuperset(entry2.entries):
                            add_edges.append((v, v2, 0.0, None, "loopback_precursors"))

        g.add_edge_list(add_edges, eprops=[g.ep["cost"], g.ep["rxn"], g.ep["type"]])

        self.precursors = precursors

    def set_target(self, target):
        g = self._g
        if target == self.target:
            return
        elif self.target:
            target_v = gt.find_vertex(g, g.vp["type"], NetworkEntryType.Target.value)[0]
            g.remove_vertex(target_v)

        target_v = g.add_vertex()
        target_entry = NetworkEntry([target], NetworkEntryType.Target)
        props = {"entry": target_entry, "type": target_entry.description.value}
        update_vertex_props(g,
                            target_v,
                            props)

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

    def find_basic_pathways(self, k=15):
        g = self._g
        paths = []

        precursors_v = gt.find_vertex(g, g.vp["type"],
                                      NetworkEntryType.Precursors.value)[0]
        target_v = gt.find_vertex(g, g.vp["type"], NetworkEntryType.Target.value)[0]

        for path in yens_ksp(g, k, precursors_v, target_v):
            paths.append(shortest_path_to_reaction_pathway(g, path))

        for path in paths:
            print(path, "\n")

        return paths

    def find_balanced_pathways(self):
        pass

    def _get_rxns(self) -> ReactionSet:
        rxns = []
        for enumerator in self.enumerators:
            rxns.extend(enumerator.enumerate(self.entries))

        rxns = ReactionSet.from_rxns(rxns, self.entries)
        return rxns

    @property
    def graph(self):
        return self._g

    def __repr__(self):
        return (
            f"ReactionNetwork for chemical system: "
            f"{self.chemsys}, "
            f"with Graph: {str(self._g)}"
        )



