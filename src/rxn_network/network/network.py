"Implementation of reaction network interface"
from typing import List
import graph_tool.all as gt

from rxn_network.core import Entry
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.core import Network
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.reactions.reaction_set import ReactionSet
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
        costs = rxn_set.calculate_costs(self.cost_functsion)
        rxns = rxn_set.get_rxns(self.open_elem, self.chempot)

        nodes, rxn_edges  = get_rxn_nodes_and_edges(rxns)

        g = initialize_graph()
        g.add_vertex(len(nodes))
        for i, network_entry in enumerate(nodes):
            update_vertex_props(g, g.vertex(i), {"entry": network_entry})

        edge_list = []
        for edge, cost, rxn in zip(rxn_edges, costs, rxns):
            v1 = g.vertex(edge[0])
            v2 = g.vertex(edge[1])
            edge_list.append([v1, v2, cost, rxn])

        loopback_edges = get_loopback_edges(g, nodes)
        edge_list.extend(loopback_edges)

        g.add_edge_list(
            edge_list, eprops=[g.ep["cost"], g.ep["rxn"]]
        )
        self._g = g
        self.nodes = nodes
        self.rxn_edges = rxn_edges

    def set_precursors(self, precursors):
        precursors = set(precursors)
        if precursors == self.precursors:
            pass
        elif not all([p in self.entries for p in precursors]):
            raise ValueError("One or more precursors are not included in network!")

        g = self._g
        precursors_v = g.add_vertex()
        precursors_entry = NetworkEntry(precursors, NetworkEntryType.Precursors)
        update_vertex_props(g, precursors_v, {"entry": precursors_entry})

        add_edges = []
        for v in g.vertices():
            entry = g.vp["entry"][v]
            if not entry:
                continue
            if entry.description == NetworkEntryType.Reactants:
                if entry.entries.issubset(precursors):
                    add_edges.append([precursors_v, v, 0.0, None])
            elif entry.description == NetworkEntryType.Products:
                for v2 in g.vertices():
                    entry2 = g.vp["entry"][v2]
                    if entry2.description == NetworkEntryType.Reactants:
                        if precursors.union(entry.entries).issuperset(entry2.entries):
                            add_edges.append([v, v2, 0.0, None])

        g.add_edge_list(add_edges, eprops=[g.ep["cost"], g.ep["rxn"]])

        self.precursors = precursors

    def set_target(self, target):
        if target == self.target:
            pass
        g = self._g
        target_v = g.add_vertex()
        target_entry = NetworkEntry([target], NetworkEntryType.Target)
        update_vertex_props(g,
                            target_v,
                            {"entry": target_entry})

        add_edges = []
        for v in g.vertices():
            entry = g.vp["entry"][v]
            if not entry:
                continue
            if entry.description != NetworkEntryType.Products:
                continue
            if target in entry.entries:
                add_edges.append([v, target_v, 0.0, None])

        g.add_edge_list(add_edges, eprops=[g.ep["cost"], g.ep["rxn"]])

        self.target = target

    def find_best_pathways(self, precursors, targets, num=15):
        g = self._g
        paths = []

        precursors_v = gt.find_vertex(g, g.vp["type"], 0)[0]
        target_v = gt.find_vertex(g, g.vp["type"], 3)[0]

        for num, path in enumerate(yens_ksp(g, k, precursors_v, target_v)):
            rxns = []
            weights = []

            for step, v in enumerate(path):
                g.vp["path"][v] = True

                if (
                    g.vp["type"][v] == 2
                ):  # add rxn step if current node in path is a product
                    e = g.edge(path[step - 1], v)
                    g.ep["path"][e] = True  # mark this edge as occurring on a path
                    rxns.append(g.ep["rxn"][e])
                    weights.append(g.ep["weight"][e])

            rxn_pathway = RxnPathway(rxns, weights)
            paths.append(rxn_pathway)

        for path in paths:
            print(path, "\n")

        return paths

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



