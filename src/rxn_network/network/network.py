"Implementation of reaction network interface"
from typing import List
from rxn_network.core import Entry
from rxn_network.core import Network
from rxn_network.network.entry import NetworkEntry, NetworkEntryType
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.pathways.utils import yens_ksp

class ReactionNetwork(Network):
    def __init__(self,
                 entries: List[Entry],
                 enumerators,
                 cost_function,
                 open_elem=None,
                 chempot=None
    ):
        super().__init__(entries=entries, enumerators=enumerators, cost_function=cost_function)

        self.open_elem = open_elem
        self.chempot = chempot
        self._g = None

    def build(self):
        rxns = self._get_rxns()
        nodes, edges  = self._get_rxn_nodes_and_edges(rxns)

        return nodes, edges

    def find_best_pathways(self, precursors, targets, num=15):
        pass

    def _get_rxns(self) -> ReactionSet:
        rxns = []
        for enumerator in self.enumerators:
            rxns.extend(enumerator.enumerate(self.entries))

        rxns = ReactionSet.from_rxns(rxns, self.entries).get_rxns(self.open_elem,
                                                                  self.chempot)
        return rxns

    def _get_rxn_nodes_and_edges(self, rxns):
        nodes, edges = [], []

        for rxn in rxns:
            reactant_node = NetworkEntry(rxn.reactant_entries,
                                         NetworkEntryType.Reactants)
            product_node = NetworkEntry(rxn.reactant_entries, NetworkEntryType.Products)

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

    @property
    def graph(self):
        return self._g



