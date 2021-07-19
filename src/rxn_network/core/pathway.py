" Basic interface for a reaction Pathway "
import logging
from abc import ABCMeta, abstractproperty, abstractmethod
from typing import List

from monty.json import MSONable

from rxn_network.core.reaction import Reaction
from rxn_network.pathways.balanced import BalancedPathway


class Pathway(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction pathway "

    @property
    def reactions(self) -> List[Reaction]:
        return self._reactions
        " List of reactions in this Pathway "

    @property
    def entries(self):
        " Entry objects in this Pathway "
        return {entry for rxn in self.reactions for entry in rxn.entries}

    @property
    def all_reactants(self):
        " Entries serving as a reactant in any sub reaction "
        return {entry for rxn in self.reactions for entry in rxn.reactants}

    @property
    def all_products(self):
        " Entries serving as a product in any sub reaction "
        return {entry for rxn in self.reactions for entry in rxn.products}

    @property
    def compositions(self):
        return list(self.all_reactants | self.all_products)

    @property
    def reactants(self):
        " The reactants of this whole reaction pathway "
        return self.all_reactants - self.all_products

    @property
    def products(self):
        " The products of this whole reaction pathway "
        return self.all_products - self.all_reactants

    @property
    def intermediates(self):
        " Intermediates as entries in this reaction pathway "
        return self.all_products & self.all_reactants

    @property
    def energy(self):
        " Total energy of this reaction pathway "
        return sum([rxn.energy for rxn in self.reactions])

    @property
    def energy_per_atom(self):
        " Total energy per atom of this reaction pathway"
        return sum([rxn.energy_per_atom for rxn in self.reactions])


class Solver(MSONable, metaclass=ABCMeta):
    def __init__(self, entries, pathways):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")

        self._entries = entries
        self._pathways = pathways
        self._reactions = list({rxn for path in self._pathways for rxn in
                                path.reactions})

        self._costs = [cost for path in self._pathways for cost in path.costs]

    @abstractmethod
    def solve(self, net_rxn) -> List[BalancedPathway]:
        "Solve paths"

    @property
    def entries(self):
        return self._entries

    @property
    def pathways(self):
        return self._pathways

    @property
    def reactions(self):
        return self._reactions

    @property
    def costs(self):
        return self._costs

    @property
    def num_rxns(self):
        return len(self.reactions)

    @property
    def num_entries(self):
        return len(self._entries)