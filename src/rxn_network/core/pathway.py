"""
Basic interface for a reaction Pathway
"""
from abc import ABCMeta
from typing import List

from monty.json import MSONable

from rxn_network.core.reaction import Reaction


class Pathway(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction pathway
    """

    @property
    def reactions(self) -> List[Reaction]:
        """List of reactions in this Pathway"""
        return self._reactions

    @property
    def entries(self):
        """Entry objects in this Pathway"""
        return {entry for rxn in self.reactions for entry in rxn.entries}

    @property
    def all_reactants(self):
        """Entries serving as a reactant in any sub reaction"""
        return {entry for rxn in self.reactions for entry in rxn.reactants}

    @property
    def all_products(self):
        """Entries serving as a product in any sub reaction"""
        return {entry for rxn in self.reactions for entry in rxn.products}

    @property
    def compositions(self):
        """Compositions in the reaction"""
        return list(self.all_reactants | self.all_products)

    @property
    def reactants(self):
        """The reactants of this whole reaction pathway"""
        return self.all_reactants - self.all_products

    @property
    def products(self):
        """The products of this whole reaction pathway"""
        return self.all_products - self.all_reactants

    @property
    def intermediates(self):
        """Intermediates as entries in this reaction pathway"""
        return self.all_products & self.all_reactants

    @property
    def energy(self):
        """Total energy of this reaction pathway"""
        return sum([rxn.energy for rxn in self.reactions])

    @property
    def energy_per_atom(self):
        """Total energy per atom of this reaction pathway"""
        return sum([rxn.energy_per_atom for rxn in self.reactions])
