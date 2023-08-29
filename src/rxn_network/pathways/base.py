"""
Basic interface for a reaction pathway.
"""
from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

from monty.json import MSONable

if TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedEntry

    from rxn_network.core import Composition
    from rxn_network.reactions.base import Reaction


class Pathway(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction pathway.
    """

    _reactions: list[Reaction]

    @property
    def entries(self) -> set[ComputedEntry]:
        """Entry objects in this pathway"""
        return {entry for rxn in self._reactions for entry in rxn.entries}

    @property
    def all_reactants(self) -> set[Composition]:
        """Reactant compositions for all reactions in the pathway"""
        return {entry for rxn in self._reactions for entry in rxn.reactants}

    @property
    def all_products(self) -> set[Composition]:
        """Product compositions reaction in the pathway"""
        return {entry for rxn in self._reactions for entry in rxn.products}

    @property
    def compositions(self) -> list[Composition]:
        """All compositions in the reaction"""
        return list(self.all_reactants | self.all_products)

    @property
    def reactants(self) -> set[Composition]:
        """The reactant compositions of this whole/net reaction pathway"""
        return self.all_reactants - self.all_products

    @property
    def products(self) -> set[Composition]:
        """The product compositions of this whole/net reaction pathway"""
        return self.all_products - self.all_reactants

    @property
    def intermediates(self) -> set[Composition]:
        """Intermediate compositions in this reaction pathway"""
        return self.all_products & self.all_reactants

    @property
    def energy(self) -> float:
        """Total energy of this reaction pathway"""
        return sum(rxn.energy for rxn in self._reactions)

    @property
    def energy_per_atom(self) -> float:
        """Total normalized energy of this reaction pathway"""
        return sum(rxn.energy_per_atom for rxn in self._reactions)
