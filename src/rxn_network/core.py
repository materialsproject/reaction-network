import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List

from monty.json import MSONable

from pymatgen.entries import Entry
from pymatgen.core.composition import Composition, Element


class Reaction(MSONable, metaclass=ABCMeta):
    " Base definition for a Reaction "

    @abstractproperty
    def reactants(self) -> List[Composition]:
        " List of reactants for this reaction "

    @abstractproperty
    def products(self) -> List[Composition]:
        " List of products for this reaction "

    @abstractproperty
    def coefficients(self) -> np.array:
        """
        Coefficients of the reaction
        """

    @abstractproperty
    def energy(self):
        " The energy of this reaction in total eV "

    @property
    def compositions(self) -> List[Composition]:
        """
        List of all compositions in the reaction.
        """
        return self.reactants + self.products

    @property
    def elements(self) -> List[Element]:
        """
        List of elements in the reaction
        """
        return list(set(el for comp in self.compositions for el in comp.elements))

    @property
    def num_atoms(self) -> float:
        " Total number of atoms in this reaction "
        return (
            sum(
                [
                    comp[element] * abs(coeff)
                    for element in self.elements
                    for coeff, comp in zip(self.coefficients, self.compositions)
                ]
            )
            / 2
        )

    @property
    def energy_per_atom(self) -> float:
        " The energy per atom of this reaction in eV "
        return self.energy / self.num_atoms


class CostFunction(MSONable, metaclass=ABCMeta):
    " Base definition for a cost function "

    @abstractmethod
    def evaluate(self, rxn: Reaction) -> float:
        " Evaluates the cost function on a reaction "


class Enumerator(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction enumeration methodology "

    @abstractmethod
    def estimate_num_reactions(self, entries) -> int:
        " Estimate of the number of reactions from a list of entires "

    @abstractmethod
    def enumerate(self, entries) -> List[Reaction]:
        " Enumerates the potential reactions from the list of entries "


class Pathway(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction pathway "

    @abstractproperty
    def reactions(self) -> List[Reaction]:
        " List of reactions in this Pathway "

    @property
    def energy(self):
        return sum([rxn.energy for rxn in self.reactions])

    @property
    def energy_per_atom(self):
        " Total energy per atom of this reaction pathway"
        return sum([rxn.energy_per_atom for rxn in self.reactions])


class ReactionNetwork(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction network "

    def __init__(self, entries: List[Entry], enumerators, cost_function):

        self.entries = entries
        self.enumerators = enumerators
        self.cost_function = cost_function

    @abstractmethod
    def find_best_rxn_pathways(self, precursors, targets, num=15):
        " Find the N best reaction pathways "
