" Basic interface for a Reaction"

from abc import ABCMeta, abstractproperty
from typing import List

import numpy as np
from monty.json import MSONable
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

    @abstractproperty
    def energy(self):
        " The energy of this reaction in total eV "

    @property
    def energy_per_atom(self) -> float:
        " The energy per atom of this reaction in eV "
        return self.energy / self.num_atoms
