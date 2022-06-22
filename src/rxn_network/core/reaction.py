"""
Basic interface for a (chemical) Reaction
"""
from abc import ABCMeta, abstractmethod
from typing import List
from functools import cached_property

import numpy as np
from monty.json import MSONable
from pymatgen.core.composition import Composition, Element


class Reaction(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction class.
    """

    @property
    @abstractmethod
    def reactants(self) -> List[Composition]:
        """List of reactants for this reaction"""

    @property
    @abstractmethod
    def products(self) -> List[Composition]:
        """List of products for this reaction"""

    @property
    @abstractmethod
    def coefficients(self) -> np.ndarray:
        """Coefficients of the reaction"""

    @property
    @abstractmethod
    def energy(self) -> float:
        """The energy of this reaction in total eV"""

    @property
    def compositions(self) -> List[Composition]:
        """List of all compositions in the reaction"""
        return self.reactants + self.products

    @property
    def elements(self) -> List[Element]:
        """List of elements in the reaction"""
        return list(set(el for comp in self.compositions for el in comp.elements))
