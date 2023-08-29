"""
Basic interface for a (chemical) Reaction
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from monty.json import MSONable

if TYPE_CHECKING:
    from numpy import ndarray
    from pymatgen.core.composition import Element

    from rxn_network.core import Composition


class Reaction(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction class.
    """

    @property
    @abstractmethod
    def reactants(self) -> list[Composition]:
        """List of reactants for this reaction"""

    @property
    @abstractmethod
    def products(self) -> list[Composition]:
        """List of products for this reaction"""

    @property
    @abstractmethod
    def coefficients(self) -> ndarray:
        """Coefficients of the reaction"""

    @property
    @abstractmethod
    def energy(self) -> float:
        """The energy of this reaction in total eV"""

    @property
    @abstractmethod
    def compositions(self) -> list[Composition]:
        """List of all compositions in the reaction"""

    @property
    def elements(self) -> list[Element]:
        """List of elements in the reaction"""
        return list(set(el for comp in self.compositions for el in comp.elements))
