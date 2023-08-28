"""
Basic interface and implementation for a Calculator and CostFunction.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from monty.json import MSONable

if TYPE_CHECKING:
    from rxn_network.reactions.base import Reaction


class Calculator(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction property calculator.
    """

    @abstractmethod
    def calculate(self, rxn: Reaction) -> float:
        """
        Evaluates a particular property of a reaction relevant to its cost ranking.
        """

    def calculate_many(self, rxns: list[Reaction]) -> list[float]:
        """
        Convenience method for performing calculate() on a list of reactions.

        Args:
            rxns: the list of Reaction objects to be evaluated

        Returns:
            A list of the reactions' calculated property values.
        """
        results = []
        for rxn in rxns:
            results.append(self.calculate(rxn))

        return results

    def decorate(self, rxn: Reaction) -> Reaction:
        """
        Returns a copy of the reaction with the calculated property by
        storing the value within the reaction's data dictionary.

        Args:
            rxn: The reaction object.

        Returns:
            A deep copy of the original reaction with a modified data dict.
        """
        new_rxn = deepcopy(rxn)

        if not getattr(new_rxn, "data"):
            new_rxn.data = {}

        new_rxn.data[self.name] = self.calculate(new_rxn)
        return new_rxn

    def decorate_many(self, rxns: list[Reaction]) -> list[Reaction]:
        """
        Convenience method for performing decorate() on a list of reactions.

        Args:
            rxns: the list of Reaction objects to be decorated

        Returns:
            A list of new (copied) reactions with modified data containing their
            calculated properties.
        """
        new_rxns = []
        for rxn in rxns:
            new_rxns.append(self.decorate(rxn))
        return new_rxns


class CostFunction(MSONable, metaclass=ABCMeta):
    """
    Base definition for a cost function.
    """

    @abstractmethod
    def evaluate(self, rxn: Reaction) -> float:
        """
        Evaluates the specified cost function equation on a reaction object.
        """
