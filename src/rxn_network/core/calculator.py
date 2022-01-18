"""
Basic interface for a reaction cost Calculator
"""
from abc import ABCMeta, abstractmethod

from monty.json import MSONable

from rxn_network.core.reaction import Reaction


class Calculator(MSONable, metaclass=ABCMeta):
    """
    Base definition for a property calculator
    """

    @abstractmethod
    def calculate(self, rxn: Reaction) -> float:
        """
        Evaluates the specified property of a reaction
        """

    def decorate(self, rxn: Reaction) -> Reaction:
        """
        Decorates the reaction (in place) with the calculated property by
        storing the value within the reaction's data dictionary.

        Args:
            rxn: The reaction object.

        Returns:
            The reaction object, modified in place
        """
        if not getattr(rxn, "data"):
            rxn.data = {}

        rxn.data[self.name] = self.calculate(rxn)

        return rxn

    @property
    @abstractmethod
    def name(self):
        """
        The name of the calculator; used to store the value within the reaction's data dictionary
        """
