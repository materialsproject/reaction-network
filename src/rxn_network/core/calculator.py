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

    @abstractmethod
    def decorate(self, rxn: Reaction) -> "Reaction":
        """
        Evaluates the specified prop. of a reaction and stores it in the reaction data
        """
