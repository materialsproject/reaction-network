"""
Basic interface for a cost function
"""
from abc import ABCMeta, abstractmethod

from monty.json import MSONable

from rxn_network.core.reaction import Reaction


class CostFunction(MSONable, metaclass=ABCMeta):
    """
    Base definition for a cost function
    """

    @abstractmethod
    def evaluate(self, rxn: Reaction) -> float:
        """
        Evaluates the total cost function on a reaction
        """
