"""
Basic interface for a reaction Enumerator
"""
import logging
from abc import ABCMeta, abstractmethod

from monty.json import MSONable


class Enumerator(MSONable, metaclass=ABCMeta):
    """
    Base definition for a class that enumerates reactions.
    """

    def __init__(self, precursors, targets, calculators):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.precursors = precursors if precursors else []
        self.targets = targets if targets else []
        self.calculators = calculators if calculators else []

    @abstractmethod
    def enumerate(self, entries):
        """
        Enumerates the potential reactions from the list of entries
        """

    @abstractmethod
    def estimate_max_num_reactions(self, entries) -> int:
        """
        Estimate of the maximum number of reactions that may be enumerated
        from a list of entries
        """
