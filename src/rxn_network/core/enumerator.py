"""
Basic interface for a reaction Enumerator
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import List

from monty.json import MSONable


class Enumerator(MSONable, metaclass=ABCMeta):
    """
    Base definition for the reaction enumeration methodology
    """

    def __init__(self, precursors, target, calculators):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.precursors = precursors
        self.target = target

        if not calculators:
            calculators = []

        self.calculators = calculators

    @abstractmethod
    def enumerate(self, entries):
        """
        Enumerates the potential reactions from the list of entries
        """

    @abstractmethod
    def estimate_max_num_reactions(self, entries) -> int:
        """
        Estimate of the number of reactions from a list of entries
        """
