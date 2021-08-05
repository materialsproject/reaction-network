" Basic interface for a reaction Enumerator "
import logging
from abc import ABCMeta, abstractmethod
from typing import List

from monty.json import MSONable

from rxn_network.core.reaction import Reaction


class Enumerator(MSONable, metaclass=ABCMeta):
    "Base definition for a reaction enumeration methodology"

    def __init__(self, precursors, target, calculators):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.precursors = precursors
        self.target = target

        if not calculators:
            calculators = []

        self.calculators = calculators

    @abstractmethod
    def enumerate(self, entries) -> List[Reaction]:
        "Enumerates the potential reactions from the list of entries"

    @abstractmethod
    def estimate_num_reactions(self, entries) -> int:
        "Estimate of the number of reactions from a list of entires"
