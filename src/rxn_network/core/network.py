"""
Basic interface for a (reaction) Network
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import List

from monty.json import MSONable
from pymatgen.entries import Entry

from rxn_network.core.pathway import Pathway


class Network(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction network
    """

    def __init__(self, entries: List[Entry], enumerators, cost_function):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.entries = entries
        self.enumerators = enumerators
        self.cost_function = cost_function
        self._g = None
        self.precursors = None
        self.target = None

    @abstractmethod
    def build(self):
        """Construct the network from the supplied enumerators"""

    @abstractmethod
    def find_pathways(self, target, k) -> List[Pathway]:
        """Find reaction pathways"""

    @abstractmethod
    def set_precursors(self):
        """Set the phases used as precursors in the network"""

    @abstractmethod
    def set_target(self):
        """Set the phase used as a target in the network"""
