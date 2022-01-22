"""
Basic interface for a reaction network.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Iterable, List

from monty.json import MSONable

from rxn_network.core.cost_function import CostFunction
from rxn_network.core.enumerator import Enumerator
from rxn_network.core.pathway import Pathway
from rxn_network.entries.entry_set import GibbsEntrySet


class Network(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction network.
    """

    def __init__(
        self,
        entries: GibbsEntrySet,
        enumerators: Iterable[Enumerator],
        cost_function: CostFunction,
    ):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.entries = entries
        self.enumerators = enumerators
        self.cost_function = cost_function
        self._g = None
        self._precursors = None
        self._target = None

        self.entries.build_indices()

    @abstractmethod
    def build(self):
        """Construct the network from the supplied enumerators"""

    @abstractmethod
    def find_pathways(self, target, k) -> List[Pathway]:
        """Find reaction pathways"""

    @abstractmethod
    def set_precursors(self, precursors):
        """Set the phases used as precursors in the network"""

    @abstractmethod
    def set_target(self, target):
        """Set the phase used as a target in the network"""

    @property
    def precursors(self):
        """The phases used as precursors in the network"""
        return self._precursors

    @property
    def target(self):
        """The phase used as a target in the network"""
        return self._target

    @property
    def graph(self):
        """Returns the network object in graph-tool"""
        return self._g

    @property
    def chemsys(self):
        """A string representing the chemical system (elements) of the network"""
        return "-".join(sorted(self.entries.chemsys))
