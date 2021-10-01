"""
Basic interface for a reaction pathway Solver
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import List

from monty.json import MSONable
from pymatgen.entries.entry_tools import EntrySet

from rxn_network.core.pathway import Pathway
from rxn_network.core.reaction import Reaction


class Solver(MSONable, metaclass=ABCMeta):
    """
    Base definition for a pathway solver class
    """

    def __init__(self, entries, pathways):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")

        self._entries = entries
        self._pathways = pathways

        rxns = []
        costs = []

        for path in self._pathways:
            for rxn, cost in zip(path.reactions, path.costs):
                if rxn not in rxns:
                    rxns.append(rxn)
                    costs.append(cost)

        self._reactions = rxns
        self._costs = costs

    @abstractmethod
    def solve(self, net_rxn) -> List[Pathway]:
        """Solve paths"""

    @property
    def entries(self) -> EntrySet:
        """Entry set used in solver"""
        return self._entries

    @property
    def pathways(self) -> List[Pathway]:
        """Pathways used in solver class"""
        return self._pathways

    @property
    def reactions(self) -> List[Reaction]:
        """Reactions used in solver class"""
        return self._reactions

    @property
    def costs(self) -> List[float]:
        """ Costs used in solver class"""
        return self._costs

    @property
    def num_rxns(self) -> int:
        """Length of reaction list"""
        return len(self.reactions)

    @property
    def num_entries(self) -> int:
        """Length of entry list"""
        return len(self._entries)
