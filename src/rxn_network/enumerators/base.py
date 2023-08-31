"""
Basic interface for a reaction Enumerator.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Collection

from monty.json import MSONable

if TYPE_CHECKING:
    from rxn_network.core import Composition
    from rxn_network.entries.entry_set import GibbsEntrySet


class Enumerator(MSONable, metaclass=ABCMeta):
    """
    Base definition for a class that enumerates reactions.
    """

    def __init__(
        self,
        precursors: Collection[str | Composition] | None,
        targets: Collection[str | Composition] | None,
    ):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.precursors = precursors or []
        self.targets = targets or []

    @abstractmethod
    def enumerate(self, entries: GibbsEntrySet):
        """
        Enumerates the potential reactions from the list of entries
        """
