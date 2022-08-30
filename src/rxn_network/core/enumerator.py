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

    def __init__(self, precursors, targets):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.logger.setLevel("INFO")
        self.precursors = precursors if precursors else []
        self.targets = targets if targets else []

    @abstractmethod
    def enumerate(self, entries):
        """
        Enumerates the potential reactions from the list of entries
        """
