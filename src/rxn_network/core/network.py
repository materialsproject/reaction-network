" Basic interface for a reaction Network "
import logging
from abc import ABCMeta, abstractmethod
from monty.json import MSONable
from typing import List
from pymatgen.entries import Entry


class Network(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction network "

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
        "Construct the network from the supplied enumerators"
