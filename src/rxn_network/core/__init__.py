" Core interfaces for the reaction-network package. "

from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from typing import List
import logging

import numpy as np
from monty.json import MSONable
from pymatgen.core.composition import Composition, Element
from pymatgen.entries import Entry

from rxn_network.core.reaction import Reaction
from rxn_network.core.pathway import Pathway


class Calculator(MSONable, metaclass=ABCMeta):
    " Base definition for a property calculator "

    @abstractmethod
    def calculate(self, rxn: Reaction) -> float:
        "Evaluates the specified property of a reaction"

    @abstractmethod
    def decorate(self, rxn: Reaction) -> "Reaction":
        "Evaluates the specified prop. of a reaction and stores it in the reaction data"


class CostFunction(MSONable, metaclass=ABCMeta):
    " Base definition for a cost function "

    @abstractmethod
    def evaluate(self, rxn: Reaction) -> float:
        " Evaluates the total cost function on a reaction "


class Enumerator(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction enumeration methodology "

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
        " Enumerates the potential reactions from the list of entries "

    @abstractmethod
    def estimate_num_reactions(self, entries) -> int:
        " Estimate of the number of reactions from a list of entires "


class ReactionNetwork(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction network "

    def __init__(self, entries: List[Entry], enumerators, cost_function):

        self.entries = entries
        self.enumerators = enumerators
        self.cost_function = cost_function

    @abstractmethod
    def find_best_rxn_pathways(self, precursors, targets, num=15):
        " Find the N best reaction pathways "
