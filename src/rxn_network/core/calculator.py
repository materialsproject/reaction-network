"""
Basic interface for a reaction cost Calculator
"""
from abc import ABCMeta, abstractmethod
from typing import List

from monty.json import MSONable

from rxn_network.core.reaction import Reaction


class Calculator(MSONable, metaclass=ABCMeta):
    """
    Base definition for a property calculator
    """

    @abstractmethod
    def calculate(self, rxn: Reaction) -> float:
        """
        Evaluates the specified property of a reaction
        """

    def decorate(self, rxn: Reaction) -> Reaction:
        """
        Decorates the reaction (in place) with the calculated property by
        storing the value within the reaction's data dictionary.

        Args:
            rxn: The reaction object.

        Returns:
            The reaction object, modified in place
        """
        if not getattr(rxn, "data"):
            rxn.data = {}

        rxn.data[self.name] = self.calculate(rxn)

        return rxn

    def calculate_many(self, rxns: List[Reaction]) -> List[float]:
        """
        Calculates the competitiveness score for a list of reactions by enumerating
        competing reactions, evaluating their cost with the supplied cost function, and
        then using the c-score formula, i.e. the _get_c_score() method, to determine the
        competitiveness score. Parallelized with ray.

        Args:
            rxns: the list of ComputedReaction objects to be evaluated

        Returns:
            The list of competitiveness scores
        """
        results = []
        for rxn in rxns:
            results.append(self.calculate(rxn))

        return results

    def decorate_many(self, rxns: List[Reaction]) -> List[Reaction]:
        """
        Decorates a list of reactions with the competitiveness score. Parallelized with
        ray.

        Args:
            rxns: the list of ComputedReaction objects to be decorated

        Returns:
            The list of decorated ComputedReaction objects
        """
        decorated_rxns = []
        for rxn in rxns:
            decorated_rxns.append(self.decorate(rxn))

        return decorated_rxns

    @property
    @abstractmethod
    def name(self):
        """
        The name of the calculator; used to store the value within the reaction's data dictionary
        """
