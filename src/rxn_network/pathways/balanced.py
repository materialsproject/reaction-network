"""
Implements a class for storing balanced reaction pathways.
"""
from typing import List, Union

import numpy as np
from pymatgen.core.composition import Composition

from rxn_network.core import Pathway, Reaction
from rxn_network.pathways.basic import BasicPathway
from rxn_network.utils import limited_powerset


class BalancedPathway(BasicPathway):
    """
    Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes costs for each
    reaction.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        coefficients: List[float],
        costs: List[float],
        balanced: bool = False,
    ):
        """
        Args:
            reactions: list of ComputedReaction objects which occur along path.
            coefficients: list of coefficients to balance each of these reactions, respectively
            costs: list of corresponding costs for each reaction.
                balanced: whether or not the reaction pathway is balanced. Defaults to False.
        """
        self.coefficients = coefficients
        super().__init__(reactions=reactions, costs=costs)

        self.balanced = balanced

    def __eq__(self, other):
        if super().__eq__(other):
            return np.allclose(self.costs, other.costs)

        return False

    def __hash__(self):
        return hash((tuple(self.reactions), tuple(self.coefficients)))

    @classmethod
    def balance(
        cls,
        pathway_sets: Union[List[Pathway], List[List[Reaction]]],
        net_reaction: Reaction,
        tol=1e-6,
    ):
        """
        TODO: Implement this method

        Balances multiple reaction pathways to a net reaction.

        NOTE: Currently, to automatically balance and create a BalancedPathway object,
        you must use the PathwaySolver class.
        """

    def comp_matrix(self) -> np.ndarray:
        """
        Internal method for getting the composition matrix used in the balancing
        procedure.

        Returns:
            An array representing the composition matrix for a reaction
        """
        return np.array(
            [
                [
                    rxn.get_coeff(comp) if comp in rxn.all_comp else 0
                    for comp in self.compositions
                ]
                for rxn in self.reactions
            ]
        )

    def get_coeff_vector_for_rxn(self, rxn) -> np.ndarray:
        """
        Internal method for getting the net reaction coefficients vector.

        Args:
            rxn: Reaction object to get coefficients for

        Returns:
            An array representing the reaction coefficients vector
        """
        return np.array(
            [
                rxn.get_coeff(comp) if comp in rxn.compositions else 0
                for comp in self.compositions
            ]
        )

    def contains_interdependent_rxns(self, precursors: List[Composition]) -> bool:
        """
        Whether or not the pathway contains interdependent reactions, given a list of
        provided precursors.

        Args:
            precursors: List of precursor compositions
        """
        precursors_set = set(precursors)
        interdependent = False

        rxns = set(self.reactions)
        num_rxns = len(rxns)

        if num_rxns == 1:
            return False

        for combo in limited_powerset(rxns, num_rxns):
            size = len(combo)
            if (
                any(set(rxn.reactants).issubset(precursors_set) for rxn in combo)
                or size == 1
            ):
                continue

            other_comp = {c for rxn in (rxns - set(combo)) for c in rxn.compositions}

            unique_reactants = []
            unique_products = []
            for rxn in combo:
                unique_reactants.append(set(rxn.reactants) - precursors_set)
                unique_products.append(set(rxn.products) - precursors_set)

            overlap = [False] * size
            for i in range(size):
                for j in range(size):
                    if i == j:
                        continue
                    overlapping_phases = unique_reactants[i] & unique_products[j]
                    if overlapping_phases and (overlapping_phases not in other_comp):
                        overlap[i] = True

            if all(overlap):
                interdependent = True

        return interdependent

    @property
    def average_cost(self) -> float:
        """Returns the mean cost of the pathway"""
        return np.dot(self.coefficients, self.costs) / sum(self.coefficients)

    def __repr__(self):
        path_info = ""
        for rxn in self.reactions:
            path_info += f"{rxn} (dG = {round(rxn.energy_per_atom, 3)} eV/atom) \n"

        path_info += f"Average Cost: {round(self.average_cost,3)}"

        return path_info
