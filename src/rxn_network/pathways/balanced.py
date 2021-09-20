"""
Implements a class for storing balanced reaction pathways.
"""
from typing import List, Optional, Union

import numpy as np

from rxn_network.core import Pathway, Reaction
from rxn_network.pathways.basic import BasicPathway
from rxn_network.utils import limited_powerset


class BalancedPathway(BasicPathway):
    """
    Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes cost of each
    reaction.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        coefficients: List[float],
        costs: Optional[List[float]] = None,
        balanced: Optional[bool] = None,
    ):
        """
        Args:
            rxns ([ComputedReaction]): list of ComputedReaction objects in pymatgen
                which occur along path.
            costs ([float]): list of corresponding costs for each reaction.
        """
        self.coefficients = coefficients
        super().__init__(reactions=reactions, costs=costs)

        if balanced is not None:
            self.balanced = balanced
        else:
            self.balanced = False

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
        Balances multiple reaction pathways to a net reaction
        """

        pass

    def comp_matrix(self):
        """
        Internal method for getting the composition matrix used in the balancing
        procedure.
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

    def get_coeff_vector_for_rxn(self, rxn):
        """
        Internal method for getting the net reaction coefficients vector.
        """
        return np.array(
            [
                rxn.get_coeff(comp) if comp in rxn.compositions else 0
                for comp in self.compositions
            ]
        )

    def contains_interdependent_rxns(self, precursors):
        precursors = set(precursors)
        interdependent = False

        rxns = set(self.reactions)
        num_rxns = len(rxns)

        if num_rxns == 1:
            return False, None

        for combo in limited_powerset(rxns, num_rxns):
            size = len(combo)
            if (
                any([set(rxn.reactants).issubset(precursors) for rxn in combo])
                or size == 1
            ):
                continue

            other_comp = {c for rxn in (rxns - set(combo)) for c in rxn.compositions}

            unique_reactants = []
            unique_products = []
            for rxn in combo:
                unique_reactants.append(set(rxn.reactants) - precursors)
                unique_products.append(set(rxn.products) - precursors)

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
    def average_cost(self):
        return np.dot(self.coefficients, self.costs) / sum(self.coefficients)

    def __repr__(self):
        path_info = ""
        for rxn in self.reactions:
            path_info += f"{rxn} (dG = {round(rxn.energy_per_atom, 3)} eV/atom) \n"

        path_info += f"Average Cost: {round(self.average_cost,3)}"

        return path_info
