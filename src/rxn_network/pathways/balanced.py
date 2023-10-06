"""Implements a class for storing a balanced reaction pathway."""
from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import numpy as np

from rxn_network.pathways.basic import BasicPathway
from rxn_network.reactions.set import ReactionSet
from rxn_network.utils.funcs import limited_powerset

if TYPE_CHECKING:
    from rxn_network.core import Composition
    from rxn_network.pathways.base import Pathway
    from rxn_network.costs.base import CostFunction
    from rxn_network.reactions.base import Reaction


class BalancedPathway(BasicPathway):
    """Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes costs for each
    reaction.
    """

    def __init__(
        self,
        reactions: list[Reaction],
        coefficients: list[float],
        costs: list[float],
        balanced: bool | None = None,
        aggregate_function: Callable | str = "sum",
    ):
        """Args:
        reactions: list of ComputedReaction objects which occur along path.
        coefficients: list of coefficients to balance each corresponding reaction.
        costs: list of corresponding costs for each reaction.
        balanced: whether or not the reaction pathway is balanced.
        Defaults to False and should ideally be set through PathwaySolver.
        """
        Args:
            reactions: list of ComputedReaction objects which occur along path.
            coefficients: list of coefficients to balance each corresponding reaction.
            costs: list of corresponding costs for each reaction.
            balanced: whether or not the reaction pathway is balanced.
                Defaults to False and should ideally be set through PathwaySolver.
            aggregate_function: function to use to aggregate reaction selectivities
                (e.g., C1 and C2). Defaults to "sum".
        """
        self.coefficients = coefficients
        super().__init__(reactions=reactions, costs=costs)

        if aggregate_function == "max":
            self.aggregate_function = max
        elif aggregate_function == "mean":
            self.aggregate_function = np.mean  # type: ignore
        elif aggregate_function == "sum":
            self.aggregate_function = sum  # type: ignore
        elif isinstance(aggregate_function, str):
            raise ValueError(
                "Provided aggregate name is not a known function; please provide the"
                " function directly."
            )

        self.balanced = balanced

    def get_comp_matrix(self) -> np.ndarray:
        """Gets the composition matrix used in the balancing procedure.

        Returns:
            An array representing the composition matrix for a reaction
        """
        return np.array(
            [
                [rxn.get_coeff(comp) if comp in rxn.all_comp else 0 for comp in self.compositions]
                for rxn in self.reactions
            ]
        )

    def get_coeff_vector_for_rxn(self, rxn: Reaction) -> np.ndarray:
        """Gets the net reaction coefficients vector.

        Args:
            rxn: Reaction object to get coefficients for

        Returns:
            An array representing the reaction coefficients vector
        """
        return np.array([rxn.get_coeff(comp) if comp in rxn.compositions else 0 for comp in self.compositions])

    def contains_interdependent_rxns(self, precursors: list[Composition]) -> bool:
        """Whether or not the pathway contains interdependent reactions given a list of
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
            if any(set(rxn.reactants).issubset(precursors_set) for rxn in combo) or size == 1:
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

    @classmethod
    def balance(
        cls,
        reactions: list[Reaction],
        net_rxn: Reaction,
        cost_function: CostFunction,
        tol: float = 1e-6,
    ):
        """Construct a balanced pathway by automatically solving for reaction
        multiplicities, i.e., how much of each reaction is needed to yield the
        stoichiometry of the net reaction. This logic has been simplified and adapted
        from PathwaySolver.
        """

        def get_entry_idx_vector(rxn, entry_idx_dict):
            """Reproduced here so entry.data property not reset."""
            n = len(entry_idx_dict)
            indices = [entry_idx_dict[e] for e in rxn.entries]
            v = np.zeros(n)
            v[indices] = rxn.coefficients
            return v

        rxn_set = ReactionSet.from_rxns(reactions)
        entry_idxs = {entry: i for i, entry in enumerate(rxn_set.entries)}

        comp_matrix = np.vstack(
            [get_entry_idx_vector(r, entry_idxs) for r in reactions]
        )
        net_coeffs = get_entry_idx_vector(net_rxn, entry_idxs)

        comp_pinv = np.linalg.pinv(comp_matrix).T
        multiplicities = comp_pinv @ net_coeffs
        solved_coeffs = comp_matrix.T @ multiplicities

        balanced = True

        if (multiplicities < tol).any():
            warnings.warn("A reaction must be removed to balance!")

        if not (
            np.abs(solved_coeffs - net_coeffs) <= (1e-08 + 1e-05 * np.abs(net_coeffs))
        ).all():
            balanced = False
            warnings.warn("A balanced pathway cannot be found!")

        costs = [cost_function.evaluate(rxn) for rxn in reactions]
        return cls(
            reactions=reactions,
            coefficients=multiplicities,
            costs=costs,
            balanced=balanced,
        )

    @cached_property
    def net_rxn(self) -> Reaction:
        """Returns the net reaction of the pathway"""
        return self.reactions[0]

    @property
    def average_cost(self) -> float:
        """Returns the mean cost of the pathway."""
        return np.dot(self.coefficients, self.costs) / sum(self.coefficients)

    @property
    def all_pairwise_interfaces(self) -> list[tuple[Composition]]:
        pass

    @property
    def all_reacting_pairwise_interfaces(self) -> list[tuple[Composition]]:
        pass

    @property
    def all_nonreacting_pairwise_interfaces(self) -> list[tuple[Composition]]:
        pass

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            return np.allclose(self.costs, other.costs)

        return False

    def __hash__(self):
        return hash((tuple(self.reactions), tuple(self.coefficients)))

    def __repr__(self) -> str:
        path_info = ""
        for rxn in self.reactions:
            path_info += f"{rxn} (dG = {round(rxn.energy_per_atom, 3)} eV/atom) \n"

        path_info += f"Average Cost: {round(self.average_cost,3)}"

        return path_info
