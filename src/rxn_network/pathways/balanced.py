"""Implements a class for storing a balanced reaction pathway."""
from __future__ import annotations

import warnings
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING, Callable

import numpy as np

from rxn_network.pathways.basic import BasicPathway
from rxn_network.reactions.hull import InterfaceReactionHull
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

        rxn_set = ReactionSet.from_rxns(reactions + [net_rxn])
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
            warnings.warn(
                "A balanced pathway cannot be found! Setting balanced = False."
            )

        costs = [cost_function.evaluate(rxn) for rxn in reactions]
        return cls(
            reactions=reactions,
            coefficients=multiplicities,
            costs=costs,
            balanced=balanced,
        )

    def _get_tertiary_competition(self, rxn_set: ReactionSet):
        """
        Returns the tertiary competition (C3) for the pathway. This is the summation of
        the maximum driving forces for all "nonreacting" pairwise interfaces and is a
        measure of the likelihood that the pathway will deviate from its predicted
        reaction steps (i.e., due to competition among unplanned interfacial reactions).
        """
        c3 = 0
        for interface in self.all_nonreacting_pairwise_interfaces:
            reactants = [c.reduced_formula for c in interface]
            competing_rxns = list(rxn_set.get_rxns_by_reactants(reactants))

            competing_rxn_energies = [r.energy_per_atom for r in competing_rxns]
            min_energy = min(*competing_rxn_energies, 0)  # must be <= 0

            c3 += -min_energy
        return c3

    @property
    def average_cost(self) -> float:
        """Returns the mean cost of the pathway."""
        return np.dot(self.coefficients, self.costs) / sum(self.coefficients)

    @property
    def all_pairwise_interfaces(self) -> set[tuple[Composition]]:
        """Returns a list of all pairwise interfaces, given as tuples of compositions
        that may react during the pathway."""
        return set(
            tuple(sorted(interface)) for interface in combinations(self.compositions, 2)
        )

    @property
    def all_reacting_pairwise_interfaces(self) -> set[tuple[Composition]]:
        return {
            tuple(sorted(combo))
            for r in self.reactions
            for combo in combinations(r.reactants, 2)
        }

    @property
    def all_nonreacting_pairwise_interfaces(self) -> set[tuple[Composition]]:
        unique_reaction_interfaces = {
            tuple(sorted(interface))
            for r in self.reactions
            for interface in combinations(r.compositions, 2)
        }  # remove interfaces within same reaction step

        return self.all_pairwise_interfaces - unique_reaction_interfaces

    @property
    def primary_competition(self) -> float | None:
        """Aggregate primary competition (C1) value for pathway."""
        all_c1: list[float] = []
        for r in self.reactions:
            c1 = r.data.get("primary_competition")

            if c1 is None:
                warnings.warn(f"primary_competition not found for {r}!")
                continue

            all_c1.append(c1)

        if not all_c1:
            return None

        return self.aggregate_function(all_c1)

    @property
    def secondary_competition(self) -> float | None:
        """Aggregate secondary competition (C2) value for pathway."""
        all_c2: list[float] = []
        for r in self.reactions:
            c2 = r.data.get("secondary_competition")

            if c2 is None:
                warnings.warn(f"secondary_competition not found for {r}!")
                continue

            all_c2.append(c2)

        if not all_c2:
            return None

        return self.aggregate_function(all_c2)

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
