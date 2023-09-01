"""
Implements a class for storing (unbalanced/unconstrained) collection of reactions
forming a reaction pathway.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rxn_network.pathways.base import Pathway

if TYPE_CHECKING:
    from rxn_network.reactions.base import Reaction


class BasicPathway(Pathway):
    """
    Simple pathway class for storing multiple ComputedReaction objects which form a
    single reaction pathway.

    In this class, there are no constraints on stoichiometry. See BalancedPathway for
    the more constrained version.
    """

    def __init__(self, reactions: list[Reaction], costs: list[float] | None = None):
        """
        Args:
            reactions: list of ComputedReaction objects making up the pathway.
            costs: Optional list of corresponding costs for each reaction.
        """
        self._reactions = reactions
        self.costs = costs or []

    @property
    def reactions(self) -> list[Reaction]:
        """A list of reactions contained in the reaction pathway"""
        return self._reactions

    @property
    def total_cost(self) -> float:
        """The sum of all costs associated with reactions in the pathway"""
        return sum(self.costs)

    @property
    def is_experimental(self) -> bool:
        """Whether or not all reactions in the pathway are experimental"""
        return all(e.is_experimental for e in self.entries)

    def __repr__(self) -> str:
        path_info = ""
        for rxn in self.reactions:
            path_info += f"{rxn} (dG = {round(rxn.energy_per_atom, 3)} eV/atom) \n"

        path_info += f"Total Cost: {round(self.total_cost,3)}"

        return path_info

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return all(
                other_rxn == rxn
                for other_rxn, rxn in zip(other.reactions, self.reactions)
            )

        return False

    def __hash__(self):
        return hash(tuple(self.reactions))
