"""Implements a class for summing multiple reactions together."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Collection

import numpy as np
from monty.json import MSONable

from rxn_network.reactions.computed import ComputedReaction
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedEntry

logger = get_logger(__name__)


class ReactionSum(MSONable):
    """
    A class for summing multiple compuited reactions together. This is useful for
    describing and calculating the properties of net synthesis reactions, e.g.,
    "one-pot" synthesis reactions with 3, 4, or 5+ precursors.
    """

    def __init__(
        self,
        reactions: Collection[ComputedReaction],
        aggregate_function: Callable | str = "sum",
    ):
        """
        Args:
            reactions: A list or set of ComputedReaction objects.
            aggregate_function: Function to use to aggregate the primary (C1) and
                secondary (C2) competition data.
        """
        self._reactions = list(reactions)

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

        self._net_reaction = self._sum_reactions()

    @property
    def reactions(self) -> list[ComputedReaction]:
        """List of reactions"""
        return self._reactions

    @property
    def net_reaction(self) -> ComputedReaction:
        """The net reaction resulting from summing the provided reactions"""
        return self._net_reaction

    @property
    def reacting_interfaces(self) -> list[list[ComputedEntry]]:
        """List of all reacting interfaces"""
        pass

    @property
    def nonreacting_interfaces(self) -> list[list[ComputedEntry]]:
        """List of all nonreacting interfaces"""
        pass

    @property
    def all_interfaces(self) -> list[list[ComputedEntry]]:
        """List of all interfaces"""
        pass

    def _sum_reactions(self) -> ComputedReaction:
        """Identifies a new ComputedReaction object that is a combination of the
        provided reactions (i.e., self.reactions)"""
        entry_coeffs: dict[ComputedEntry, float] = {}
        all_c1, all_c2 = [], []
        for rxn in self.reactions:
            for entry, coeff in zip(rxn.entries, rxn.coefficients):
                if entry in entry_coeffs:
                    entry_coeffs[entry] += coeff
                else:
                    entry_coeffs[entry] = coeff

            all_c1.append(rxn.data.get("primary_competition"))
            all_c2.append(rxn.data.get("secondary_competition"))

        for r, c1, c2 in zip(self.reactions, all_c1, all_c2):
            if c1 is None:
                logger.warning(f"Reaction {r} does not have primary competition data!")
            if c2 is None:
                logger.warning(
                    f"Reaction {r} does not have secondary competition data!"
                )

        aggregate_c1 = self.aggregate_function(all_c1)
        aggregate_c2 = self.aggregate_function(all_c2)

        data = {
            "c1": aggregate_c1,
            "c2": aggregate_c2,
            "aggregate_function": self.aggregate_function.__name__,
        }

        net_rxn = ComputedReaction(
            entry_coeffs.keys(), entry_coeffs.values(), data=data
        )

        if not net_rxn.balanced:
            raise ValueError("Unexpected error: net reaction is not balanced!")

        return net_rxn
