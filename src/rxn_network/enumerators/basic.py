from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np

from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import (
    get_total_chemsys,
    group_by_chemsys,
)
from rxn_network.utils import limited_powerset


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n).
    """

    def __init__(self, n):
        self.n = n

    def enumerate(
        self,
        entries: List[ComputedEntry],
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
    ) -> List[Reaction]:
        """
        Calculate all possible reactions given a list of entries.

        Args:
            entries: A list of all entries to consider
            remove_unbalanced: Whether to remove reactions which are unbalanced
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides

        Returns:
            List of reactions.
        """

        combos = list(limited_powerset(entries, self.n))
        combos_dict = group_by_chemsys(combos)

        rxns = []
        for chemsys, selected_combos in combos_dict.items():
            for reactants, products in combinations(selected_combos, 2):
                forward_rxn, backward_rxn = self._get_rxns(
                    reactants, products, remove_unbalanced, remove_changed
                )
                if forward_rxn:
                    rxns.append(forward_rxn)
                    rxns.append(backward_rxn)

        return list(set(rxns))

    def estimate_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return sum([comb(len(entries), i) for i in range(self.n)]) ** 2

    def _get_rxns(self, reactants, products, remove_unbalanced, remove_changed):
        forward_rxn = ComputedReaction.balance(reactants, products)
        if (remove_unbalanced and not (forward_rxn.balanced)) or (
            remove_changed and forward_rxn.lowest_num_errors != 0
        ):
            forward_rxn = None
            backward_rxn = None
        else:
            backward_rxn = ComputedReaction(
                forward_rxn.entries, forward_rxn.coefficients * -1
            )
        return forward_rxn, backward_rxn


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases.
    """

    def __init__(self, n, open_entries):
        super().__init__(n)
        self.open_entries = open_entries

    def enumerate(self, entries, remove_unbalanced=True, remove_changed=True):
        combos = [set(c) for c in limited_powerset(entries, self.n)]
        open_combos = [
            set(c) for c in limited_powerset(self.open_entries, len(self.open_entries))
        ]
        combos_with_open = [
            combo | open_combo
            for combo in combos
            for open_combo in open_combos
            if not combo & open_combo
        ]
        combos_dict = group_by_chemsys(combos)
        combos_open_dict = group_by_chemsys(combos_with_open)

        rxns = []
        for chemsys, selected_combos in combos_dict.items():
            if chemsys not in combos_open_dict:
                continue
            selected_open_combos = combos_open_dict[chemsys]
            for reactants, products in product(selected_combos, selected_open_combos):
                forward_rxn, backward_rxn = self._get_rxns(
                    reactants, products, remove_unbalanced, remove_changed
                )
                if forward_rxn:
                    rxns.append(forward_rxn)
                    rxns.append(backward_rxn)

        closed_entries = [e for e in entries if e not in self.open_entries]
        simple_rxns = super().enumerate(
            closed_entries, remove_unbalanced, remove_changed
        )

        return rxns + simple_rxns
