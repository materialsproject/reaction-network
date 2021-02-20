from typing import List, Optional
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from tqdm import tqdm

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.entry_tools import EntrySet

from rxn_network.core import Enumerator, Reaction, Calculator
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import (
    get_total_chemsys,
    group_by_chemsys,
)

from rxn_network.thermo.chempot_diagram import ChempotDiagram
from rxn_network.costs.calculators import ChempotDistanceCalculator
from rxn_network.utils import limited_powerset


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (dim).
    """

    def __init__(
        self,
        n: int = 2,
        calculators: Optional[List[Calculator]]=None,
        target: Optional[ComputedEntry] = None,
    ):
        self.n = n
        self.calculators = calculators
        self.target = target

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
        entries = EntrySet(entries)

        if self.target:
            target_elems = {str(e) for e in self.target.composition.elements}

        rxns = []
        for chemsys, selected_combos in tqdm(combos_dict.items()):
            calculators = []
            #  TODO: make this calculator check/initialization less awkward
            if ChempotDistanceCalculator or "ChempotDistanceCalculator" in self.calculators:
                pd = PhaseDiagram(entries.get_subset_in_chemsys(chemsys.split("-")))
                cpd = ChempotDiagram(pd, default_limit=-50)
                calculators.append(ChempotDistanceCalculator(cpd))

            if self.target and not target_elems.issubset(chemsys.split("-")):
                continue

            rxn_iter = combinations(selected_combos, 2)
            rxns.extend(
                self._filter_and_get_rxns(
                    rxn_iter, self.target, calculators, remove_unbalanced,
                    remove_changed
                )
            )

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

    def _filter_and_get_rxns(self, rxn_iter, target, calculators, remove_unbalanced,
                             remove_changed):
        rxns = []
        for reactants, products in rxn_iter:
            r = set(reactants)
            p = set(products)

            if r & p:  # do not allow repeated phases
                continue
            if target and target not in r | p:
                continue

            forward_rxn = ComputedReaction.balance(r, p)

            if (remove_unbalanced and not (forward_rxn.balanced)) or (
                remove_changed and forward_rxn.lowest_num_errors != 0
            ):
                forward_rxn = None
                backward_rxn = None
            else:
                backward_rxn = ComputedReaction(
                    forward_rxn.entries, forward_rxn.coefficients * -1
                )

            if forward_rxn:
                if not target or target in p:
                    forward_rxn = self._apply_calculators(forward_rxn, calculators)
                    rxns.append(forward_rxn)
                if not target or target in r:
                    backward_rxn = self._apply_calculators(backward_rxn, calculators)
                    rxns.append(backward_rxn)

        return rxns

    @staticmethod
    def _apply_calculators(rxn, calculators):
        for calc in calculators:
            rxn = calc.decorate(rxn)
        return rxn


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (dim), with any number of open phases.
    """

    def __init__(
        self, n: int, open_entries: List[ComputedEntry], calculators: Optional[List[
                Calculator]]=None, target: ComputedEntry = None
    ):
        super().__init__(n, calculators, target)

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

        entries = EntrySet(entries)

        if self.target:
            target_elems = {e.symbol for e in self.target.composition.elements}

        rxns = []
        for chemsys, selected_combos in tqdm(combos_dict.items()):
            if chemsys not in combos_open_dict:
                continue
            if self.target and not target_elems.issubset(chemsys.split("-")):
                continue

            calculators = []
            #  TODO: make this calculator check/initialization less awkward
            if ChempotDistanceCalculator or "ChempotDistanceCalculator" in self.calculators:
                pd = PhaseDiagram(entries.get_subset_in_chemsys(chemsys.split("-")))
                cpd = ChempotDiagram(pd, default_limit=-50)
                calculators.append(ChempotDistanceCalculator(cpd))

            if self.target and not target_elems.issubset(chemsys.split("-")):
                continue

            selected_open_combos = combos_open_dict[chemsys]
            rxn_iter = product(selected_combos, selected_open_combos)

            rxns.extend(
                self._filter_and_get_rxns(
                    rxn_iter, self.target, calculators, remove_unbalanced,
                    remove_changed
                )
            )

        return list(set(rxns))
