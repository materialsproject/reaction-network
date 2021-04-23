from typing import List, Optional
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from tqdm.auto import tqdm

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core import Enumerator, Reaction, Calculator
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import (
    initialize_entry,
    initialize_open_entries,
    initialize_calculators,
    apply_calculators,
    filter_entries_by_chemsys,
    get_total_chemsys,
    group_by_chemsys,
)

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.thermo.chempot_diagram import ChempotDiagram
from rxn_network.utils import limited_powerset


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant and product cardinality (n).
    """

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[str]] = None,
        n: int = 2,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            precursors: Optional collection of precursor formulas; only reactions
                with these phases as reactants will be enumerated.
            target: Optional formula of target; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator])
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction. Defaults to 2.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
        """
        super().__init__(precursors, target, calculators)

        self.n = n
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed

    def enumerate(
        self,
        entries: GibbsEntrySet,
    ) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with precursors/target, the reactions will be filtered by
        these constraints.

        Args:
            entries: the set of all entries to enumerate from

        Returns:
            List of unique computed reactions.
        """
        entries = GibbsEntrySet(entries)

        target = None
        if self.target:
            target = initialize_entry(self.target, entries)
            entries.add(target)
            target_elems = {str(elem) for elem in target.composition.elements}

        precursors = None
        if self.precursors:
            precursors = {initialize_entry(f, entries) for f in self.precursors}
            for p in precursors:
                entries.add(p)
            precursor_elems = {
                str(elem) for p in precursors for elem in p.composition.elements
            }

        if "ChempotDistanceCalculator" in self.calculators:
            entries = entries.filter_by_stability(e_above_hull=0.0)
            self.logger.info(
                "Filtering by stable entries due to use of " "ChempotDistanceCalculator"
            )

        combos = list(limited_powerset(entries, self.n))
        combos_dict = group_by_chemsys(combos)

        rxns = []
        for chemsys, selected_combos in tqdm(combos_dict.items()):
            elems = chemsys.split("-")
            if (
                (target and not target_elems.issubset(elems))
                or (precursors and not precursor_elems.issuperset(elems))
                or len(elems) >= 10
            ):
                continue

            filtered_entries = filter_entries_by_chemsys(entries, chemsys)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            rxn_iter = combinations(selected_combos, 2)
            rxns.extend(self._get_rxns(rxn_iter, precursors, target, calculators))

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

    def _get_rxns(self, rxn_iter, precursors, target, calculators, open=[]):
        rxns = []
        open = set(open)
        for reactants, products in rxn_iter:
            r = set(reactants)
            p = set(products)
            all_phases = r | p | open

            if r & p:  # do not allow repeated phases
                continue
            if target and target not in all_phases:
                continue
            if precursors and not precursors.issubset(all_phases):
                continue

            forward_rxn = ComputedReaction.balance(r, p)

            if (self.remove_unbalanced and not (forward_rxn.balanced)) or (
                self.remove_changed and forward_rxn.lowest_num_errors != 0
            ):
                forward_rxn = None
                backward_rxn = None
            else:
                backward_rxn = forward_rxn.reverse()

            if forward_rxn:
                if not target or target in p:
                    if not precursors or precursors == (r-open):
                        forward_rxn = apply_calculators(forward_rxn, calculators)
                        rxns.append(forward_rxn)
                if not target or target in r:
                    if not precursors or precursors == (p-open):
                        backward_rxn = apply_calculators(backward_rxn, calculators)
                        rxns.append(backward_rxn)

        return rxns


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases.
    """

    def __init__(
        self,
        open_entries: List[str],
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[Calculator]] = None,
        n: int = 2,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            open_entries: List of formulas of open entries (e.g. ["O2"])
            precursors:
            target: Optional formula of target; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator]).
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
        """
        super().__init__(precursors, target, calculators, n, remove_unbalanced,
                         remove_changed)
        self.open_entries = open_entries

    def enumerate(self, entries: GibbsEntrySet) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with precursors/target, only reactions containing these
        phases will be enumerated.

        Args:
            entries: the set of all entries to enumerate from

        Returns:
            List of unique computed reactions. Note: this does not return
            OpenComputedReaction objects (this can be calculated using the
            ReactionSet class).
        """
        entries = GibbsEntrySet(entries)

        target = None
        if self.target:
            target = initialize_entry(self.target, entries)
            entries.add(target)
            target_elems = {str(e) for e in target.composition.elements}
            precursors = None

        precursors = None
        if self.precursors:
            precursors = {initialize_entry(f, entries) for f in self.precursors}
            for p in precursors:
                entries.add(p)
            precursor_elems = {
                str(elem) for p in precursors for elem in p.composition.elements
            }

        if "ChempotDistanceCalculator" in self.calculators:
            entries = entries.filter_by_stability(e_above_hull=0.0)
            self.logger.info(
                "Filtering by stable entries due to use of 'ChempotDistanceCalculator'"
            )

        combos = [set(c) for c in limited_powerset(entries, self.n)]
        open_entries = initialize_open_entries(self.open_entries, entries)
        open_entry_elems = {str(elem) for entry in open_entries for elem in
                            entry.composition.elements}
        if precursors:
            precursor_elems = precursor_elems | open_entry_elems
        open_combos = [
            set(c) for c in limited_powerset(open_entries, len(open_entries))
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
        for chemsys, selected_combos in tqdm(combos_dict.items()):
            elems = chemsys.split("-")
            if chemsys not in combos_open_dict:
                continue
            if target and not target_elems.issubset(elems):
                continue

            filtered_entries = filter_entries_by_chemsys(entries, chemsys)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            if (
                (target and not target_elems.issubset(elems))
                or (precursors and not precursor_elems.issuperset(elems))
                or len(elems) >= 10
            ):
                continue

            selected_open_combos = combos_open_dict[chemsys]
            rxn_iter = product(selected_combos, selected_open_combos)

            rxns.extend(self._get_rxns(rxn_iter, precursors, target, calculators,
                                       open=open_entries))

        return list(set(rxns))
