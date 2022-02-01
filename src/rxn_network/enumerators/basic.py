"""
This module implements two types of basic reaction enumerators, differing in the option
to consider open entries.
"""
from copy import deepcopy
from itertools import combinations, product
from math import comb
from typing import List, Optional, Set

from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from tqdm.auto import tqdm

from rxn_network.core import Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import (
    apply_calculators,
    group_by_chemsys,
    initialize_calculators,
    initialize_entry,
)
from rxn_network.reactions import ComputedReaction
from rxn_network.utils import limited_powerset


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n); i.e., how many phases on either side of
    the reaction. This approach does not explicitly take into account thermodynamic
    stability (i.e. phase diagram). This allows for enumeration of reactions where the
    products may not be stable with respect to each other.
    """

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        calculators: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            precursors: Optional list of precursor formulas; only reactions
                which contain at least these phases as reactants will be enumerated. See the
                "exclusive_precursors" parameter for more details.
            targets: Optional list of target formulas; only reactions which include
                formation of at least one of these targets will be enumerated. See the
                "exclusive_targets" parameter for more details.
            calculators: Optional list of Calculator object names to be initialized; see calculators
                module for options (e.g., ["ChempotDistanceCalculator"])
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction. Defaults to 2.
            exclusive_precursors: Whether to consider only reactions that have
            reactants which are a subset of the provided list of precursors. In other
                words, if True, this only identifies reactions with reactants selected from the precursors
                argument. Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                form products that are a subset of the provided list of targets. If False,
                this only identifies reactions with no unspecified byproducts. Defualts to False.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors, targets=targets, calculators=calculators
        )

        self.n = n
        self.exclusive_precursors = exclusive_precursors
        self.exclusive_targets = exclusive_targets
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed
        self.quiet = quiet

        self._stabilize = False
        if "ChempotDistanceCalculator" in self.calculators:
            self._stabilize = True  # ChempotDistanceCalculator requires stable entries

        self.open_phases: Optional[List] = None
        self._build_pd = False
        self._build_grand_pd = False

    def enumerate(self, entries: GibbsEntrySet) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with specified precursors or target, the reactions will be
        filtered by these constraints. Every enumerator follows a standard procedure:

        1. Initialize entries, i.e. ensure that precursors and target are considered
        stable entries within the entry set. If using ChempotDistanceCalculator,
        ensure that entries are filtered by stability.

        2. Get a dictionary representing every possible "node", i.e. phase combination,
        grouped by chemical system.

        3. Filter the combos dictionary for chemical systems which are not relevant;
        i.e., don't contain elements in precursors and/or target.

        4. Iterate through each chemical system, initializing calculators,
        and computing all possible reactions for reactant/product pair and/or
        thermodynamically predicted reactions for given reactants.

        5. Add reactions to growing list, repeat Step 4 until combos dict exhausted.

        Args:
            entries: the set of all entries to enumerate from
        """
        entries, precursors, targets, open_entries = self._get_initialized_entries(
            entries
        )

        combos_dict = self._get_combos_dict(entries, precursors, targets, open_entries)
        open_combos = self._get_open_combos(open_entries)

        pbar = tqdm(
            combos_dict.items(), desc=self.__class__.__name__, disable=self.quiet
        )

        rxns = []
        for chemsys, combos in pbar:
            pbar.set_description(f"{chemsys}")

            elems = chemsys.split("-")

            filtered_entries = entries.get_subset_in_chemsys(elems)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            rxn_iter = self._get_rxn_iterable(combos, open_combos)

            r = self._get_rxns(
                rxn_iter,
                precursors,
                targets,
                calculators,
                filtered_entries,
                open_entries,
            )
            rxns.extend(r)

        return list(set(rxns))

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return sum([comb(len(entries), i) for i in range(1, self.n + 1)]) ** 2

    def _get_combos_dict(
        self, entries, precursor_entries, target_entries, open_entries
    ):
        """
        Gets all possible entry combinations up to predefined cardinality n, filtered and
        grouped by chemical system
        """
        precursor_elems = [
            [str(el) for el in e.composition.elements] for e in precursor_entries
        ]
        target_elems = [
            [str(el) for el in e.composition.elements] for e in target_entries
        ]
        open_elems = [[str(el) for el in e.composition.elements] for e in open_entries]

        entries = entries - open_entries

        combos = [set(c) for c in limited_powerset(entries, self.n)]
        combos_dict = group_by_chemsys(combos)

        filtered_combos = self._filter_dict_by_elems(
            combos_dict, precursor_elems, target_elems, open_elems
        )

        return filtered_combos

    def _get_open_combos(  # pylint: disable=R1711
        self, open_entries
    ) -> Optional[List[Set[ComputedEntry]]]:
        """No open entries for BasicEnumerator, returns None"""
        _ = (self, open_entries)  # unused_arguments
        return None

    def _get_rxns(
        self,
        rxn_iterable,
        precursors,
        targets,
        calculators,
        filtered_entries=None,
        open_entries=None,
    ):
        """
        Returns reactions from an iterable representing the combination of 2 sets
        of phases. Works for reactions with open phases.
        """
        pd = None
        if self.build_pd:
            pd = PhaseDiagram(filtered_entries)

        grand_pd = None
        if self.build_grand_pd:
            chempots = getattr(self, "chempots")
            grand_pd = GrandPotentialPhaseDiagram(filtered_entries, chempots)

        p_set_func = "issuperset" if self.exclusive_precursors else "intersection"
        t_set_func = "issuperset" if self.exclusive_targets else "intersection"

        if not open_entries:
            open_entries = set()

        rxns = []

        for reactants, products in rxn_iterable:
            r = set(reactants) if reactants else set()
            p = set(products) if products else set()
            all_phases = r | p

            precursor_func = (
                getattr(precursors | open_entries, p_set_func)
                if precursors
                else lambda e: True
            )
            target_func = (
                getattr(targets | open_entries, t_set_func)
                if targets
                else lambda e: True
            )

            if (
                (r & p)
                or (precursors and not precursors & all_phases)
                or (p and targets and not targets & all_phases)
            ):
                continue

            if not (precursor_func(r) or precursor_func(p)):
                continue
            if p and not (target_func(r) or target_func(p)):
                continue

            suggested_rxns = self._react(r, p, calculators, pd, grand_pd)

            for rxn in suggested_rxns:
                if (
                    rxn.is_identity
                    or (self.remove_unbalanced and not rxn.balanced)
                    or (self.remove_changed and rxn.lowest_num_errors != 0)
                ):
                    continue

                reactant_entries = set(rxn.reactant_entries) - open_entries
                product_entries = set(rxn.product_entries) - open_entries

                if precursor_func(reactant_entries) and target_func(product_entries):
                    rxns.append(rxn)

        return rxns

    def _react(self, reactants, products, calculators, pd=None, grand_pd=None):
        """
        Generates reactions from a list of reactants, products, and optional
        calculator(s)
        """
        _ = (pd, grand_pd, self)  # unused arguments in BasicEnumerator class

        forward_rxn = ComputedReaction.balance(reactants, products)
        backward_rxn = forward_rxn.reverse()

        forward_rxn = apply_calculators(forward_rxn, calculators)
        backward_rxn = apply_calculators(backward_rxn, calculators)

        return [forward_rxn, backward_rxn]

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Get all reaction/product combinations"""
        _ = open_combos  # unused argument in BasicEnumerator class

        return combinations(combos, 2)

    def _get_initialized_entries(self, entries):
        """Returns initialized entries, precursors, target, and open entries"""
        precursors, targets = set(), set()
        entries_new = deepcopy(entries)

        def initialize_entries_list(ents):
            new_ents = {initialize_entry(f, entries, self.stabilize) for f in ents}
            return new_ents

        if self.precursors:
            precursors = initialize_entries_list(self.precursors)

        if self.targets:
            targets = initialize_entries_list(self.targets)

        for e in precursors | targets:
            if e not in entries_new:
                try:
                    old_e = entries_new.get_min_entry_by_formula(
                        e.composition.reduced_formula
                    )
                    entries_new.discard(old_e)
                except IndexError:
                    pass

                entries_new.add(e)

        if self.stabilize:
            entries_new = entries_new.filter_by_stability(e_above_hull=0.0)
            self.logger.info("Filtering by stable entries!")

        open_entries = set()
        if self.open_phases:
            open_entries = {
                e for e in entries if e.composition.reduced_formula in self.open_phases
            }

        return entries_new, precursors, targets, open_entries

    def _filter_dict_by_elems(
        self,
        combos_dict,
        precursor_elems,
        target_elems,
        open_elems,
    ):
        """Filters the dictionary of combinations by elements"""
        filtered_dict = {}

        all_precursor_elems = {el for g in precursor_elems for el in g}
        all_target_elems = {el for g in target_elems for el in g}

        for chemsys, combos in combos_dict.items():
            elems = set(chemsys.split("-"))

            if len(elems) >= 10 or len(elems) == 1:
                continue

            all_open_elems = {el for g in open_elems for el in g}
            if not all_open_elems.issubset(elems):
                continue

            if precursor_elems:
                if self.exclusive_precursors:
                    if not all_precursor_elems == elems:
                        continue
                else:
                    if not any(
                        elems.issuperset(el_group) for el_group in precursor_elems
                    ):
                        continue

            if target_elems:
                if self.exclusive_targets:
                    if not all_target_elems == elems:
                        continue
                else:
                    if not any(elems.issuperset(el_group) for el_group in target_elems):
                        continue

            filtered_dict[chemsys] = combos

        return filtered_dict

    @property
    def stabilize(self):
        """Whether or not to use only stable entries in analysis"""
        return self._stabilize

    @property
    def build_pd(self):
        """Whether or not to build a PhaseDiagram object during reaction enumeration (
        useful for some analyses)"""
        return self._build_pd

    @property
    def build_grand_pd(self):
        """Whether or not to build a GrandPotentialPhaseDiagram object during
        reaction enumeration (useful for some analyses)"""
        return self._build_grand_pd


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases. Note:
    this does not return OpenComputedReaction objects (this can be calculated using
    the ReactionSet class).
    """

    def __init__(
        self,
        open_phases: List[str],
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        calculators: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            open_phases: List of formulas of open entries (e.g. ["O2"])
            precursors: Optional list of formulas of precursor phases; only reactions
                which have these phases as reactants will be enumerated.
            targets: Optional list of formulas of targets; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator]).
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction.
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
            calculators=calculators,
            n=n,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
            quiet=quiet,
        )
        self.open_phases: List[str] = open_phases

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        num_open_phases = len(self.open_phases)
        num_combos_with_open = sum(
            [comb(num_open_phases, i) for i in range(1, num_open_phases + 1)]
        )

        num_total_combos = 0
        for i in range(1, self.n + 1):
            num_combos = comb(len(entries), i)
            num_total_combos += num_combos_with_open * num_combos

        return num_total_combos ** 2

    def _get_open_combos(self, open_entries):
        """Get all possible combinations of open entries. For a single entry,
        this is just the entry itself."""
        open_combos = [
            set(c) for c in limited_powerset(open_entries, len(open_entries))
        ]
        return open_combos

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Get all reaction/product combinations."""
        combos_with_open = [
            combo | open_combo
            for combo in combos
            for open_combo in open_combos
            if not combo & open_combo
        ]
        rxn_iter = product(combos, combos_with_open)

        return rxn_iter
