"""
This module implements two types of basic (combinatorial) reaction enumerators.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
from math import comb
from typing import TYPE_CHECKING

import ray
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from tqdm import tqdm

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.entries.utils import initialize_entry
from rxn_network.enumerators.base import Enumerator
from rxn_network.enumerators.utils import group_by_chemsys
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger, grouper, limited_powerset
from rxn_network.utils.ray import initialize_ray, to_iterator

if TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedEntry

logger = get_logger(__name__)


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a maximum
    reactant/product cardinality (n); i.e., how many phases on either side of the
    reaction. This approach does not explicitly take into account thermodynamic
    stability (i.e. phase diagram). This allows for enumeration of reactions where the
    products may not be stable with respect to each other.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.
    """

    MIN_CHUNK_SIZE = 2500
    MAX_NUM_JOBS = 5000

    def __init__(
        self,
        precursors: list[str] | None = None,
        targets: list[str] | None = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_duplicates: bool = False,
        filter_by_chemsys: str | None = None,
        chunk_size: int = MIN_CHUNK_SIZE,
        max_num_jobs: int = MAX_NUM_JOBS,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        max_num_constraints: int = 1,
        quiet: bool = False,
    ):
        """

        Args:
            precursors: Optional list of precursor formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions
                as reactants. The "exclusive_precursors" parameter allows one to tune
                whether the enumerated reactions should include ALL precursors (the
                default) or just one.
            targets: Optional list of target formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions
                as products. The "exclusive_targets" parameter allows one to tune
                whether the enumerated reactions should include ALL targets or just one
                (the default).
            n: Maximum reactant/product cardinality. This it the largest possible number
                of entries on either side of the reaction. Defaults to 2.
            exclusive_precursors: Whether enumerated reactions are required to have
                reactants that are a subset of the provided list of precursors. If True
                (the default), this only identifies reactions with reactants selected
                from the provided precursors.
            exclusive_targets: Whether enumerated reactions are required to have
                products that are a subset of the provided list of targets. If False,
                (the default), this identifies all reactions containing at least one
                composition from the provided list of targets (and any number of
                byproducts).
            filter_duplicates: Whether to remove duplicate reactions. Defaults to False.
            filter_by_chemsys: An optional chemical system for which to filter produced
                reactions by. This ensures that all output reactions contain at least
                one element within the provided system.
            chunk_size: The minimum number of reactions per chunk procssed. Needs to be
                sufficiently large to make parallelization a cost-effective strategy.
                Defaults to MIN_CHUNK_SIZE.
            max_num_jobs: The upper limit for the number of jobs created. Defaults to
                MAX_NUM_JOBS.
            remove_unbalanced: Whether to remove reactions which are unbalanced; this is
                usually advisable. Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. This is also
                advisable for ensuring that only unique reaction sets are produced.
                Defaults to True.
            max_num_constraints: The maximum number of allowable constraints enforced by
                reaction balancing. Defaults to 1 (which is usually advisable).
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """

        super().__init__(precursors=precursors, targets=targets)

        self.n = n
        self.exclusive_precursors = exclusive_precursors
        self.exclusive_targets = exclusive_targets
        self.filter_duplicates = filter_duplicates
        self.filter_by_chemsys = filter_by_chemsys
        self.chunk_size = chunk_size
        self.max_num_jobs = max_num_jobs
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed
        self.max_num_constraints = max_num_constraints
        self.quiet = quiet

        self._stabilize = False

        self._p_set_func = "issuperset" if self.exclusive_precursors else "intersection"
        self._t_set_func = "issuperset" if self.exclusive_targets else "intersection"

        self.open_phases: list | None = None
        self._build_pd = False
        self._build_grand_pd = False

    def enumerate(
        self,
        entries: GibbsEntrySet,
    ) -> ReactionSet:
        """
        Calculate all possible reactions given a set of entries. If the enumerator was
        initialized with specified precursors or target, the reactions will be filtered
        by these constraints. Every enumerator follows a standard procedure:

        1) Initialize entries, i.e., ensure that precursors and target are considered
            stable entries within the entry set.

        2) Get a dictionary representing every possible "node", i.e. phase combination,
            grouped by chemical system.

        3) Filter the combos dictionary for chemical systems which are not relevant;
            i.e., don't contain elements in precursors and/or target.

        4) Iterate through each chemical system, initializing calculators,
            and computing all possible reactions for reactant/product pair and/or
            thermodynamically predicted reactions for given reactants.

        5) Add reactions to growing list, repeat Step 4 until combos dict exhausted.

        Args:
            entries: the set of all entries to enumerate from
        """
        if not ray.is_initialized():
            initialize_ray()

        entries, precursors, targets, open_entries = self._get_initialized_entries(
            entries
        )

        combos_dict = self._get_combos_dict(
            entries,
            precursors,
            targets,
            open_entries,
        )
        open_combos = self._get_open_combos(open_entries)

        if not open_combos:
            open_combos = []

        precursors = ray.put(precursors)
        targets = ray.put(targets)
        react_function = ray.put(self._react_function)
        open_entries = ray.put(open_entries)
        p_set_func = ray.put(self._p_set_func)
        t_set_func = ray.put(self._t_set_func)
        remove_unbalanced = ray.put(self.remove_unbalanced)
        remove_changed = ray.put(self.remove_changed)
        max_num_constraints = ray.put(self.max_num_constraints)

        entries_ref = ray.put(entries)

        num_cpus = int(ray.cluster_resources()["CPU"])

        pd_dict = {}
        if self._build_pd or self._build_grand_pd:
            # pre-loop for phase diagram construction
            pd_chunk_size = int(len(combos_dict) // num_cpus) + 1

            pd_dict_refs = []
            for item_chunk in grouper(combos_dict.items(), pd_chunk_size):
                pd_dict_refs.append(
                    _get_entries_and_pds.remote(
                        item_chunk,
                        entries_ref,
                        self.build_pd,
                        self.build_grand_pd,
                        getattr(self, "chempots", None),
                    )
                )

            for completed in tqdm(
                to_iterator(pd_dict_refs),
                total=len(pd_dict_refs),
                disable=self.quiet,
                desc=f"Building phase diagrams ({self.__class__.__name__})",
            ):
                pd_dict.update(completed)

        chunk_size = self.chunk_size
        total = sum(self._rxn_iter_length(c, open_combos) for c in combos_dict.values())

        if total / chunk_size > self.max_num_jobs:
            chunk_size = int(total // self.max_num_jobs) + 1
            logger.info(
                f"Increasing chunk size to {chunk_size} due to max job limit of"
                f" {self.max_num_jobs}"
            )

        to_run, current_chunk = [], []  # type: ignore
        for item in tqdm(
            combos_dict.items(),
            disable=self.quiet,
            desc="Building chunks...",
            total=len(combos_dict),
        ):
            chemsys, combos = item
            rxn_iter = list(self._get_rxn_iterable(combos, open_combos))

            filtered_entries, pd, grand_pd = None, None, None
            if self._build_pd or self._build_grand_pd:
                filtered_entries, pd, grand_pd = pd_dict[chemsys]

            filtered_entries = ray.put(filtered_entries)
            pd = ray.put(pd)
            grand_pd = ray.put(grand_pd)

            current_chunk_length = sum(len(c[0]) for c in current_chunk)  # type: ignore

            current_chunk.append(([], filtered_entries, pd, grand_pd))
            for r in rxn_iter:
                if current_chunk_length == chunk_size:
                    to_run.append(current_chunk)
                    current_chunk = [([r], filtered_entries, pd, grand_pd)]
                    current_chunk_length = 1
                else:
                    current_chunk[-1][0].append(r)
                    current_chunk_length += 1

            if current_chunk_length == chunk_size:
                to_run.append(current_chunk)
                current_chunk = []

        if current_chunk:
            to_run.append(current_chunk)

        rxn_chunk_refs, results = [], []  # type: ignore
        for chunk in to_run:
            rxn_chunk_refs.append(
                _react.remote(
                    chunk,
                    react_function,
                    open_entries,
                    precursors,
                    targets,
                    p_set_func,
                    t_set_func,
                    remove_unbalanced,
                    remove_changed,
                    max_num_constraints,
                )
            )

        for completed in tqdm(
            to_iterator(rxn_chunk_refs),
            total=len(to_run),
            disable=self.quiet,
            desc=f"Enumerating reactions ({self.__class__.__name__})",
        ):
            results.extend(completed)

        rxn_set = ReactionSet.from_rxns(
            results, entries=entries, filter_duplicates=self.filter_duplicates
        )

        return rxn_set

    @classmethod
    def _get_num_chunks(cls, items, open_combos, chunk_size) -> int:
        _ = open_combos  # not used in BasicEnumerator

        num_chunks = 0
        for _, i in items:
            num_combos = cls._rxn_iter_length(i, open_combos)
            num_chunks += num_combos // chunk_size + bool(num_combos % chunk_size)

        return num_chunks

    @staticmethod
    def _rxn_iter_length(combos, open_combos) -> int:
        _ = open_combos  # not used in BasicEnumerator
        return comb(len(combos), 2)

    def _get_combos_dict(
        self, entries, precursor_entries, target_entries, open_entries
    ):
        """
        Gets all possible entry combinations up to predefined cardinality (n), filtered
        and grouped by chemical system.
        """
        precursor_elems = [
            [str(el) for el in e.composition.elements] for e in precursor_entries
        ]
        target_elems = [
            [str(el) for el in e.composition.elements] for e in target_entries
        ]
        all_open_elems = {el for e in open_entries for el in e.composition.elements}

        entries = entries - open_entries

        combos = [set(c) for c in limited_powerset(entries, self.n)]
        combos_dict = group_by_chemsys(combos, all_open_elems)

        filtered_combos = self._filter_dict_by_elems(
            combos_dict,
            precursor_elems,
            target_elems,
            all_open_elems,
        )

        return filtered_combos

    def _get_open_combos(  # pylint: disable=useless-return
        self, open_entries
    ) -> list[set[ComputedEntry]] | None:
        """No open entries for BasicEnumerator, returns None"""
        _ = (self, open_entries)  # unused
        return None

    @staticmethod
    def _react_function(reactants, products, **kwargs):
        _ = kwargs  # unused
        forward_rxn = ComputedReaction.balance(reactants, products)
        backward_rxn = forward_rxn.reverse()
        return [forward_rxn, backward_rxn]

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Get all reaction/product combinations"""
        _ = open_combos  # unused

        return combinations(combos, 2)

    def _get_initialized_entries(self, entries):
        """Returns initialized entries, precursors, target, and open entries"""

        def initialize_entries_list(ents):
            new_ents = {initialize_entry(f, entries, self.stabilize) for f in ents}
            return new_ents

        precursors, targets = set(), set()

        entries_new = GibbsEntrySet(
            deepcopy(entries),
        )

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
                except KeyError:
                    pass

                entries_new.add(e)

        if self.stabilize:
            entries_new = entries_new.filter_by_stability(e_above_hull=0.0)
            logger.info("Filtering by stable entries!")

        entries_new.build_indices()

        open_entries = set()
        if self.open_phases:
            open_entries = {
                e
                for e in entries_new
                if e.composition.reduced_formula in self.open_phases
            }

        return entries_new, precursors, targets, open_entries

    def _filter_dict_by_elems(
        self,
        combos_dict,
        precursor_elems,
        target_elems,
        all_open_elems,
    ):
        """Filters the dictionary of combinations by elements"""
        filtered_dict = {}

        all_precursor_elems = {el for g in precursor_elems for el in g}
        all_target_elems = {el for g in target_elems for el in g}
        all_open_elems = {str(el) for el in all_open_elems}

        filter_elems = None
        if self.filter_by_chemsys:
            filter_elems = set(self.filter_by_chemsys.split("-"))

        for chemsys, combos in combos_dict.items():
            elems = set(chemsys.split("-"))

            if filter_elems:
                if not elems.issuperset(filter_elems):
                    continue

            if len(elems) >= 10 or len(elems) == 1:  # too few or too many elements
                continue

            if precursor_elems:
                if not getattr(all_precursor_elems | all_open_elems, self._p_set_func)(
                    elems
                ):
                    continue

            if target_elems:
                if not getattr(all_target_elems | all_open_elems, self._t_set_func)(
                    elems
                ):
                    continue

            filtered_dict[chemsys] = combos

        return filtered_dict

    @property
    def stabilize(self) -> bool:
        """Whether or not to use only stable entries in analysis"""
        return self._stabilize

    @property
    def build_pd(self) -> bool:
        """Whether or not to build a PhaseDiagram object during reaction enumeration (
        useful for some analyses)"""
        return self._build_pd

    @property
    def build_grand_pd(self) -> bool:
        """Whether or not to build a GrandPotentialPhaseDiagram object during
        reaction enumeration (useful for some analyses)"""
        return self._build_grand_pd


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases. Note:
    this does not return OpenComputedReaction objects (this can be calculated using
    the ReactionSet class).

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.
    """

    MIN_CHUNK_SIZE = 2500
    MAX_NUM_JOBS = 5000

    def __init__(
        self,
        open_phases: list[str],
        precursors: list[str] | None = None,
        targets: list[str] | None = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_duplicates: bool = False,
        filter_by_chemsys: str | None = None,
        chunk_size: int = MIN_CHUNK_SIZE,
        max_num_jobs: int = MAX_NUM_JOBS,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        max_num_constraints: int = 1,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            open_phases: List of formulas of open entries (e.g. ["O2"]).
            precursors: Optional list of precursor formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions
                as reactants. The "exclusive_precursors" parameter allows one to tune
                whether the enumerated reactions should include ALL precursors (the
                default) or just one.
            targets: Optional list of target formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions
                as products. The "exclusive_targets" parameter allows one to tune
                whether the enumerated reactions should include ALL targets or just one
                (the default).
            n: Maximum reactant/product cardinality. This it the largest possible number
                of entries on either side of the reaction. Defaults to 2.
            exclusive_precursors: Whether enumerated reactions are required to have
                reactants that are a subset of the provided list of precursors. If True
                (the default), this only identifies reactions with reactants selected
                from the provided precursors.
            exclusive_targets: Whether enumerated reactions are required to have
                products that are a subset of the provided list of targets. If False,
                (the default), this identifies all reactions containing at least one
                composition from the provided list of targets (and any number of
                byproducts).
            filter_duplicates: Whether to remove duplicate reactions. Defaults to False.
            filter_by_chemsys: An optional chemical system for which to filter produced
                reactions by. This ensures that all output reactions contain at least
                one element within the provided system.
            chunk_size: The minimum number of reactions per chunk procssed. Needs to be
                sufficiently large to make parallelization a cost-effective strategy.
                Defaults to MIN_CHUNK_SIZE.
            max_num_jobs: The upper limit for the number of jobs created. Defaults to
                MAX_NUM_JOBS.
            remove_unbalanced: Whether to remove reactions which are unbalanced; this is
                usually advisable. Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. This is also
                advisable for ensuring that only unique reaction sets are produced.
                Defaults to True.
            max_num_constraints: The maximum number of allowable constraints enforced by
                reaction balancing. Defaults to 1 (which is usually advisable).
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
            n=n,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            filter_duplicates=filter_duplicates,
            filter_by_chemsys=filter_by_chemsys,
            chunk_size=chunk_size,
            max_num_jobs=max_num_jobs,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
            max_num_constraints=max_num_constraints,
            quiet=quiet,
        )
        self.open_phases: list[str] = open_phases

    @staticmethod
    def _rxn_iter_length(combos, open_combos):
        num_combos_with_open = sum(
            1 if not i & j else 0 for i in combos for j in open_combos
        )

        return len(combos) * num_combos_with_open

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


@ray.remote
def _react(
    chunk,
    react_function,
    open_entries,
    precursors,
    targets,
    p_set_func,
    t_set_func,
    remove_unbalanced,
    remove_changed,
    max_num_constraints,
):
    """
    This function is a wrapper for the specific react function of each enumerator. This
    wrapper contains the logic used for filtering out reactions based on the
    user-defined enumerator settings. It can also be called as a remote function using
    ray, allowing for parallel computation during reaction enumeration.

    WARNING: this function is not intended to to be called directly by the user and
    should only be used by the enumerator classes.

    """
    all_rxns = []
    for rxn_iterable, filtered_entries, pd, grand_pd in chunk:
        filtered_entries = ray.get(filtered_entries)
        pd = ray.get(pd)
        grand_pd = ray.get(grand_pd)

        for rp in rxn_iterable:
            if not rp:
                continue

            r = set(rp[0]) if rp[0] else set()
            p = set(rp[1]) if rp[1] else set()

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

            if not (precursor_func(r) or (p and precursor_func(p))):
                continue
            if p and not (target_func(r) or target_func(p)):
                continue

            suggested_rxns = react_function(
                r, p, filtered_entries=filtered_entries, pd=pd, grand_pd=grand_pd
            )

            rxns = []
            for rxn in suggested_rxns:
                if (
                    rxn.is_identity
                    or (remove_unbalanced and not rxn.balanced)
                    or (remove_changed and rxn.lowest_num_errors != 0)
                    or rxn.data["num_constraints"] > max_num_constraints
                ):
                    continue

                reactant_entries = set(rxn.reactant_entries) - open_entries
                product_entries = set(rxn.product_entries) - open_entries

                if precursor_func(reactant_entries) and target_func(product_entries):
                    rxns.append(rxn)

            all_rxns.extend(rxns)

    return all_rxns


@ray.remote
def _get_entries_and_pds(
    combos_dict_chunk, entries, build_pd, build_grand_pd, chempots
):
    pd_dict = {}
    for item in combos_dict_chunk:
        if item is None:
            continue
        chemsys, _ = item
        elems = chemsys.split("-")

        filtered_entries = None
        pd = None
        grand_pd = None

        if build_pd or build_grand_pd:
            filtered_entries = entries.get_subset_in_chemsys(elems)

        if build_pd:
            pd = PhaseDiagram(filtered_entries)

        if build_grand_pd:
            grand_pd = GrandPotentialPhaseDiagram(filtered_entries, chempots)

        pd_dict[chemsys] = (filtered_entries, pd, grand_pd)
    return pd_dict
