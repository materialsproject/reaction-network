"""This module implements two types of reaction enumerators using a free energy
minimization technique, with or without the option of an open entry.
"""

from __future__ import annotations

from itertools import product

from pymatgen.core.composition import Element

from rxn_network.core import Composition
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.utils import react_interface


class MinimizeGibbsEnumerator(BasicEnumerator):
    """Enumerator for finding all reactions between two reactants that are predicted by
    thermodynamics; i.e., they appear when taking the convex hull along a straight line
    connecting any two phases in G-x phase space. Identity reactions are automatically
    excluded.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.
    """

    MIN_CHUNK_SIZE = 1000
    MAX_NUM_JOBS = 1000

    def __init__(
        self,
        precursors: list[str] | None = None,
        targets: list[str] | None = None,
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
        """Initialize a MinimizeGibbsEnumerator.

        Args:
            precursors: Optional list of precursor formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions as reactants. The
                "exclusive_precursors" parameter allows one to tune whether the enumerated reactions should include ALL
                precursors (the default) or just one.
            targets: Optional list of target formulas. The only reactions that
                will be enumerated are those featuring one or more of these compositions as products. The
                "exclusive_targets" parameter allows one to tune whether the enumerated reactions should include ALL
                targets or just one (the default).
            exclusive_precursors: Whether enumerated reactions are required to have
                reactants that are a subset of the provided list of precursors. If True (the default), this only
                identifies reactions with reactants selected from the provided precursors.
            exclusive_targets: Whether enumerated reactions are required to have products that are a subset of the
                provided list of targets. If False, (the default), this identifies all reactions containing at least one
                composition from the provided list of targets (and any number of byproducts).
            filter_duplicates: Whether to remove duplicate reactions. Defaults to False.
            filter_by_chemsys: An optional chemical system for which to filter produced reactions by. This ensures that
                all output reactions contain at least one element within the provided system.
            chunk_size: The minimum number of reactions per chunk procssed. Needs to be sufficiently large to make
                parallelization a cost-effective strategy. Defaults to MIN_CHUNK_SIZE.
            max_num_jobs: The upper limit for the number of jobs created. Defaults to MAX_NUM_JOBS.
            remove_unbalanced: Whether to remove reactions which are unbalanced; this is
                usually advisable. Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by removing a reactant/product or
                having it change sides. This is also advisable for ensuring that only unique reaction sets are produced.
                Defaults to True.
            max_num_constraints: The maximum number of allowable
                constraints enforced by reaction balancing. Defaults to 1 (which is usually advisable).
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
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
        self._build_pd = True

    @staticmethod
    def _react_function(reactants, products, filtered_entries=None, pd=None, grand_pd=None, **kwargs):
        """React method for MinimizeGibbsEnumerator, which uses the interfacial reaction
        approach (see _react_interface()).
        """
        _ = products, kwargs  # unused

        r = list(reactants)
        r0 = r[0]

        r1 = r[0] if len(r) == 1 else r[1]

        return react_interface(
            r0.composition,
            r1.composition,
            filtered_entries,
            pd,
            grand_pd,
        )

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Gets the iterable used to generate reactions."""
        _ = open_combos  # unused

        return product(combos, [None])

    @staticmethod
    def _rxn_iter_length(combos, open_combos):
        _ = open_combos
        return len(combos)


class MinimizeGrandPotentialEnumerator(MinimizeGibbsEnumerator):
    """Enumerator for finding all reactions between two reactants and an open element
    that are predicted by thermo; i.e., they appear when taking the
    convex hull along a straight line connecting any two phases in Phi-x
    phase space. Identity reactions are excluded.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.
    """

    MIN_CHUNK_SIZE = 1000
    MAX_NUM_JOBS = 1000

    def __init__(
        self,
        open_elem: Element,
        mu: float,
        precursors: list[str] | None = None,
        targets: list[str] | None = None,
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
            open_elem: The element to be considered as open.
            mu: The chemical potential of the open element (eV).
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
        self.open_elem = Element(open_elem)
        self.open_phases = [Composition(str(self.open_elem)).reduced_formula]
        self.mu = mu
        self.chempots = {self.open_elem: self.mu}
        self._build_grand_pd = True

    @staticmethod
    def _react_function(reactants, products, filtered_entries=None, pd=None, grand_pd=None, **kwargs):
        """Same as the MinimizeGibbsEnumerator react function, but with ability to
        specify open element and grand potential phase diagram.
        """
        _ = products, kwargs  # unused

        r = list(reactants)
        r0 = r[0]

        r1 = r[0] if len(r) == 1 else r[1]

        open_elem = next(iter(grand_pd.chempots.keys()))

        for reactant in r:
            elems = reactant.composition.elements
            if len(elems) == 1 and elems[0] == open_elem:  # skip if reactant = open_e
                return []

        return react_interface(
            r0.composition,
            r1.composition,
            filtered_entries,
            pd,
            grand_pd=grand_pd,
        )
