"""
This module implements two types of reaction enumerators using a free energy
minimization technique, with or without the option of an open entry.
"""

from itertools import product
from typing import List, Optional

from monty.json import MontyDecoder
from pymatgen.analysis.interface_reactions import (
    GrandPotentialInterfacialReactivity,
    InterfacialReactivity,
)
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.utils import get_computed_rxn


class MinimizeGibbsEnumerator(BasicEnumerator):
    """
    Enumerator for finding all reactions between two reactants that are predicted by
    thermodynamics; i.e., they appear when taking the convex hull along a straight line
    connecting any two phases in G-x phase space. Identity reactions are automatically
    excluded.
    """

    MIN_CHUNK_SIZE = 5000

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_by_chemsys: Optional[str] = None,
        max_num_constraints: int = 1,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        calculate_e_above_hulls: bool = False,
        quiet: bool = False,
        filter_duplicates: bool = False,
        chunk_size: int = MIN_CHUNK_SIZE,
    ):
        """
        Args:
            precursors: Optional formulas of precursors.
            targets: Optional formulas of targets; only reactions which make
                these targets will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator"])
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            filter_by_chemsys=filter_by_chemsys,
            max_num_constraints=max_num_constraints,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
            calculate_e_above_hulls=calculate_e_above_hulls,
            quiet=quiet,
            filter_duplicates=filter_duplicates,
            chunk_size=chunk_size,
        )
        self._build_pd = True

    @staticmethod
    def _react_function(
        reactants, products, filtered_entries=None, pd=None, grand_pd=None, **kwargs
    ):
        """React method for MinimizeGibbsEnumerator, which uses the interfacial reaction
        approach (see _react_interface())"""
        decoder = MontyDecoder()

        filtered_entries = decoder.process_decoded(filtered_entries)
        pd = decoder.process_decoded(pd)

        r = list(reactants)
        r0 = r[0]

        if len(r) == 1:
            r1 = r[0]
        else:
            r1 = r[1]

        return react_interface(
            r0.composition,
            r1.composition,
            filtered_entries,
            pd,
            grand_pd,
        )

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Gets the iterable used to generate reactions"""
        _ = open_combos  # unused argument

        return product(combos, [None])

    @staticmethod
    def _rxn_iter_length(combos, open_combos):
        _ = open_combos
        return len(combos)


class MinimizeGrandPotentialEnumerator(MinimizeGibbsEnumerator):
    """
    Enumerator for finding all reactions between two reactants and an open element
    that are predicted by thermo; i.e., they appear when taking the
    convex hull along a straight line connecting any two phases in Phi-x
    phase space. Identity reactions are excluded.
    """

    MIN_CHUNK_SIZE = 5000

    def __init__(
        self,
        open_elem: Element,
        mu: float,
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_by_chemsys: Optional[str] = None,
        max_num_constraints: int = 1,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        calculate_e_above_hulls: bool = False,
        quiet: bool = False,
        filter_duplicates: bool = True,
        chunk_size: int = MIN_CHUNK_SIZE,
    ):
        """
        Args:
            open_elem: The element to be considered as open
            mu: The chemical potential of the open element
            precursors: Optional formulas of precursors.
            targets: Optional formulas of targets; only reactions which make
                these targets will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator])
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
            filter_duplicates: Whether to filter duplicate reactions. Defaults to True
                for grand potential enumerator due to how common duplicate reactions
                are.
        """

        super().__init__(
            precursors=precursors,
            targets=targets,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            filter_by_chemsys=filter_by_chemsys,
            max_num_constraints=max_num_constraints,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
            calculate_e_above_hulls=calculate_e_above_hulls,
            quiet=quiet,
            filter_duplicates=filter_duplicates,
            chunk_size=chunk_size,
        )
        self.open_elem = Element(open_elem)
        self.open_phases = [Composition(str(self.open_elem)).reduced_formula]
        self.mu = mu
        self.chempots = {self.open_elem: self.mu}
        self._build_grand_pd = True

    @staticmethod
    def _react_function(
        reactants, products, filtered_entries=None, pd=None, grand_pd=None, **kwargs
    ):
        """Same as the MinimizeGibbsEnumerator react function, but with ability to
        specify open element and grand potential phase diagram"""
        decoder = MontyDecoder()

        filtered_entries = decoder.process_decoded(filtered_entries)
        pd = decoder.process_decoded(pd)
        grand_pd = decoder.process_decoded(grand_pd)

        r = list(reactants)
        r0 = r[0]

        if len(r) == 1:
            r1 = r[0]
        else:
            r1 = r[1]

        open_elem = list(grand_pd.chempots.keys())[0]

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


def react_interface(r1, r2, filtered_entries, pd, grand_pd=None):
    """Simple API for InterfacialReactivity module from pymatgen."""
    chempots = None

    if grand_pd:
        interface = GrandPotentialInterfacialReactivity(
            r1,
            r2,
            grand_pd,
            pd_non_grand=pd,
            norm=True,
            include_no_mixing_energy=True,
            use_hull_energy=True,
        )
        chempots = grand_pd.chempots

    else:
        interface = InterfacialReactivity(
            r1,
            r2,
            pd,
            use_hull_energy=True,
        )

    rxns = []
    for _, _, _, rxn, _ in interface.get_kinks():
        rxn = get_computed_rxn(rxn, filtered_entries, chempots)
        rxns.append(rxn)

    return rxns
