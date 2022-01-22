"""
Utility functions used by the enumerator classes.
"""
import warnings
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry, Entry

import rxn_network.costs.calculators as calcs
from rxn_network.core import Reaction
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


def initialize_entry(formula: str, entry_set: GibbsEntrySet, stabilize: bool = True):
    """
    Acquire a (stabilized) entry by user-specified formula.

    Args:
        formula: Chemical formula
        entry_set: GibbsEntrySet containing 1 or more entries corresponding to
            given formula
        stabilize: Whether or not to stabilize the entry by decreasing its energy
            such that it is 'on the hull'
    """
    try:
        entry = entry_set.get_min_entry_by_formula(formula)
    except IndexError:
        entry = entry_set.get_interpolated_entry(formula)
        warnings.warn(
            f"Using interpolated entry for {entry.composition.reduced_formula}"
        )

    if stabilize:
        entry = entry_set.get_stabilized_entry(entry, tol=1e-1)

    return entry


def initialize_calculators(
    calculators: Union[List[calcs.Calculator], List[str]], entries: GibbsEntrySet
):
    """
    Initialize a list of Calculators given a list of their names (strings) or
    uninitialized objects, and a provided list of entries.

    Args:
        calculators: List of names of calculators
        entries: List of entries or EntrySet-type object
    """
    calculators = [getattr(calcs, c) if isinstance(c, str) else c for c in calculators]
    return [c.from_entries(entries) for c in calculators]  # type: ignore


def apply_calculators(
    rxn: ComputedReaction, calculators: List[calcs.Calculator]
) -> ComputedReaction:
    """
    Decorates a reaction by applying decorate() from a list of calculators.

    Args:
        rxn: ComputedReaction object
        calculators: List of (initialized) calculators

    """
    for calc in calculators:
        rxn = calc.decorate(rxn)
    return rxn


def get_elems_set(entries: Iterable[Entry]) -> Set[str]:
    """
    Returns chemical system as a set of element names, for set of entries.

    Args:
        entries: An iterable of entry-like objects

    Returns:
        Set of element names (strings).
    """
    return {str(elem) for e in entries for elem in e.composition.elements}


def get_total_chemsys_str(
    entries: Iterable[Entry], open_elem: Optional[Element] = None
) -> str:
    """
    Returns chemical system string for set of entries, with optional open element.

    Args:
        entries: An iterable of entry-like objects
        open_elem: optional open element to include in chemical system
    """
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elem:
        elements.add(open_elem)
    return "-".join(sorted([str(e) for e in elements]))


def group_by_chemsys(
    combos: Iterable[Tuple[Entry]], open_elem: Optional[Element] = None
) -> dict:
    """
    Groups entry combinations by chemical system, with optional open element.

    Args:
        combos: Iterable of entry combinations
        open_elem: optional open element to include in chemical system grouping

    Returns:
        Dictionary of entry combos grouped by chemical system
    """
    combo_dict: Dict[str, List[Tuple[Entry]]] = {}
    for combo in combos:
        key = get_total_chemsys_str(combo, open_elem)
        if key in combo_dict:
            combo_dict[key].append(combo)
        else:
            combo_dict[key] = [combo]

    return combo_dict


def stabilize_entries(
    pd: PhaseDiagram, entries_to_adjust: Iterable[Entry], tol: float = 1e-6
) -> List[Entry]:
    """
    Simple method for stabilizing entries by decreasing their energy to be on the hull.

    WARNING: This method is not guaranteed to work *simultaneously* for all entries due
    to the fact that stabilization of one entry may destabilize others. Use with
    caution.

    Args:
        pd: PhaseDiagram object
        entries_to_adjust: Iterable of entries requiring energies to be adjusted
        tol: Numerical tolerance to ensure that the energy of the entry is below the hull

    Returns:
        A list of new entries with energies adjusted to be on the hull
    """
    indices = [pd.all_entries.index(entry) for entry in entries_to_adjust]

    new_entries = []
    for _, entry in zip(indices, entries_to_adjust):
        e_above_hull = pd.get_e_above_hull(entry)
        entry_dict = entry.to_dict()
        entry_dict["energy"] = entry.uncorrected_energy + (
            e_above_hull * entry.composition.num_atoms - tol
        )
        new_entry = ComputedEntry.from_dict(entry_dict)
        new_entries.append(new_entry)

    return new_entries


def get_min_entry_by_comp(comp: Composition, entries: Iterable[Entry]) -> Entry:
    """
    Gets the entry with the lowest energy for a given composition from a set of provided entries.

    Args:
        comp: Composition object
        entries: Iterable of entries

    Returns:
        Entry with the lowest enegy per atom for the given composition
    """
    possible_entries = filter(
        lambda e: e.composition.reduced_composition == comp.reduced_composition, entries
    )
    return sorted(possible_entries, key=lambda e: e.energy_per_atom)[0]


def get_computed_rxn(rxn: Reaction, entries: Iterable[Entry]) -> ComputedReaction:
    """
    Provided with a Reaction object and a list of possible entries, this function
    returns a new ComputedReaction object containing a selection of those entries.

    Args:
        rxn: Reaction object
        entries: Iterable of entries

    Returns:
        A ComputedReaction object transformed from a normal Reaction object
    """
    reactant_entries = [get_min_entry_by_comp(r, entries) for r in rxn.reactants]
    product_entries = [get_min_entry_by_comp(p, entries) for p in rxn.products]

    rxn = ComputedReaction.balance(reactant_entries, product_entries)
    return rxn


def get_open_computed_rxn(
    rxn: Reaction, entries: Iterable[Entry], chempots: Dict[Element, float]
) -> OpenComputedReaction:
    """
    Provided with a Reaction object and a list of possible entries, as well as a
    dictionary of chemical potentials of open elements, this function returns a new
    OpenComputedReaction object containing a selection of those entries.

    Args:
        rxn: Reaction object
        entries: Iterable of entries
        chempots: Dictionary of chemical potentials of open elements
    `
    Returns:
        An OpenComputedReaction object transformed from a normal Reaction object
    """
    reactant_entries = [get_min_entry_by_comp(r, entries) for r in rxn.reactants]
    product_entries = [get_min_entry_by_comp(p, entries) for p in rxn.products]

    rxn = OpenComputedReaction.balance(reactant_entries, product_entries, chempots)

    return rxn
