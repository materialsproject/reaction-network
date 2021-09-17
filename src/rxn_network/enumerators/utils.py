"""
Helpful utility functions used by the enumerator classes.
"""
from itertools import permutations

from typing import List

import numpy as np
from pymatgen.entries.computed_entries import Entry, ComputedEntry

import rxn_network.costs.calculators as calcs
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


def initialize_entry(formula: str, entry_set: GibbsEntrySet, stabilize: bool = True):
    """
    Acquire a (stabilizeD) entry by user-specified formula.

    Args:
        formula: Chemical formula
        entry_set: GibbsEntrySet containing 1 or more entries corresponding to
            given formula
        stabilize: Whether or not to stabilize the entry by decreasing its energy
            such that it is 'on the hull'

    """
    entry = entry_set.get_min_entry_by_formula(formula)
    if stabilize:
        entry = entry_set.stabilize_entry(entry)
    return entry


def initialize_open_entries(open_entries, entry_set):
    """

    Args:
        open_entries:
        entry_set:

    Returns:

    """
    open_entries = [entry_set.get_min_entry_by_formula(e) for e in open_entries]
    return open_entries


def initialize_calculators(calculators, entries):
    """

    Args:
        calculators:
        entries:

    Returns:

    """
    calculators = [getattr(calcs, c) if isinstance(c, str) else c for c in calculators]
    return [c.from_entries(entries) for c in calculators]


def apply_calculators(rxn, calculators):
    """

    Args:
        rxn:
        calculators:

    Returns:

    """
    for calc in calculators:
        rxn = calc.decorate(rxn)
    return rxn


def get_total_chemsys(entries: List[Entry], open_elem=None):
    """
    Returns chemical system for set of entries, with optional open element.

    Args:
        entries:
        open_elem:
    """
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elem:
        elements.add(open_elem)
    return "-".join(sorted([str(e) for e in elements]))


def get_elems_set(entries):
    return {str(elem) for e in entries for elem in e.composition.elements}


def group_by_chemsys(combos, open_elem=None):
    """

    Args:
        combos:
        open_elem:

    Returns:

    """
    combo_dict = {}
    for combo in combos:
        key = get_total_chemsys(combo, open_elem)
        if key in combo_dict:
            combo_dict[key].append(combo)
        else:
            combo_dict[key] = [combo]

    return combo_dict


def stabilize_entries(pd, entries_to_adjust, tol=1e-6):
    """

    Args:
        pd:
        entries_to_adjust:
        tol:

    Returns:

    """
    indices = [pd.all_entries.index(entry) for entry in entries_to_adjust]
    new_entries = []
    for idx, entry in zip(indices, entries_to_adjust):
        e_above_hull = pd.get_e_above_hull(entry)
        entry_dict = entry.to_dict()
        entry_dict["energy"] = entry.uncorrected_energy + (
            e_above_hull * entry.composition.num_atoms
        )
        new_entry = ComputedEntry.from_dict(entry_dict)
        new_entries.append(new_entry)
    return new_entries


def filter_entries_by_chemsys(entries, chemsys):
    """

    Args:
        entries:
        chemsys:

    Returns:

    """
    chemsys = set(chemsys.split("-"))

    filtered_entries = list(
        filter(
            lambda e: chemsys.issuperset(e.composition.chemical_system.split("-")),
            entries,
        )
    )
    return filtered_entries


def get_entry_by_comp(comp, entries):
    """

    Args:
        comp:
        entries:

    Returns:

    """
    possible_entries = filter(
        lambda e: e.composition.reduced_composition == comp.reduced_composition, entries
    )
    return sorted(possible_entries, key=lambda e: e.energy_per_atom)[0]


def get_computed_rxn(rxn, entries):
    """

    Args:
        rxn:
        entries:

    Returns:

    """
    reactant_entries = [get_entry_by_comp(r, entries) for r in rxn.reactants]
    product_entries = [get_entry_by_comp(p, entries) for p in rxn.products]

    rxn = ComputedReaction.balance(reactant_entries, product_entries)
    return rxn


def get_open_computed_rxn(rxn, entries, chempots):
    """

    Args:
        rxn:
        entries:
        chempots:

    Returns:

    """
    reactant_entries = [get_entry_by_comp(r, entries) for r in rxn.reactants]
    product_entries = [get_entry_by_comp(p, entries) for p in rxn.products]

    rxn = OpenComputedReaction.balance(reactant_entries, product_entries, chempots)

    return rxn
