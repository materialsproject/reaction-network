from itertools import permutations
import numpy as np
from pymatgen.entries.computed_entries import ComputedEntry
from rxn_network.utils import limited_powerset
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


def get_total_chemsys(entries, open_elem=None):
    """

    Args:
        entries:
        open_elem:

    Returns:

    """
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elem:
        elements.add(open_elem)
    return "-".join(sorted([str(e) for e in elements]))


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