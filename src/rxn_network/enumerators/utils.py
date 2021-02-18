from itertools import permutations
import numpy as np
from pymatgen.entries.computed_entries import ComputedEntry
from rxn_network.utils import limited_powerset
from rxn_network.reactions.computed import ComputedReaction, OpenComputedReaction


def get_total_chemsys(entries, open_elem=None):
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elem:
        elements.add(open_elem)
    return "-".join(sorted([str(e) for e in elements]))


def group_by_chemsys(combos, open_elem=None):
    combo_dict = {}
    for combo in combos:
        key = get_total_chemsys(combo, open_elem)
        if key in combo_dict:
            combo_dict[key].append(combo)
        else:
            combo_dict[key] = [combo]

    return combo_dict


def stabilize_entries(pd, entries_to_adjust, tol=1e-6):
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
    chemsys = set(chemsys.split("-"))
    filtered_entries = list(
        filter(
            lambda e: chemsys.issuperset(e.composition.chemical_system.split("-")),
            entries,
        )
    )
    return filtered_entries


def get_entry_by_comp(comp, entries):
    comp = comp.reduced_composition
    possible_entries = filter(
        lambda e: e.composition.reduced_composition == comp, entries
    )
    return sorted(possible_entries, key=lambda e: e.energy_per_atom)[0]


def get_entries_and_coeffs(comps, coeffs, entries):
    found_entries = []
    new_coeffs = []
    for c, coeff in zip(comps, coeffs):
        if np.isclose(coeff, 0):
            continue
        entry = get_entry_by_comp(c, entries)

        ratio = c.get_reduced_composition_and_factor()[1] / \
            entry.composition.get_reduced_composition_and_factor()[1]

        found_entries.append(entry)
        new_coeffs.append(coeff * ratio)
    return found_entries, new_coeffs


def get_computed_rxn(rxn, entries):
    rxn_coeffs = np.array(rxn.coeffs)
    reactant_coeffs = rxn_coeffs[rxn_coeffs < 0]
    product_coeffs = rxn_coeffs[rxn_coeffs > 0]

    reactant_entries, reactant_new_coeffs = get_entries_and_coeffs(rxn.reactants,
                                                                reactant_coeffs,
                                                                entries)

    product_entries, product_new_coeffs = get_entries_and_coeffs(rxn.products,
                                                               product_coeffs, entries)

    new_coeffs = np.array(reactant_new_coeffs + product_new_coeffs)

    rxn = ComputedReaction(reactant_entries, product_entries, new_coeffs)
    return rxn


def get_open_computed_rxn(rxn, entries, open_entry, chempots):
    rxn_coeffs = np.array(rxn.coeffs)
    reactant_coeffs = rxn_coeffs[rxn_coeffs < 0]
    product_coeffs = rxn_coeffs[rxn_coeffs > 0]

    reactant_entries, reactant_new_coeffs = get_entries_and_coeffs(rxn.reactants,
                                                                reactant_coeffs,
                                                                entries)

    product_entries, product_new_coeffs = get_entries_and_coeffs(rxn.products,
                                                               product_coeffs, entries)

    new_coeffs = np.array(reactant_new_coeffs + product_new_coeffs)

    rxn = OpenComputedReaction(reactant_entries, product_entries, new_coeffs, chempots)

    return rxn
