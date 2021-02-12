from itertools import permutations
import numpy as np
from pymatgen.entries.computed_entries import ComputedEntry
from rxn_network.utils import limited_powerset
from rxn_network.reactions.computed import ComputedReaction


def get_total_chemsys(entries):
    elements = sorted(
        list({elem for entry in entries for elem in entry.composition.elements})
    )
    return "-".join([str(e) for e in elements])


def group_by_chemsys(combos):
    combo_dict = {}
    for combo in combos:
        key = get_total_chemsys(combo)
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
    possible_entries = filter(
        lambda x: x.composition.reduced_composition == comp, entries
    )
    return sorted(possible_entries, key=lambda x: x.energy_per_atom)[0]


def get_computed_rxn(rxn, entries):
    reactants = [
        r.reduced_composition
        for r in rxn.reactants
        if not np.isclose(rxn.get_coeff(r), 0)
    ]
    products = [
        p.reduced_composition
        for p in rxn.products
        if not np.isclose(rxn.get_coeff(p), 0)
    ]
    reactant_entries = [get_entry_by_comp(r, entries) for r in reactants]
    product_entries = [get_entry_by_comp(p, entries) for p in products]

    return ComputedReaction.balance(reactant_entries, product_entries)
