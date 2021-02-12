from itertools import permutations
from pymatgen.entries.computed_entries import ComputedEntry
from rxn_network.utils import limited_powerset
from rxn_network.reactions.computed import ComputedReaction

def get_total_chemsys(entries):
    elements = sorted(list({elem for entry in entries for elem in
                                 entry.composition.elements}))
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
        entry_dict["energy"] = entry.uncorrected_energy + \
                               (e_above_hull * entry.composition.num_atoms)
        new_entry = ComputedEntry.from_dict(entry_dict)
        new_entries.append(new_entry)
    return new_entries

