from more_itertools import powerset

def get_entry_combinations(entries, max_size=2):
    return list(powerset(entries, max_size))

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
