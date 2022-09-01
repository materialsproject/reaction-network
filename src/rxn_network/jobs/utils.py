from rxn_network.entries.utils import get_all_entries_in_chemsys, process_entries


def get_processed_entries(chemsys, db=None):
    if db:
        entries = get_all_entries_in_chemsys(
            db,
            chemsys,
            inc_structure=inc_structure,
            compatible_only=compatible_only,
            property_data=property_data,
            use_premade_entries=False,
        )

    with MPRester() as mpr:
        entries = mpr.get_entries_in_chemsys(elements=chemsys, inc_structure="final")

    entries = process_entries(
        entries,
        temperature=temperature,
        include_nist_data=include_nist_data,
        include_barin_data=include_barin_data,
        include_freed_data=include_freed_data,
        e_above_hull=e_above_hull,
        include_polymorphs=include_polymorphs,
        formulas_to_include=formulas_to_include,
    )

    return entries


def run_enumerators(enumerators):
    rxn_set = None
    for enumerator in enumerators:
        logger.info(f"Running {enumerator.__class__.__name__}")
        rxns = enumerator.enumerate(entries)

        if rxn_set is None:
            rxn_set = rxns
        else:
            rxn_set = rxn_set.add_rxn_set(rxns)

    rxn_set = rxn_set.filter_duplicates()
    return rxn_set

def build_network()


def run_solver():

