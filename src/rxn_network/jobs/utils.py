"""Definitions of common job functions"""
from collections import OrderedDict

from pymatgen.core.composition import Element
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

from rxn_network.core.composition import Composition
from rxn_network.utils.funcs import get_logger

logger = get_logger(__name__)


def run_enumerators(enumerators, entries):
    rxn_set = None
    for enumerator in enumerators:
        logger.info(f"Running {enumerator.__class__.__name__}")
        rxns = enumerator.enumerate(entries)

        logger.info(f"Adding {len(rxns)} reactions to reaction set")

        if rxn_set is None:
            rxn_set = rxns
        else:
            rxn_set = rxn_set.add_rxn_set(rxns)

    logger.info("Completed reaction enumeration. Filtering duplicates...")
    rxn_set = rxn_set.filter_duplicates()
    return rxn_set


def get_added_elem_data(entries, targets):
    added_elems = entries.chemsys - {
        str(e) for target in targets for e in Composition(target).elements
    }
    added_chemsys = "-".join(sorted(list(added_elems)))
    added_elements = [Element(e) for e in added_elems]

    return added_elements, added_chemsys


def process_entries_with_mixing_scheme(entries):
    """
    Temporary utility method for processing entries with the new R2SCAN mixing scheme.
    This is useful if mp-api is delivering incorrect entries.
    """
    reverse_entry_dict = OrderedDict()

    for e in sorted(entries, key=lambda i: len(i.composition.elements), reverse=True):
        chemsys_str = "-".join(sorted([str(el) for el in e.composition.elements]))
        if chemsys_str in reverse_entry_dict:
            reverse_entry_dict[chemsys_str].append(e)
        else:
            reverse_entry_dict[chemsys_str] = [e]

    processed_entries = set()
    seen_chemsyses = []

    for chemsys, _ in reverse_entry_dict.items():
        if chemsys in seen_chemsyses:
            continue
        els = chemsys.split("-")
        ents = []
        for chemsys_2, entries_2 in reverse_entry_dict.items():
            els_2 = chemsys_2.split("-")
            if set(els_2).issubset(els):
                seen_chemsyses.append(chemsys_2)
                ents.extend(entries_2)

        new_ents = MaterialsProjectDFTMixingScheme().process_entries(ents)

        processed_entries.update(new_ents)

    return processed_entries
