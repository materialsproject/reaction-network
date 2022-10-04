"""Definitions of common job functions"""

from pymatgen.core.composition import Element

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
