"""Definitions of common job functions"""
import logging

logger = logging.getLogger(__name__)


def run_enumerators(enumerators, entries):
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


def build_network(enumerators, entries):
    return None


def run_solver():
    return None
