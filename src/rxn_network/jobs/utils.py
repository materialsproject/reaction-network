"""Definitions of common job functions"""
import logging

import numpy as np
from mp_api import MPRester
import ray

from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.costs.calculators import (
    PrimarySelectivityCalculator,
    SecondarySelectivityCalculator,
    ChempotDistanceCalculator,
)
from rxn_network.reactions.reaction_set import ReactionSet


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


def build_network():
    pass


def run_solver():
    pass


def _get_decorated_rxns(rxns, entries, cpd_kwargs):
    cpd_calc_dict = {}
    new_rxns = []

    open_elem = rxns.open_elem
    if open_elem:
        open_elem_set = {open_elem}
    chempot = rxns.chempot

    for rxn in sorted(rxns, key=lambda rxn: len(rxn.elements), reverse=True):
        chemsys = rxn.chemical_system
        elems = chemsys.split("-")

        for c, cpd_calc in cpd_calc_dict.items():
            if set(c.split("-")).issuperset(elems):
                break
        else:
            if open_elem:
                filtered_entries = entries.get_subset_in_chemsys(
                    elems + [str(open_elem)]
                )
                filtered_entries = [
                    e.to_grand_entry({Element(open_elem): chempot})
                    for e in filtered_entries
                    if set(e.composition.elements) != open_elem_set
                ]
            else:
                filtered_entries = entries.get_subset_in_chemsys(elems)

            cpd_calc = ChempotDistanceCalculator.from_entries(
                filtered_entries, **cpd_kwargs
            )
            cpd_calc_dict[chemsys] = cpd_calc

        new_rxns.append(cpd_calc.decorate(rxn))

    results = ReactionSet.from_rxns(new_rxns, entries=entries)
    return results


def get_decorated_rxn(rxn, competing_rxns, precursors_list, temp):
    """ """
    if len(precursors_list) == 1:
        other_energies = np.array(
            [r.energy_per_atom for r in competing_rxns if r != rxn]
        )
        primary_selectivity = InterfaceReactionHull._primary_selectivity_from_energies(  # pylint: disable=protected-access
            rxn.energy_per_atom, other_energies, temp=temp
        )
        energy_diffs = rxn.energy_per_atom - other_energies
        secondary_rxn_energies = energy_diffs[energy_diffs > 0]
        secondary_selectivity = (
            secondary_rxn_energies.max() if secondary_rxn_energies.any() else 0.0
        )
        rxn.data["primary_selectivity"] = primary_selectivity
        rxn.data["secondary_selectivity"] = secondary_selectivity
        decorated_rxn = rxn
    else:
        if rxn not in competing_rxns:
            competing_rxns.append(rxn)

        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            competing_rxns,
        )

        calc_1 = PrimarySelectivityCalculator(irh=irh, temp=temp)
        calc_2 = SecondarySelectivityCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)

    return decorated_rxn


@ray.remote
def get_decorated_rxns_by_chunk(rxn_chunk, all_rxns, open_formula, temp):
    decorated_rxns = []

    for rxn in rxn_chunk:
        if not rxn:
            continue

        precursors = [r.reduced_formula for r in rxn.reactants]
        competing_rxns = list(all_rxns.get_rxns_by_reactants(precursors))

        if open_formula:
            open_formula = Composition(open_formula).reduced_formula
            competing_rxns.extend(
                all_rxns.get_rxns_by_reactants(precursors + [open_formula])
            )

        if len(precursors) >= 3:
            precursors = list(set(precursors) - {open_formula})

        decorated_rxns.append(get_decorated_rxn(rxn, competing_rxns, precursors, temp))

    return decorated_rxns
