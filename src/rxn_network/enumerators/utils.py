"""
Utility functions used by the enumerator classes.
"""
import warnings
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import ray
from pymatgen.analysis.interface_reactions import (
    GrandPotentialInterfacialReactivity,
    InterfacialReactivity,
)
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry, Entry

from rxn_network.core.reaction import Reaction
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


@ray.remote
def _react(
    rxn_iterable,
    react_function,
    open_entries,
    precursors,
    targets,
    p_set_func,
    t_set_func,
    remove_unbalanced,
    remove_changed,
    max_num_constraints,
    filtered_entries=None,
    pd=None,
    grand_pd=None,
):
    """
    This function is a wrapper for the specific react function of each enumerator. This
    wrapper contains the logic used for filtering out reactions based on the
    user-defined enumerator settings. It can also be called as a remote function using
    ray, allowing for parallel computation during reaction enumeration.

    Note: this function is not intended to to be called directly!

    """

    all_rxns = []
    for rp in rxn_iterable:
        if not rp:
            continue

        r = set(rp[0]) if rp[0] else set()
        p = set(rp[1]) if rp[1] else set()

        all_phases = r | p

        precursor_func = (
            getattr(precursors | open_entries, p_set_func)
            if precursors
            else lambda e: True
        )
        target_func = (
            getattr(targets | open_entries, t_set_func) if targets else lambda e: True
        )

        if (
            (r & p)
            or (precursors and not precursors & all_phases)
            or (p and targets and not targets & all_phases)
        ):
            continue

        if not (precursor_func(r) or (p and precursor_func(p))):
            continue
        if p and not (target_func(r) or target_func(p)):
            continue

        suggested_rxns = react_function(
            r, p, filtered_entries=filtered_entries, pd=pd, grand_pd=grand_pd
        )

        rxns = []
        for rxn in suggested_rxns:
            if (
                rxn.is_identity
                or (remove_unbalanced and not rxn.balanced)
                or (remove_changed and rxn.lowest_num_errors != 0)
                or rxn.data["num_constraints"] > max_num_constraints
            ):
                continue

            reactant_entries = set(rxn.reactant_entries) - open_entries
            product_entries = set(rxn.product_entries) - open_entries

            if precursor_func(reactant_entries) and target_func(product_entries):
                rxns.append(rxn)

        all_rxns.extend(rxns)

    return all_rxns


def react_interface(r1, r2, filtered_entries, pd, grand_pd=None):
    """Simple API for InterfacialReactivity module from pymatgen."""
    chempots = None

    if grand_pd:
        interface = GrandPotentialInterfacialReactivity(
            r1,
            r2,
            grand_pd,
            pd_non_grand=pd,
            norm=True,
            include_no_mixing_energy=True,
            use_hull_energy=True,
        )
        chempots = grand_pd.chempots

    else:
        interface = InterfacialReactivity(
            r1,
            r2,
            pd,
            use_hull_energy=True,
        )

    rxns = []
    for _, _, _, rxn, _ in interface.get_kinks():
        rxn = get_computed_rxn(rxn, filtered_entries, chempots)
        rxns.append(rxn)

    return rxns


def get_elems_set(entries: Iterable[Entry]) -> Set[str]:
    """
    Returns chemical system as a set of element names, for set of entries.

    Args:
        entries: An iterable of entry-like objects

    Returns:
        Set of element names (strings).
    """
    return {str(elem) for e in entries for elem in e.composition.elements}


def get_total_chemsys_str(
    entries: Iterable[Entry], open_elems: Optional[Iterable[Union[Element]]] = None
) -> str:
    """
    Returns chemical system string for set of entries, with optional open element.

    Args:
        entries: An iterable of entry-like objects
        open_elem: optional open element to include in chemical system
    """
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elems:
        elements.update([e for e in open_elems])
    return "-".join(sorted([str(e) for e in elements]))


def group_by_chemsys(
    combos: Iterable[Tuple[Entry]], open_elems: Optional[Iterable[Element]] = None
) -> dict:
    """
    Groups entry combinations by chemical system, with optional open element.

    Args:
        combos: Iterable of entry combinations
        open_elem: optional open element to include in chemical system grouping

    Returns:
        Dictionary of entry combos grouped by chemical system
    """
    combo_dict: Dict[str, List[Tuple[Entry]]] = {}
    for combo in combos:
        key = get_total_chemsys_str(combo, open_elems)
        if key in combo_dict:
            combo_dict[key].append(combo)
        else:
            combo_dict[key] = [combo]

    return combo_dict


def stabilize_entries(
    pd: PhaseDiagram, entries_to_adjust: Iterable[Entry], tol: float = 1e-6
) -> List[Entry]:
    """
    Simple method for stabilizing entries by decreasing their energy to be on the hull.

    WARNING: This method is not guaranteed to work *simultaneously* for all entries due
    to the fact that stabilization of one entry may destabilize others. Use with
    caution.

    Args:
        pd: PhaseDiagram object
        entries_to_adjust: Iterable of entries requiring energies to be adjusted
        tol: Numerical tolerance to ensure that the energy of the entry is below the hull

    Returns:
        A list of new entries with energies adjusted to be on the hull
    """
    indices = [pd.all_entries.index(entry) for entry in entries_to_adjust]

    new_entries = []
    for _, entry in zip(indices, entries_to_adjust):
        e_above_hull = pd.get_e_above_hull(entry)
        entry_dict = entry.to_dict()
        entry_dict["energy"] = entry.uncorrected_energy + (
            e_above_hull * entry.composition.num_atoms - tol
        )
        new_entry = ComputedEntry.from_dict(entry_dict)
        new_entries.append(new_entry)

    return new_entries


def get_computed_rxn(
    rxn: Reaction, entries: GibbsEntrySet, chempots=None
) -> ComputedReaction:
    """
    Provided with a Reaction object and a list of possible entries, this function
    returns a new ComputedReaction object containing a selection of those entries.

    Args:
        rxn: Reaction object
        entries: Iterable of entries

    Returns:
        A ComputedReaction object transformed from a normal Reaction object
    """
    reactant_entries = [
        entries.get_min_entry_by_formula(r.reduced_formula) for r in rxn.reactants
    ]
    product_entries = [
        entries.get_min_entry_by_formula(p.reduced_formula) for p in rxn.products
    ]

    if chempots:
        rxn = OpenComputedReaction.balance(reactant_entries, product_entries, chempots)
    else:
        rxn = ComputedReaction.balance(reactant_entries, product_entries)

    return rxn
