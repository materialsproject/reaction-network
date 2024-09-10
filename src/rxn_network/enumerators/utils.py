"""Utility functions used by the reaction enumerator classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.analysis.interface_reactions import (
    GrandPotentialInterfacialReactivity,
    InterfacialReactivity,
)
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
    from pymatgen.core.periodic_table import Element
    from pymatgen.entries.computed_entries import Entry

    from rxn_network.core import Composition
    from rxn_network.entries.entry_set import GibbsEntrySet
    from rxn_network.enumerators.base import Enumerator
    from rxn_network.reactions.base import Reaction

logger = get_logger(__name__)


def get_computed_rxn(
    rxn: Reaction, entries: GibbsEntrySet, chempots: dict[Element, float] | None = None
) -> ComputedReaction | OpenComputedReaction:
    """Provided with a Reaction object and a list of possible entries, this function
    returns a new ComputedReaction object containing a selection of those entries.

    Args:
        rxn: Reaction object
        entries: Iterable of entries
        chempots: Optional dictionary of chemical potentials (will return an OpenComputedReaction if supplied).

    Returns:
        A ComputedReaction object transformed from a normal Reaction object
    """
    reactant_entries = [entries.get_min_entry_by_formula(r.reduced_formula) for r in rxn.reactants]
    product_entries = [entries.get_min_entry_by_formula(p.reduced_formula) for p in rxn.products]

    if chempots:
        rxn = OpenComputedReaction.balance(reactant_entries, product_entries, chempots)
    else:
        rxn = ComputedReaction.balance(reactant_entries, product_entries)

    return rxn


def react_interface(
    r1: Composition,
    r2: Composition,
    filtered_entries: GibbsEntrySet,
    pd: PhaseDiagram,
    grand_pd: GrandPotentialPhaseDiagram | None = None,
):
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


def get_elems_set(entries: Iterable[Entry]) -> set[str]:
    """Returns chemical system as a set of element names, for set of entries.

    Args:
        entries: An iterable of entry-like objects

    Returns:
        Set of element names (strings).
    """
    return {str(elem) for e in entries for elem in e.composition.elements}


def get_total_chemsys_str(entries: Iterable[Entry], open_elems: Iterable[Element] | None = None) -> str:
    """Returns chemical system string for set of entries, with optional open element.

    Args:
        entries: An iterable of entry-like objects
        open_elems: optional open elements to include in chemical system
    """
    elements = {elem for entry in entries for elem in entry.composition.elements}
    if open_elems:
        elements.update(list(open_elems))
    return "-".join(sorted([str(e) for e in elements]))


def group_by_chemsys(combos: Iterable[tuple[Entry, ...]], open_elems: Iterable[Element] | None = None) -> dict:
    """Groups entry combinations by chemical system, with optional open element.

    Args:
        combos: Iterable of entry combinations
        open_elems: optional open elements to include in chemical system grouping

    Returns:
        Dictionary of entry combos grouped by chemical system
    """
    combo_dict: dict[str, list[tuple[Entry, ...]]] = {}
    for combo in combos:
        key = get_total_chemsys_str(combo, open_elems)
        if key in combo_dict:
            combo_dict[key].append(combo)
        else:
            combo_dict[key] = [combo]

    return combo_dict


def stabilize_entries(pd: PhaseDiagram, entries_to_adjust: Iterable[Entry], tol: float = 1e-6) -> list[Entry]:
    """Simple method for stabilizing entries by decreasing their energy to be on the hull.

    WARNING: This method is not guaranteed to work *simultaneously* for all entries due
    to the fact that stabilization of one entry may destabilize others. Use with
    caution.

    Args:
        pd: PhaseDiagram object
        entries_to_adjust: Iterable of entries requiring energies to be adjusted
        tol: Numerical tolerance to ensure that the energy of the entry is below the
            hull

    Returns:
        A list of new entries with energies adjusted to be on the hull
    """
    indices = [pd.all_entries.index(entry) for entry in entries_to_adjust]

    new_entries = []
    for _, entry in zip(indices, entries_to_adjust):
        e_above_hull = pd.get_e_above_hull(entry)
        entry_dict = entry.to_dict()
        entry_dict["energy"] = entry.uncorrected_energy + (e_above_hull * entry.composition.num_atoms - tol)
        new_entry = ComputedEntry.from_dict(entry_dict)
        new_entries.append(new_entry)

    return new_entries


def run_enumerators(enumerators: Iterable[Enumerator], entries: GibbsEntrySet):
    """Utility method for calling enumerate() for a list of enumerators on a particular set
    of entries. Reaction sets are automatically combined and duplicates are filtered.

    Args:
        enumerators: an iterable of enumerators to use for reaction enumeration
        entries: an entry set to provide to the enumerate() function.
    """
    rxn_set = None
    for idx, enumerator in enumerate(enumerators):
        logger.info(f"Running {enumerator.__class__.__name__}")
        rxns = enumerator.enumerate(entries)

        logger.info(f"Adding {len(rxns)} reactions to reaction set")
        rxn_set = rxns if idx == 0 else rxn_set.add_rxn_set(rxns)  # type: ignore

    logger.info("Completed reaction enumeration. Filtering duplicates...")
    if rxn_set is not None:
        rxn_set = rxn_set.filter_duplicates()
    logger.info("Completed duplicate filtering.")
    return rxn_set
