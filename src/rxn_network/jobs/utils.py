"""Definitions of common job functions"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from pymatgen.core.composition import Element

from rxn_network.core import Composition
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from rxn_network.entries.entry_set import GibbsEntrySet

logger = get_logger(__name__)


def get_added_elem_data(
    entries: GibbsEntrySet, targets: Iterable[Composition | str]
) -> tuple[list[Element], str]:
    """
    Given a provided entry set and targets, this identifies which elements in the entry
    set are "additional" (not found in the target)

    Args:
        entries: the full entry set
        targets: the target phase compositions

    Returns:
        A tuple of the additional elements and their chemical system string.
    """
    added_elems = entries.chemsys - {
        str(e) for target in targets for e in Composition(target).elements
    }
    added_chemsys = "-".join(sorted(list(added_elems)))
    added_elements = [Element(e) for e in added_elems]

    return added_elements, added_chemsys
