"""Utility functions used in the thermodynamic analysis classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, GrandPotPDEntry, PhaseDiagram
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymatgen.entries import Entry


def expand_pd(entries: Iterable[Entry], pbar: bool = False) -> dict[str, PhaseDiagram]:
    """Helper method for generating a set of smaller phase diagrams for analyzing
    thermodynamic stability in large chemical systems. This is necessary when
    considering chemical systems which contain 10 or more elements, due to dimensional
    limitations of the Qhull algorithm.

    Args:
        entries ([Entry]): list of Entry objects for building phase diagram.
        pbar (bool): whether to show a progress bar.

    Returns:
        Dictionary of PhaseDiagram objects indexed by chemical subsystem string;
        e.g. {"Li-Mn-O": <PhaseDiagram object>, "C-Y": <PhaseDiagram object>, ...}
    """
    pd_dict: dict[str, PhaseDiagram] = {}

    is_grand = all(isinstance(e, GrandPotPDEntry) for e in entries)

    sorted_entries = sorted(entries, key=lambda x: len(x.composition.elements), reverse=True)

    for e in tqdm(sorted_entries, disable=not pbar, desc="Building phase diagrams"):
        for chemsys in pd_dict:
            if set(e.composition.chemical_system.split("-")).issubset(chemsys.split("-")):
                break
        else:
            filtered_entries = list(
                filter(
                    lambda x: set(x.composition.elements).issubset(e.composition.elements),
                    entries,
                )
            )
            if is_grand:  # use grand potential phase diagram with first entry's chempots
                pd = GrandPotentialPhaseDiagram(filtered_entries, filtered_entries[0].chempots)
            else:
                pd = PhaseDiagram(filtered_entries)

            pd_dict[e.composition.chemical_system] = pd

    return pd_dict
