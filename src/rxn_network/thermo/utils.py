"""
Utility functions used in the thermodynamic analysis classes.
"""
from typing import Dict, Iterable

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries import Entry
from tqdm import tqdm


def expand_pd(entries: Iterable[Entry], pbar: bool = False) -> Dict[str, PhaseDiagram]:
    """
    Helper method for generating a set of smaller phase diagrams for analyzing
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

    pd_dict: Dict[str, PhaseDiagram] = {}

    sorted_entries = sorted(
        entries, key=lambda x: len(x.composition.elements), reverse=True
    )

    for e in tqdm(sorted_entries, disable=not pbar, desc="Building phase diagrams"):
        for chemsys in pd_dict:
            if set(e.composition.chemical_system.split("-")).issubset(
                chemsys.split("-")
            ):
                break
        else:
            pd_dict[e.composition.chemical_system] = PhaseDiagram(
                list(
                    filter(
                        lambda x: set(x.composition.elements).issubset(
                            e.composition.elements
                        ),
                        entries,
                    )
                )
            )

    return pd_dict
