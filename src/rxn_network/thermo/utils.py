"""
Utility functions used in the thermodynamic analysis classes.
"""

from typing import Dict, List

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries import Entry


def expand_pd(entries: List[Entry]) -> Dict[str, PhaseDiagram]:
    """
    Helper method for generating a set of smaller phase diagrams for analyzing
    thermodynamic staiblity in large chemical systems. This is necessary when
    considering chemical systems which contain 10 or more elements, due to dimensional
    limitations of the Qhull algorithm.

    Args:
        entries ([Entry]): list of Entry objects for building phase diagram.
    Returns:
        Dictionary of PhaseDiagram objects indexed by chemical subsystem string;
        e.g. {"Li-Mn-O": <PhaseDiagram object>, "C-Y": <PhaseDiagram object>, ...}
    """

    pd_dict: Dict[str, PhaseDiagram] = dict()

    sorted_entries = sorted(
        entries, key=lambda x: len(x.composition.elements), reverse=True
    )

    for e in sorted_entries:
        for chemsys in pd_dict.keys():
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
