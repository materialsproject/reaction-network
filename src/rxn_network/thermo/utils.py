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

    pd_dict = dict()

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


def check_chempot_bounds(pd, rxn):
    phases = rxn._reactant_entries
    other_phases = rxn._product_entries

    entry_mu_ranges = {}

    for e in phases + other_phases:
        elems = e.composition.elements
        chempot_ranges = {}
        all_chempots = {e: [] for e in elems}
        for simplex, chempots in pd.get_all_chempots(e.composition).items():
            for elem in elems:
                all_chempots[elem].append(chempots[elem])
        for elem in elems:
            chempot_ranges[elem] = (min(all_chempots[elem]), max(all_chempots[elem]))

        entry_mu_ranges[e] = chempot_ranges

    elems = {
        elem for phase in phases + other_phases for elem in phase.composition.elements
    }

    reactant_mu_ranges = [entry_mu_ranges[e] for e in phases]
    reactant_mu_span = {
        elem: (
            min([r[elem][0] for r in reactant_mu_ranges if elem in r]),
            max([r[elem][1] for r in reactant_mu_ranges if elem in r]),
        )
        for elem in elems
    }

    product_mu_ranges = [entry_mu_ranges[e] for e in other_phases]
    product_mu_span = {
        elem: (
            min([p[elem][0] for p in product_mu_ranges if elem in p]),
            max([p[elem][1] for p in product_mu_ranges if elem in p]),
        )
        for elem in elems
    }

    out_of_bounds = False
    for elem in product_mu_span:
        if not out_of_bounds:
            if product_mu_span[elem][0] > (reactant_mu_span[elem][1] + 1e-5):
                out_of_bounds = True
        else:
            break
    return out_of_bounds
