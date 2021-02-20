import numpy as np
from pymatgen.entries.computed_entries import GibbsComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram


def expand_pd(entries):
    """
    Helper method for expanding a single PhaseDiagram into a set of smaller phase
    diagrams, indexed by chemical subsystem. This is an absolutely necessary
    approach when considering chemical systems which contain > ~10 elements,
    due to limitations of the ConvexHull algorithm.
    Args:
        entries ([ComputedEntry]): list of ComputedEntry-like objects for building
            phase diagram.
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


def filter_entries(all_entries, e_above_hull, temp, include_polymorphs=False):
    """
    Helper method for filtering entries by specified energy above hull

    Args:
        all_entries ([ComputedEntry]): List of ComputedEntry-like objects to be
            filtered
        e_above_hull (float): Thermodynamic stability threshold (energy above hull)
            [eV/atom]
        include_polymorphs (bool): whether to include higher energy polymorphs of
            existing structures

    Returns:
        [ComputedEntry]: list of all entries with energies above hull equal to or
            less than the specified e_above_hull.
    """
    pd_dict = expand_pd(all_entries)
    pd_dict = {
        chemsys: PhaseDiagram(GibbsComputedStructureEntry.from_pd(pd, temp))
        for chemsys, pd in pd_dict.items()
    }

    filtered_entries = set()
    all_comps = dict()
    for chemsys, pd in pd_dict.items():
        for entry in pd.all_entries:
            if (
                    entry in filtered_entries
                    or pd.get_e_above_hull(entry) > e_above_hull
            ):
                continue
            formula = entry.composition.reduced_formula
            if not include_polymorphs and (formula in all_comps):
                if all_comps[formula].energy_per_atom < entry.energy_per_atom:
                    continue
                filtered_entries.remove(all_comps[formula])
            all_comps[formula] = entry
            filtered_entries.add(entry)

    return pd_dict, list(filtered_entries)


def simple_pca(data, k=2):
    data = data - np.mean(data.T, axis=1)  # centering the data
    cov = np.cov(data.T)  # calculating covariance matrix
    v, w = np.linalg.eig(cov)  # performing eigendecomposition
    idx = v.argsort()[::-1]  # sorting the components
    v = v[idx]
    w = w[:, idx]
    scores = data.dot(w[:, :k])

    return scores, v[:k], w[:, :k]


def get_centroid_2d(vertices):
    """ vertices must be in order"""
    n = len(vertices)
    cx = 0
    cy = 0
    a = 0

    for i in range(0, n - 1):
        xi = vertices[i, 0]
        yi = vertices[i, 1]
        xi_p = vertices[i + 1, 0]
        yi_p = vertices[i + 1, 1]
        common_term = xi * yi_p - xi_p * yi

        cx += (xi + xi_p) * common_term
        cy += (yi + yi_p) * common_term
        a += common_term

    prefactor = 0.5 / (6 * a)
    return np.array([prefactor * cx, prefactor * cy])


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

    elems = {elem for phase in phases + other_phases for elem in
             phase.composition.elements}

    reactant_mu_ranges = [entry_mu_ranges[e] for e in phases]
    reactant_mu_span = {
        elem: (min([r[elem][0] for r in reactant_mu_ranges if elem in r]),
               max([r[elem][1] for r in reactant_mu_ranges if elem in r])) for elem in
        elems}

    product_mu_ranges = [entry_mu_ranges[e] for e in other_phases]
    product_mu_span = {elem: (min([p[elem][0] for p in product_mu_ranges if elem in p]),
                              max([p[elem][1] for p in product_mu_ranges if elem in p]))
                       for elem in elems}

    out_of_bounds = False
    for elem in product_mu_span:
        if not out_of_bounds:
            if product_mu_span[elem][0] > (reactant_mu_span[elem][1] + 1e-5):
                out_of_bounds = True
        else:
            break
    return out_of_bounds