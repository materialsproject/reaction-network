import numpy as np


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