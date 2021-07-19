import numpy as np
from numba import njit, prange
from rxn_network.pathways.basic import BasicPathway
from rxn_network.network.entry import NetworkEntryType

def shortest_path_to_reaction_pathway(g, path):
    rxns = []
    costs = []

    for step, v in enumerate(path):
        if (g.vp["type"][v] == NetworkEntryType.Products.value):
            e = g.edge(path[step - 1], v)

            rxns.append(g.ep["rxn"][e])
            costs.append(g.ep["cost"][e])

    return BasicPathway(rxns, costs)


@njit(parallel=True)
def balance_path_arrays(
        comp_matrices,
        net_coeffs,
        tol=1e-6,
):
    """
    Fast solution for reaction multiplicities via mass balance stochiometric
    constraints. Parallelized using Numba.

    Args:
        comp_matrices ([np.array]): list of numpy arrays containing stoichiometric
            coefficients of all compositions in all reactions, for each trial
            combination.
        net_coeffs ([np.array]): list of numpy arrays containing stoichiometric
            coefficients of net reaction.
        tol (float): numerical tolerance for determining if a multiplicity is zero
            (reaction was removed).

    Returns:
        ([bool],[np.array]): Tuple containing bool identifying which trial
            BalancedPathway objects were successfully balanced, and a list of all
            multiplicities arrays.
    """
    shape = comp_matrices.shape
    net_coeff_filter = np.argwhere(net_coeffs != 0).flatten()
    len_net_coeff_filter = len(net_coeff_filter)
    all_multiplicities = np.zeros((shape[0], shape[1]), np.float64)
    indices = np.full(shape[0], False)

    for i in prange(shape[0]):
        correct = True
        for j in range(len_net_coeff_filter):
            idx = net_coeff_filter[j]
            if not comp_matrices[i][:, idx].any():
                correct = False
                break
        if not correct:
            continue

        comp_pinv = np.linalg.pinv(comp_matrices[i]).T
        multiplicities = comp_pinv @ net_coeffs
        solved_coeffs = comp_matrices[i].T @ multiplicities

        if (multiplicities < tol).any():
            continue
        elif not (
                np.abs(solved_coeffs - net_coeffs)
                <= (1e-08 + 1e-05 * np.abs(net_coeffs))
        ).all():
            continue
        all_multiplicities[i] = multiplicities
        indices[i] = True

    filtered_indices = np.argwhere(indices != 0).flatten()
    length = filtered_indices.shape[0]
    filtered_comp_matrices = np.empty((length, shape[1], shape[2]), np.float64)
    filtered_multiplicities = np.empty((length, shape[1]), np.float64)

    for i in range(length):
        idx = filtered_indices[i]
        filtered_comp_matrices[i] = comp_matrices[idx]
        filtered_multiplicities[i] = all_multiplicities[idx]

    return filtered_comp_matrices, filtered_multiplicities

def find_interdependent_rxns(path, precursors, verbose=True):
    precursors = set(precursors)
    interdependent = False
    combined_rxn = None

    rxns = set(path.all_rxns)
    num_rxns = len(rxns)

    if num_rxns == 1:
        return False, None

    for combo in powerset(rxns, num_rxns):
        size = len(combo)
        if any([set(rxn.reactants).issubset(precursors) for rxn in combo]) or size == 1:
            continue
        other_comp = {c for rxn in (rxns - set(combo)) for c in rxn.all_comp}

        unique_reactants = []
        unique_products = []
        for rxn in combo:
            unique_reactants.append(set(rxn.reactants) - precursors)
            unique_products.append(set(rxn.products) - precursors)

        overlap = [False] * size
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                overlapping_phases = unique_reactants[i] & unique_products[j]
                if overlapping_phases and (overlapping_phases not in other_comp):
                    overlap[i] = True

        if all(overlap):
            interdependent = True

            combined_reactants = {c for p in combo for c in p.reactants}
            combined_products = {c for p in combo for c in p.products}
            shared = combined_reactants & combined_products

            combined_reactants = combined_reactants - shared
            combined_products = combined_products - shared
            try:
                combined_rxn = Reaction(
                    list(combined_reactants), list(combined_products)
                )
                if verbose:
                    print(combined_rxn)
            except ReactionError:
                print("Could not combine interdependent reactions!")

    return interdependent, combined_rxn