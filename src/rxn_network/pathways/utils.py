"""
Utility functions used in reaction pathway balancing.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def balance_path_arrays(
    comp_matrices: np.ndarray,
    net_coeffs: np.ndarray,
    tol: float = 1e-6,
):
    """
    Fast solution for reaction multiplicities via mass balance stochiometric
    constraints. Parallelized using Numba.

    Args:
        comp_matrices: Array containing stoichiometric coefficients of all
            compositions in all reactions, for each trial combination.
        net_coeffs: Array containing stoichiometric coefficients of net reaction.
        tol: numerical tolerance for determining if a multiplicity is zero
            (reaction was removed).
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
            np.abs(solved_coeffs - net_coeffs) <= (1e-08 + 1e-05 * np.abs(net_coeffs))
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
