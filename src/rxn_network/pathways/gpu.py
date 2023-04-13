from datetime import datetime

import cupy as cp
import numpy as np
import ray


@ray.remote(num_gpus=1)
def _balance_path_arrays_gpu(
    comp_matrices: np.ndarray,
    net_coeffs: np.ndarray,
    tol: float = 1e-6,
):
    """
    WARNING: this GPU-based method is experimental. Use at your own risk.

    It turns out that pseudoinverses are not faster on the GPU than the CPU. (at least,
    not yet!)

    Args:
        comp_matrices: Array containing stoichiometric coefficients of all
            compositions in all reactions, for each trial combination.
        net_coeffs: Array containing stoichiometric coefficients of net reaction.
        tol: numerical tolerance for determining if a multiplicity is zero
            (reaction was removed).
    """
    comp_matrices = cp.asarray(comp_matrices)
    net_coeffs = cp.asarray(net_coeffs)

    checkpoint1 = datetime.now()

    comp_pinv = cp.linalg.pinv(comp_matrices)

    checkpoint2 = datetime.now()
    print(f"performing pseudoinverse: {checkpoint2 - checkpoint1}")

    comp_pinv = comp_pinv.transpose(0, 2, 1)

    checkpoint2_5 = datetime.now()
    print(f"transposing pseudoinverse result: {checkpoint2_5 - checkpoint2}")

    multiplicities = comp_pinv @ net_coeffs

    checkpoint3 = datetime.now()
    print(f"get multiplicities: {checkpoint3 - checkpoint2_5}")

    multiplicities_filter = (multiplicities < tol).any(axis=1)

    checkpoint4 = datetime.now()
    print(f"create multiplicities filter: {checkpoint4 - checkpoint3}")

    multiplicities = multiplicities[~multiplicities_filter]

    checkpoint5 = datetime.now()
    print(f"filter multiplicities: {checkpoint5 - checkpoint4}")
    comp_matrices = comp_matrices[~multiplicities_filter]

    checkpoint6 = datetime.now()
    print(f"filter comp_matrices: {checkpoint6 - checkpoint5}")

    stack_size = len(comp_matrices)
    if stack_size == 0:
        return [None], [None]

    checkpoint7 = datetime.now()
    print(f"check stack size: {checkpoint7 - checkpoint6}")

    solved_coeffs = (
        comp_matrices.transpose(0, 2, 1) @ multiplicities.reshape(stack_size, -1, 1)
    ).reshape(stack_size, -1)

    checkpoint8 = datetime.now()
    print(f"solving for coeffs: {checkpoint8 - checkpoint7}")

    correct_filter = cp.isclose(
        solved_coeffs, cp.repeat(net_coeffs.reshape(1, -1), stack_size, axis=0)
    ).all(axis=1)

    checkpoint9 = datetime.now()
    print(f"finding correct filter: {checkpoint9 - checkpoint8}")

    filtered_comp_matrices = comp_matrices[correct_filter]

    checkpoint10 = datetime.now()
    print(f"filtering comp matrices: {checkpoint10 - checkpoint9}")

    filtered_multiplicities = multiplicities[correct_filter]

    checkpoint11 = datetime.now()
    print(f"filtering multiplicities: {checkpoint11 - checkpoint10}")

    # checkpoint2 = datetime.now()
    # print(f"GPU main operations took: {checkpoint2-checkpoint1}")

    filtered_comp_matrices = cp.asnumpy(filtered_comp_matrices)
    filtered_multiplicities = cp.asnumpy(filtered_multiplicities)

    # checkpoint3 = datetime.now()
    # print(f"Converting back to CPU took: {checkpoint3-checkpoint2}")

    return filtered_comp_matrices, filtered_multiplicities
