"""Functions for performing Pareto front analysis."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas import DataFrame


def get_pareto_front(
    df: DataFrame,
    metrics: tuple[str, ...] = (
        "energy",
        "primary_competition",
        "secondary_competition",
    ),
    maximize: bool = False,
):
    """Get the Pareto Front for a dataframe of reactions over the specified columns.

    Args:
        df: pandas DataFrame containing synthesis reactions
        metrics: Names of columns over which to calculate the Pareto front
        maximize: Whether or not maximal metrics are desired. Defaults to
            False (i.e., desired solution is to minimize the metrics).

    """
    df_original = df.copy()
    df = df_original[list(metrics)]
    pts = df.to_numpy()

    if maximize:
        pts[:, 1:] = pts[:, 1:] * -1

    return df_original[is_pareto_efficient(pts, return_mask=True)]


def is_pareto_efficient(costs, return_mask=True):
    """
    Directly lifted from @Peter's numpy-based solution on stackoverflow. Please
    give him an upvote here: https://stackoverflow.com/a/40239615. Thank you @Peter!

    Args:
        costs: An (n_points, n_costs) array
        return_mask: True to return a mask

    Returns:
        An array of indices of pareto-efficient points. If return_mask is True, this
        will be an (n_points, ) boolean array. Otherwise it will be a
        (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

    return is_efficient
