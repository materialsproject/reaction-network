""" Utility functions used throughout the reaction-network package."""

from typing import Iterable, Any
from itertools import chain, combinations, zip_longest
from pathlib import Path


def limited_powerset(iterable, max_size) -> Iterable:
    """
    Helper method for generating subsets ranging from singular
    length to maximum length specified by max_size.

    Args:
        iterable (list/set): all objects to consider.
        max_size (int): upper limit for size of combination subsets.

    Returns:
        All combination sets up to maximum size
    """
    return chain.from_iterable(
        [combinations(iterable, num_combos) for num_combos in range(1, max_size + 1)]
    )


def grouper(iterable: Iterable, n: int, fillvalue: Any = None):
    """
    Collects data into fixed-length chunks or blocks.

    Args:
        iterable: An iterable object to group.
        n: The number of items to include in each group.
        fillvalue: The value to use for the last group, if the length of the group is
            less than n.
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_project_root() -> Path:
    """
    Gets a Path object for the reaction-network project root directory.

    Note:
        This is specific to this file and project.

    Returns:
        Path object for the project root directory.
    """
    return Path(__file__).parent.parent.parent
