from itertools import chain, combinations
from pathlib import Path


def limited_powerset(iterable, max_size):
    """
    Helper method for generating subsets ranging from singular
    length to maximum length specified by max_size.

    Args:
        iterable (list/set): all objects to consider.
        max_size (int): upper limit for size of combination subsets.

    Returns:
        list: all combination sets up to maximum size
    """
    return chain.from_iterable(
        [combinations(iterable, num_combos) for num_combos in range(1, max_size + 1)]
    )


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent
