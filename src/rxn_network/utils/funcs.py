""" Utility functions used throughout the reaction-network package."""

import logging
import sys
from datetime import datetime
from itertools import chain, combinations, zip_longest
from pathlib import Path
from typing import Any, Iterable


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


def get_logger(
    name: str,
    level=logging.DEBUG,
    log_format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
):
    """
    Code borrowed from the atomate package.

    Helper method for acquiring logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(log_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger


def datetime_str() -> str:
    """
    Get a string representation of the current time. Borrowed from atomate2.
    """
    return str(datetime.utcnow())
