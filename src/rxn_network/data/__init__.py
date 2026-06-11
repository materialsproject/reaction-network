"""Experimental Gibbs free energy data from NIST-JANAF (compounds, gases), FREED (all
compounds), and FactSage (elemental chemical potentials).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from monty.serialization import loadfn

cwd = Path(__file__).parent.resolve()

PATH_TO_FREED = cwd / "freed"
PATH_TO_NIST = cwd / "nist"

COMMON_GASES = loadfn(cwd / "common_gases.json")
G_ELEMS = loadfn(cwd / "mu_elements.json")
CONFIG_ENTROPY = loadfn(cwd / "icsd_ideal_config_entropy.json")

def load_experimental_data(fn: str | Path) -> dict[str, dict[float, Any]]:
    """Load experimental data from a json file.

    Args:
        fn: The filename of the json file
    """
    d = loadfn(fn)
    return {comp: make_float_keys(data) for comp, data in d.items()}


def make_float_keys(d) -> dict[float, Any]:
    """Convert all keys in a dict to floats."""
    return {float(k): v for k, v in d.items()}
