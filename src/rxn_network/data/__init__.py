"""
Experimental Gibbs free energy data from NIST-JANAF (compounds, gases), Barin tables
(all compounds), and FactSage (elements)
"""
from pathlib import Path
from typing import Any, Dict, Union

from monty.serialization import loadfn

cwd = Path(__file__).parent.resolve()

PATH_TO_BARIN = cwd / "barin"
PATH_TO_FREED = cwd / "freed"
PATH_TO_NIST = cwd / "nist"

G_ELEMS = loadfn(cwd / "elements.json")


def load_experimental_data(fn: Union[str, Path]) -> Dict[str, Dict[float, Any]]:
    """
    Load experimental data from a json file.

    Args:
        fn: The filename of the json file
    """
    d = loadfn(fn)
    return {comp: make_float_keys(data) for comp, data in d.items()}


def make_float_keys(d) -> Dict[float, Any]:
    """
    Convert all keys in a dict to floats.
    """
    return {float(k): v for k, v in d.items()}
