"""
Experimental Gibbs free energy data from NIST-JANAF (compounds, gases) and FactSage (elements)
"""
from pathlib import Path

from monty.serialization import loadfn

cwd = Path(__file__).parent.resolve()

G_COMPOUNDS = loadfn(cwd / "compounds.json")
G_ELEMS = loadfn(cwd / "elements.json")
G_GASES = loadfn(cwd / "gases.json")
