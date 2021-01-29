from pathlib import Path
from monty.serialization import loadfn

cwd = Path(__file__).parent.resolve()

G_COMPOUNDS = loadfn(cwd / "compounds.json")
G_ELEMS = loadfn(cwd / "g_els.json")
G_GASES = loadfn(cwd / "nist_gas_gf.json")