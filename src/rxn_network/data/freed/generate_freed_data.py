"""
This script generates the data/freed/compounds.json.gz file by using the s4 package as an interface.
"""

import gzip
import json
from collections import OrderedDict

import numpy as np
from s4.thermo.exp.freed import ExpThermoDatabase

if __name__ == "__main__":
    xp = ExpThermoDatabase()
    temps = np.arange(300, 2100, 100)
    data = {}  # type: ignore

    # Acquire data
    for c, phases in xp.compositions.items():
        if c.is_element:
            continue

        formula, factor = c.get_integer_formula_and_factor()

        if not data.get(formula):
            data[formula] = dict()

        for t in temps:
            energies = []
            for p in phases:
                if t > p.tmax:
                    continue
                energies.append(p.dgf(t))

            if not energies:
                continue

            final_energy = min(energies)
            g = final_energy / factor

            if data[formula].get(t):
                data[formula][t].append(g)
            else:
                data[formula][t] = [g]

    # Clean the data
    cleaned_data = {}
    for f, energy_dict in data.items():
        cleaned_data[f] = {str(t): min(e) for t, e in energy_dict.items()}

    # Write to file
    cleaned_data = OrderedDict(sorted(cleaned_data.items()))
    with gzip.open("compounds.json.gz", "wt", encoding="ascii") as zipfile:
        json.dump(cleaned_data, zipfile)
