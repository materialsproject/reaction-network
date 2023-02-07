""" 
Script for generating file: data/nist/compounds.json.gz
"""

import gzip
import json
from collections import OrderedDict
from typing import Dict

import numpy as np
import pandas
from thermochem.janaf import Janafdb
from tqdm import tqdm

from rxn_network.core.composition import Composition

JMOL_PER_EV = 96485.307499258

if __name__ == "__main__":
    jdb = Janafdb()

    db = pandas.read_csv(
        "JANAF_index.txt",
        delimiter="|",
    )
    db.columns = ["formula", "name", "phase", "filename"]
    db["formula"] = db["formula"].map(str.strip)
    db["name"] = db["name"].map(str.strip)
    db["phase"] = db["phase"].map(str.strip)
    db["filename"] = db["filename"].map(str.strip)

    # Clean formulas

    cleaned_formulas = []
    for f in db["formula"].unique():
        if "-" in f or "+" in f or "D" in f:  # remove ions
            continue

        comp = Composition(f)
        if comp.is_element:
            continue

        cleaned_formulas.append(f)

    # Acquire data
    all_data: Dict[str, Dict[float, float]] = {}

    for f in tqdm(cleaned_formulas):
        phase_df = db[db["formula"] == f]
        phases = phase_df["phase"]

        # Priortize mixed phases and then single phases
        if "cr,l" in phases.unique():
            phase = phase_df[phases == "cr,l"]
        elif "l,g" in phases.unique():
            phase = phase_df[phases == "l,g"]
        elif "cr" in phases.unique():
            phase = phase_df[phases == "cr"]
        elif "l" in phases.unique():
            phase = phase_df[phases == "l"]
        elif "g" in phases.unique():
            phase = phase_df[phases == "g"]

        fn = phase["filename"].values[0]

        try:
            data = jdb.getphasedata(filename=fn)
        except pandas.errors.ParserError as e:
            print(e)
            continue

        max_temp = int((data.rawdata["T"].max() // 100) * 100)
        max_temp = min(max_temp, 2000)
        temps = [i for i in range(300, max_temp + 100, 100)]

        dg_f = [data.DeltaG(t) / JMOL_PER_EV for t in temps]

        formula, factor = Composition(f).get_integer_formula_and_factor()
        energies = {temp: g / factor for temp, g in zip(temps, dg_f)}

        existing_data = all_data.get(formula)
        if not existing_data:
            new_energies = energies
        else:
            new_energies = dict()
            for t in sorted(set(temps + list(existing_data.keys()))):
                new_e = energies.get(t, np.inf)
                old_e = existing_data.get(t, np.inf)

                new_energies[t] = min(new_e, old_e)

        all_data[formula] = new_energies

    all_data = OrderedDict(sorted(all_data.items()))

    with gzip.open("compounds.json.gz", "wt", encoding="ascii") as zipfile:
        json.dump(all_data, zipfile)
