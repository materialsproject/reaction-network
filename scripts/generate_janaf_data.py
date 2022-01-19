""" 
This script generates the data/nist/compounds.json file by downloading data from the
NIST-JANAF website, using the thermochem package to acquire it.
"""

import os
import json
from collections import OrderedDict
import pandas
from tqdm import tqdm

from pymatgen.core.composition import Composition
from thermochem.janaf import Janafdb

JMOL_PER_EV = 96485.307499258

if __name__ == "__main__":
    jdb = Janafdb()

    db = pandas.read_csv("JANAF_index.txt", delimiter="|")
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

    all_data = {}

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
        if max_temp > 2000:
            max_temp = 2000
        temps = [i for i in range(300, max_temp + 100, 100)]

        dGf = [data.DeltaG(t) / JMOL_PER_EV for t in temps]

        formula, factor = Composition(f).get_integer_formula_and_factor()
        all_data[formula] = {temp: g / factor for temp, g in zip(temps, dGf)}

all_data = OrderedDict(sorted(all_data.items()))
with open("compounds.json", "w") as f:
    json.dump(all_data, f)
