"""
Implements an Entry that looks up pre-tabulated Gibbs free energies from the NIST-JANAF
tables.
"""
from typing import Dict, List, Optional

from pymatgen.entries.computed_entries import EnergyAdjustment

from rxn_network.core.composition import Composition
from rxn_network.data import PATH_TO_FREED, load_experimental_data
from rxn_network.entries.experimental import ExperimentalReferenceEntry

G_COMPOUNDS = load_experimental_data(PATH_TO_FREED / "compounds.json.gz")


class FREEDReferenceEntry(ExperimentalReferenceEntry):
    """
    An Entry class for FREED experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values.

    Reference:
        https://www.thermart.net/freed-thermodynamic-database/
    """

    REFERENCES = G_COMPOUNDS

    def __init__(
        self,
        composition: Composition,
        temperature: float,
        energy_adjustments: Optional[List[EnergyAdjustment]] = None,
        data: Optional[Dict] = None,
    ):
        """
        Args:
            composition: Composition object (within pymatgen).
            temperature: Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
            data: Optional dictionary containing entry data
        """
        super().__init__(
            composition=composition,
            temperature=temperature,
            energy_adjustments=energy_adjustments,
            data=data,
        )
