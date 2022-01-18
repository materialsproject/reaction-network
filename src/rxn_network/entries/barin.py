"""
Implements an Entry that looks up pre-tabulated Gibbs free energies from the Barin tables.
"""
from typing import Dict, Optional

from pymatgen.core.composition import Composition

from rxn_network.data import PATH_TO_BARIN, load_experimental_data
from rxn_network.entries.experimental import ExperimentalReferenceEntry

G_COMPOUNDS = load_experimental_data(PATH_TO_BARIN / "compounds.json")


class BarinReferenceEntry(ExperimentalReferenceEntry):
    """
    An Entry class for Barin experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values.

    Reference:
        Barin, I. (1995). Thermochemical data of pure substances. John Wiley & Sons,
            Ltd. https://doi.org/10.1002/9783527619825
    """

    REFERENCES = G_COMPOUNDS

    def __init__(
        self, composition: Composition, temperature: float, data: Optional[Dict] = None
    ):
        """
        Args:
            composition: Composition object (within pymatgen).
            temperature: Absolute temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
        """
        super().__init__(composition, temperature, data)
