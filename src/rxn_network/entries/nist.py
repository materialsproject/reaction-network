"""
Implements an Entry that looks up pre-tabulated Gibbs free energies from the NIST-JANAF
tables.
"""
from typing import Dict, Optional

from pymatgen.core.composition import Composition

from rxn_network.data import PATH_TO_NIST, load_experimental_data
from rxn_network.entries.experimental import ExperimentalReferenceEntry

G_COMPOUNDS = load_experimental_data(PATH_TO_NIST / "compounds.json")


class NISTReferenceEntry(ExperimentalReferenceEntry):
    """
    An Entry class for NIST-JANAF experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values.

    Reference:
        Malcolm W. Chase Jr. NIST-JANAF thermochemical tables. Fourth edition.
        Washington, DC : American Chemical Society;  New York : American Institute of
        Physics for the National Institute of Standards and Technology, 1998.
    """

    REFERENCES = G_COMPOUNDS

    def __init__(
        self, composition: Composition, temperature: float, data: Optional[Dict] = None
    ):
        """
        Args:
            composition: Composition object (within pymatgen).
            temperature: Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
            data: Optional dictionary containing entry data
        """
        super().__init__(composition, temperature, data)
