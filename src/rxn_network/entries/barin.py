"""
Implements an Entry that looks up pre-tabulated Gibbs free energies from the Barin tables
"""
import hashlib
from typing import Dict, Any, List

from pymatgen.core.composition import Composition
from pymatgen.entries import Entry

from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.data import PATH_TO_BARIN, load_experimental_data

G_COMPOUNDS = load_experimental_data(PATH_TO_BARIN / "compounds.json")


class BarinReferenceEntry(ExperimentalReferenceEntry):
    """
    An Entry class for NIST-JANAF experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values (G_GASES, G_COMPOUNDS).

    Reference:
        Malcolm W. Chase Jr. NIST-JANAF thermochemical tables. Fourth edition.
        Washington, DC : American Chemical Society;  New York : American Institute of
        Physics for the National Institute of Standards and Technology, 1998.
    """

    REFERENCES = G_COMPOUNDS

    def __init__(self, composition: Composition, temperature: float):
        """
        Args:
            composition: Composition object (pymatgen).
            temperature: Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
        """
        super().__init__(composition, temperature)

    @classmethod
    def _validate_temperature(cls, formula, temperature) -> None:
        """ Ensure that the temperature is from a valid range. """
        g = cls.REFERENCES[formula]
        if temperature < min(g) or temperature > max(g):
            raise ValueError(
                f"Temperature must be selected from range: {min(g)} K to {max(g)} K"
            )
