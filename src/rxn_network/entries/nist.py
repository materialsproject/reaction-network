"""
Implements an Entry that looks up NIST pre-tabulated Gibbs free energies
"""
import hashlib
from typing import Dict, Any

from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from scipy.interpolate import interp1d

from rxn_network.data import G_COMPOUNDS, G_GASES


class NISTReferenceEntry(Entry):
    """
    An Entry class for NIST-JANAF experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values (G_GASES, G_COMPOUNDS).

    Reference:
        Malcolm W. Chase Jr. NIST-JANAF thermochemical tables. Fourth edition.
        Washington, DC : American Chemical Society;  New York : American Institute of
        Physics for the National Institute of Standards and Technology, 1998.
    """

    REFERENCES = {**G_COMPOUNDS, **G_GASES}

    def __init__(self, composition: Composition, temperature: float):
        """
        Args:
            composition: Composition object (pymatgen).
            temperature: Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
        """
        composition = Composition(composition)
        formula = composition.reduced_formula

        if formula not in NISTReferenceEntry.REFERENCES:
            raise ValueError("Formula must be in NIST-JANAF thermochemical tables")

        if temperature < 300 or temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K")

        energy = self._get_nist_energy(formula, temperature)

        self.temperature = temperature
        self._formula = formula
        self.name = formula
        self.entry_id = "NISTReferenceEntry"
        self.data = {}  # type: Dict[Any, Any]

        super().__init__(composition.reduced_composition, energy)

    @staticmethod
    def _get_nist_energy(formula: str, temperature: float) -> float:
        """
        Convenience method for accessing and interpolating NIST-JANAF data.

        Args:
            formula: Chemical formula by which to search NIST-JANAF data.
            temperature: Absolute temperature [K].

        Returns:
            Gibbs free energy of formation of formula at specified temperature [eV]
        """
        data = NISTReferenceEntry.REFERENCES[formula]
        if temperature % 100 > 0:
            g_interp = interp1d([float(t) for t in data.keys()], list(data.values()))
            return g_interp(temperature)[()]

        return data[str(temperature)]

    @property
    def energy(self) -> float:
        """The energy of the entry, as supplied by the NIST-JANAF tables."""
        return self._energy

    @property
    def correction_uncertainty(self) -> float:
        """ Uncertainty of NIST-JANAF data is not supplied."""
        return 0

    @property
    def correction_uncertainty_per_atom(self) -> float:
        """Uncertainty of NIST-JANAF data is not supplied."""
        return 0

    @property
    def is_experimental(self) -> bool:
        """ Returns True by default."""
        return True

    def as_dict(self) -> dict:
        """ Returns an MSONable dict."""
        data = super().as_dict()
        data["temperature"] = self.temperature
        return data

    @classmethod
    def from_dict(cls, d) -> "NISTReferenceEntry":
        """ Returns NISTReferenceEntry constructed from MSONable dict."""
        return cls(composition=d["composition"], temperature=d["temperature"])

    def __repr__(self):
        output = [
            f"NISTReferenceEntry | {self._formula}",
            f"Gibbs Energy ({self.temperature} K) = {self.energy:.4f}",
        ]
        return "\n".join(output)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self._formula == other._formula) and (
                self.temperature == other.temperature
            )
        return False

    def __hash__(self):
        data_md5 = hashlib.md5(
            "NISTReferenceEntry"
            f"{self.composition}_"
            f"{self.temperature}".encode("utf-8")
        ).hexdigest()
        return int(data_md5, 16)
