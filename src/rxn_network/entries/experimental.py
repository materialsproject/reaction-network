"""
Implements an Entry that looks up NIST pre-tabulated Gibbs free energies
"""
import hashlib
from typing import Dict, Any, List

from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from scipy.interpolate import interp1d


class ExperimentalReferenceEntry(Entry):
    """
    An Entry class for experimental reference data. Given a composition,
    automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values.
    """

    REFERENCES: Dict = {}

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
        self._validate_temperature(formula, temperature)

        energy = self._get_energy(formula, temperature)

        self.temperature = temperature
        self._formula = formula
        self.name = formula
        self.entry_id = self.__class__.__name__
        self.data = {}  # type: Dict[Any, Any]

        super().__init__(composition.reduced_composition, energy)

    def get_new_temperature(self, new_temperature: float) -> "NISTReferenceEntry":
        """
        Return a copy of the NISTReferenceEntry at the new specified temperature.

        Args:
            new_temperature: The new temperature to use [K]

        Returns:
            A copy of the NISTReferenceEntry at the new specified temperature.
        """
        new_entry_dict = self.as_dict()
        new_entry_dict["temperature"] = new_temperature

        new_entry = self.from_dict(new_entry_dict)
        return new_entry

    @classmethod
    def _validate_temperature(cls, formula, temperature) -> None:
        """ Ensure that the temperature is from a valid range. """
        if temperature < 300 or temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K")

    @classmethod
    def _get_energy(cls, formula: str, temperature: float) -> float:
        """
        Convenience method for accessing and interpolating experimental data.

        Args:
            formula: Chemical formula by which to search experimental data.
            temperature: Absolute temperature [K].

        Returns:
            Gibbs free energy of formation of formula at specified temperature [eV]
        """
        data = cls.REFERENCES[formula]
        if temperature % 100 > 0:
            g_interp = interp1d(list(data.keys()), list(data.values()))
            return g_interp(temperature)[()]

        return data[temperature]

    @property
    def energy(self) -> float:
        """The energy of the entry, as supplied by the reference tables."""
        return self._energy

    @property
    def correction_uncertainty(self) -> float:
        """ Uncertainty of experimental data is not supplied."""
        return 0

    @property
    def correction_uncertainty_per_atom(self) -> float:
        """Uncertainty of experimental data is not supplied."""
        return 0

    @property
    def energy_adjustments(self) -> List:
        """
        Returns a list of energy adjustments. Not implemented for experimental data.
        """
        return list()

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
    def from_dict(cls, d) -> "ExperimentalReferenceEntry":
        """ Returns ExperimentalReferenceEntry constructed from MSONable dict."""
        return cls(composition=d["composition"], temperature=d["temperature"])

    def __repr__(self):
        output = [
            f"{self.__class__.__name__} | {self._formula}",
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
            f"{self.__class__.__name__}"
            f"{self.composition}_"
            f"{self.temperature}".encode("utf-8")
        ).hexdigest()
        return int(data_md5, 16)
