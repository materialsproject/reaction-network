"""
Implements an Entry that looks up NIST pre-tabulated Gibbs free energies
"""
import hashlib
from typing import Any, Dict, List, Optional
from monty.json import MSONable

from pymatgen.core.composition import Composition
from scipy.interpolate import interp1d


class ExperimentalReferenceEntry(MSONable):
    """
    An Entry class for experimental reference data, to be sub-classed for specific data
    sources.  Given a composition, automatically finds the Gibbs free energy of formation, dGf(T) from tabulated
    reference values.
    """

    REFERENCES: Dict = {}

    def __init__(
        self,
        composition: Composition,
        temperature: float,
        data: Optional[dict] = None,
    ):
        """
        Args:
            composition: Composition object (pymatgen).
            temperature: Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
            data: Optional dictionary containing entry data
        """
        self._composition = Composition(composition)
        self.temperature = temperature
        self.data = data if data else {}  # type: Dict[Any, Any]

        formula = self._composition.reduced_formula
        self._validate_temperature(formula, temperature)

        self._energy = self._get_energy(formula, temperature)

        self._formula = formula
        self.name = formula
        self.entry_id = self.__class__.__name__

    def get_new_temperature(
        self, new_temperature: float
    ) -> "ExperimentalReferenceEntry":
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
    def _validate_temperature(cls, formula: str, temperature: float) -> None:
        """Ensure that the temperature is from a valid range."""
        if formula not in cls.REFERENCES:
            raise ValueError(f"{formula} not in reference data!")

        g = cls.REFERENCES[formula]

        if temperature < min(g) or temperature > max(g):
            raise ValueError(
                f"Temperature must be selected from range: {min(g)} K to {max(g)} K"
            )

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
    def composition(self) -> Composition:
        """
        :return: the composition of the entry.
        """
        return self._composition

    @property
    def energy(self) -> float:
        """The energy of the entry, as supplied by the reference tables."""
        return self._energy

    @property
    def energy_per_atom(self) -> float:
        """The energy of the entry, as supplied by the reference tables."""
        return self.energy / self.composition.num_atoms

    @property
    def correction_uncertainty(self) -> float:
        """Uncertainty of experimental data is not supplied."""
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
        return []

    @property
    def is_experimental(self) -> bool:
        """Returns True by default."""
        return True

    @property
    def is_element(self) -> bool:
        """Returns True if the entry is an element."""
        return self.composition.is_element

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
