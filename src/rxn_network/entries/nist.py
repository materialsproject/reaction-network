" Implements an Entry that looks up NIST pre-tabulated Gibbs free energies "
import hashlib

from monty.json import MontyDecoder
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from scipy.interpolate import interp1d

from rxn_network.data import G_COMPOUNDS, G_GASES


class NISTReferenceEntry(Entry):
    """
    An Entry class for NIST-JANAF experimental reference data. Given a composition,
    the Gibbs free energy of formation, dGf(T) from tabulated values (G_GASES,
    G_COMPOUNDS).

    Reference:

    Malcolm W. Chase Jr. NIST-JANAF thermochemical tables. Fourth edition.
    Washington, DC : American Chemical Society;  New York : American Institute of
    Physics for the National Institute of Standards and Technology, 1998.

    """
    REFERENCES = {**G_COMPOUNDS, **G_GASES}

    def __init__(
        self, composition: Composition, temperature: float = 300,
    ):
        """
        Args:
            composition (Composition): pymatgen Composition object
            temperature (float): Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will be
                interpolated. Defaults to 300 K.
        """
        composition = Composition(composition)
        formula = composition.reduced_formula

        if formula not in NISTReferenceEntry.REFERENCES:
            raise ValueError("Formula must be in NIST-JANAF thermochemical tables")

        if temperature < 300 or temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        energy = self.get_nist_energy(formula, temperature)

        self.temperature = temperature
        self._formula = formula
        self.name = formula
        self.entry_id = "NISTReferenceEntry"

        super().__init__(composition.reduced_composition, energy)

    @property
    def energy(self) -> float:
        """
        :return: the energy of the entry.
        """
        return self._energy

    @staticmethod
    def get_nist_energy(formula: str, temperature: float):
        """
        Convenience method for accessing and interpolating NIST-JANAF data.

        Args:
            formula:
            temperature:

        Returns:

        """
        data = NISTReferenceEntry.REFERENCES[formula]
        if temperature % 100 > 0:
            g_interp = interp1d([float(t) for t in data.keys()], list(data.values()))
            return g_interp(temperature)[()]

        return data[str(temperature)]

    @property
    def correction_uncertainty(self) -> float:
        """
        Returns:
            float: the uncertainty of the energy adjustments applied to the entry, in eV
        """
        return 0

    @property
    def correction_uncertainty_per_atom(self) -> float:
        """
        Returns:
            float: the uncertainty of the energy adjustments applied to the entry,
                normalized by atoms (units of eV/atom)
        """
        return 0

    def as_dict(self) -> dict:
        """
        :return: MSONAble dict.
        """
        data = super().as_dict()
        data["temperature"] = self.temperature
        return data

    @classmethod
    def from_dict(cls, d) -> "NISTReferenceEntry":
        """
        :param d: Dict representation.
        :return: NISTReferenceEntry
        """
        return cls(composition=d["composition"],
                   temperature=d["temperature"])

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
