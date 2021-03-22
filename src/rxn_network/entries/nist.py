" Implements an Entry that looks up NIST pre-tabulated Gibbs free energies "
import hashlib

from monty.json import MontyDecoder
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from scipy.interpolate import interp1d

from rxn_network.data import G_COMPOUNDS, G_GASES


class NISTReferenceEntry(Entry):
    """
    Makes a reference entry using parameters from tabulated NIST values
    #TODO Citation
    """

    REFERENCES = {**G_COMPOUNDS, **G_GASES}.keys()

    def __init__(
        self, composition: Composition, temperature: float = 300,
    ):
        """
        Args:
            temperature (float): Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will
                be interpolated. Defaults to 300 K.
        """
        formula = composition.reduced_formula

        if formula not in NISTReferenceEntry.REFERENCES:
            raise ValueError("Formula must be in NIST Referecne table to initialize")

        if temperature < 300 or temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        self.temperature = temperature
        self._formula = formula

        # TODO: Supplying energy here is a bug of the pymatgen Entry implementation
        super().__init__(composition.reduced_composition, energy=0)

    @property
    def energy(self):
        data = NISTReferenceEntry.REFERENCES[self._formula]
        if self.temperature % 100 > 0:
            g_interp = interp1d([int(t) for t in data.keys()], list(data.values()))
            return g_interp(self.temperature)

        return data[str(self.temperature)]

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
        dec = MontyDecoder()
        new_d = dec.process_decoded(d)
        return cls(**new_d)

    def __repr__(self):
        output = [
            f"NISTReferenceEntry {self._formula} - {self.temperature}",
            f"Gibbs Free Energy (Formation) = {self.energy:.4f}",
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
