import hashlib
from itertools import combinations
from typing import Optional, List
import numpy as np
from monty.json import MontyDecoder
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.computed_entries import ConstantEnergyAdjustment
from scipy.interpolate import interp1d

from rxn_network.data import G_ELEMS


class GibbsComputedEntry(ComputedEntry):
    """
    An extension to ComputedEntry which includes the estimated Gibbs
    free energy of formation via a machine-learned model.

    WARNING: This descriptor only applies to solids. See NISTEntry for common
    gases (e.g. CO2) where possible.
    """

    def __init__(
        self,
        volume_per_atom: float,
        temperature: float = 300,
        energy_adjustments: Optional[List] = None,
        **kwargs,
    ):
        """
        Args:
            temperature (float): Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will
                be interpolated. Defaults to 300 K.
            volume_per_atom (float): the volume per atom in Angstrom^3
        """
        self.volume_per_atom = volume_per_atom
        self._temperature = temperature

        if self.temperature < 300 or self.temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        if energy_adjustments is not None:
            energy_adjustments = [
                adjustment
                for adjustment in energy_adjustments
                if adjustment.name != "Gibbs SISSO Correction"
            ]
        else:
            energy_adjustments = []

        energy_adjustments.append(
            ConstantEnergyAdjustment(
                self.gf_sisso(),
                name="Gibbs SISSO Correction",
                description=f"Correction from the SISSO description of G^delta by Bartel et al at T={self.temperature}K",
            )
        )

        super().__init__(energy_adjustments=energy_adjustments, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def change_temperature(self, temperature: float):
        " Update the temperature in this SISSO Gibbs corrected ComputedEntry "
        self.energy_adjustments = [
            adjustment
            for adjustment in self.energy_adjustments
            if adjustment.name != "Gibbs SISSO Correction"
        ]

        self._temperature = temperature

        self.energy_adjustments.append(
            ConstantEnergyAdjustment(
                self.gf_sisso(),
                name="Gibbs SISSO Correction",
                description=f"Correction from the SISSO description of G^delta by Bartel et al at T={self.temperature}K",
            )
        )

    def gf_sisso(self) -> float:
        """
        Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel
        et al. (2018). Units: eV (not normalized)

        Reference: Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R.,
        Tumas, W., Weimer, A. W., … Holder, A. M. (2018). Physical descriptor for
        the Gibbs energy of inorganic crystalline solids and
        temperature-dependent materials chemistry. Nature Communications, 9(1),
        4168. https://doi.org/10.1038/s41467-018-06682-4

        Returns:
            float: the correction to Gibbs free energy of formation (eV) from DFT energy
        """
        if self.composition.is_element:
            return 0

        reduced_mass = self._reduced_mass()

        return (
            self.composition.num_atoms
            * self._g_delta_sisso(self.volume_per_atom, reduced_mass, self.temperature)
            - self._sum_g_i()
        )

    def _sum_g_i(self) -> float:
        """
        Sum of the stoichiometrically weighted chemical potentials of the elements
        at specified temperature, as acquired from "g_els.json".

        Returns:
             float: sum of weighted chemical potentials [eV]
        """
        elems = self.composition.get_el_amt_dict()

        if self.temperature % 100 > 0:
            sum_g_i = 0
            for elem, amt in elems.items():
                g_interp = interp1d(
                    [float(t) for t in G_ELEMS.keys()],
                    [g_dict[elem] for g_dict in G_ELEMS.values()],
                )
                sum_g_i += amt * g_interp(self.temperature)
        else:
            sum_g_i = sum(
                [
                    amt * G_ELEMS[str(self.temperature)][elem]
                    for elem, amt in elems.items()
                ]
            )

        return sum_g_i

    def _reduced_mass(self) -> float:
        """
        Reduced mass as calculated via Eq. 6 in Bartel et al. (2018)

        Returns:
            float: reduced mass (amu)
        """
        reduced_comp = self.composition.reduced_composition
        num_elems = len(reduced_comp.elements)
        elem_dict = reduced_comp.get_el_amt_dict()

        denominator = (num_elems - 1) * reduced_comp.num_atoms

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j)

        reduced_mass = (1 / denominator) * mass_sum

        return reduced_mass

    @staticmethod
    def _g_delta_sisso(vol_per_atom, reduced_mass, temp) -> float:
        """
        G^delta as predicted by SISSO-learned descriptor from Eq. (4) in
        Bartel et al. (2018).

        Args:
            vol_per_atom (float): volume per atom [Å^3/atom]
            reduced_mass (float) - reduced mass as calculated with pair-wise sum formula
                [amu]
            temp (float) - Temperature [K]

        Returns:
            float: G^delta
        """

        return (
            (-2.48e-4 * np.log(vol_per_atom) - 8.94e-5 * reduced_mass / vol_per_atom)
            * temp
            + 0.181 * np.log(temp)
            - 0.882
        )

    def as_dict(self) -> dict:
        """
        :return: MSONAble dict.
        """
        d = super().as_dict()
        d["volume_per_atom"] = self.volume_per_atom
        d["temperature"] = self.temperature
        return d

    @classmethod
    def from_dict(cls, d) -> "GibbsComputedEntry":
        """
        :param d: Dict representation.
        :return: GibbsComputedEntry
        """
        dec = MontyDecoder()
        new_d = dec.process_decoded(d)
        return cls(**new_d)

    def __repr__(self):
        output = [
            f"GibbsComputedEntry {self.entry_id} - {self.composition.formula}",
            f"Gibbs Free Energy (Formation) = {self.energy:.4f}",
        ]
        return "\n".join(output)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                (self.entry_id == other.entry_id)
                and (self.temperature == other.temperature)
                and (self.composition == other.composition)
                and (self.energy == other.energy)
            )
        return False

    def __hash__(self):
        data_md5 = hashlib.md5(
            "GibbsComputedEntry"
            f"{self.composition}_"
            f"{self.energy}_{self.entry_id}_"
            f"{self.temperature}".encode("utf-8")
        ).hexdigest()
        return int(data_md5, 16)
