" Specialized entry to estimate Gibbs Free Energy"
import hashlib
from itertools import combinations
from typing import List, Optional

import numpy as np
from monty.json import MontyDecoder
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry, ConstantEnergyAdjustment
from scipy.interpolate import interp1d

from rxn_network.data import G_ELEMS


class GibbsComputedEntry(ComputedEntry):
    """
    An extension to ComputedEntry which estimates the Gibbs free energy of formation
    of solids using energy adjustments from a model, such as the machine learned
    SISSO descriptor from Bartel et al. (2018).

    WARNING: This descriptor only applies to solids. See NISTEntry for common
    gases (e.g. CO2) where possible.
    """

    def __init__(
            self,
            composition: Composition,
            formation_energy_per_atom: float,
            volume_per_atom: float,
            temperature: float,
            energy_adjustments: Optional[List] = None,
            parameters: dict = None,
            data: dict = None,
            entry_id: object = None,
    ):
        """
        Args:
            composition (Composition): the composition of the structure
            formation_energy_per_atom (float): the formation enthalpy, dHf (T=298 K)
            volume_per_atom (float): the volume per atom in Angstrom^3
        """
        self._composition = Composition(composition)
        self.formation_energy_per_atom = formation_energy_per_atom
        self.volume_per_atom = volume_per_atom
        self.temperature = temperature

        num_atoms = self._composition.num_atoms

        if temperature < 300 or temperature > 2000:
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
                self.gibbs_adjustment(temperature),
                uncertainty=0.05 * num_atoms,  # descriptor has ~50 meV/atom MAD
                name="Gibbs SISSO Correction",
                description=f"Gibbs correction: dGf({self.temperature} K) - dHf (298 K)"
            )
        )

        formation_energy = num_atoms * formation_energy_per_atom

        super().__init__(composition=composition, energy=formation_energy,
                         energy_adjustments=energy_adjustments,
                         parameters=parameters, data=data, entry_id=entry_id)

    def set_temperature(self, new_temperature):
        new_entry_dict = self.as_dict()
        new_entry_dict["temperature"] = new_temperature

        new_entry = self.from_dict(new_entry_dict)
        return new_entry

    def gibbs_adjustment(self, temperature) -> float:
        """
        Returns the difference between the predicted Gibbs formation energy and the
        formation enthalpy at 298 K, i.e., dGf(T) - dHf(298 K). Calculated using
        SISSO descriptor from Bartel et al. (2018) and elemental chemical potentials
        (FactSage).

        Units: eV (not normalized)

        Reference: Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R.,
        Tumas, W., Weimer, A. W., … Holder, A. M. (2018). Physical descriptor for
        the Gibbs energy of inorganic crystalline solids and
        temperature-dependent materials chemistry. Nature Communications, 9(1),
        4168. https://doi.org/10.1038/s41467-018-06682-4

        Args:
            temperature (float): temperature
        Returns:
            float: the correction to Gibbs free energy of formation (eV) from DFT energy
        """
        if self._composition.is_element:
            return 0

        num_atoms = self._composition.num_atoms
        reduced_mass = self._reduced_mass(self._composition)

        return (
                num_atoms
                * self._g_delta_sisso(self.volume_per_atom, reduced_mass, temperature)
                - self._sum_g_i(self._composition, temperature)
        )

    @staticmethod
    def _g_delta_sisso(volume_per_atom, reduced_mass, temp) -> float:
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
                (-2.48e-4 * np.log(volume_per_atom) - 8.94e-5 * reduced_mass /
                 volume_per_atom) * temp
                + 0.181 * np.log(temp)
                - 0.882
        )

    @staticmethod
    def _sum_g_i(composition, temperature) -> float:
        """
        Sum of the stoichiometrically weighted chemical potentials of the elements
        at specified temperature, as acquired from "elements.json".

        Returns:
             float: sum of weighted chemical potentials [eV]
        """
        elems = composition.get_el_amt_dict()

        if temperature % 100 > 0:
            sum_g_i = 0
            for elem, amt in elems.items():
                g_interp = interp1d(
                    [float(t) for t in G_ELEMS.keys()],
                    [g_dict[elem] for g_dict in G_ELEMS.values()],
                )
                sum_g_i += amt * g_interp(temperature)
        else:
            sum_g_i = sum(
                [
                    amt * G_ELEMS[str(temperature)][elem]
                    for elem, amt in elems.items()
                ]
            )

        return sum_g_i

    @staticmethod
    def _reduced_mass(composition) -> float:
        """
        Reduced mass as calculated via Eq. 6 in Bartel et al. (2018), to be used in
        SISSO descriptor equation.

        Returns:
            float: reduced mass (amu)
        """
        reduced_comp = composition.reduced_composition
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

    @classmethod
    def from_structure(cls, structure, formation_energy_per_atom, temperature,
                       **kwargs):
        composition = structure.composition
        volume_per_atom = structure.volume / structure.num_sites
        entry = cls(composition=composition,
                    formation_energy_per_atom=formation_energy_per_atom,
                    volume_per_atom=volume_per_atom,
                    temperature=temperature, **kwargs
                    )
        return entry

    def as_dict(self) -> dict:
        """
        :return: MSONAble dict.
        """
        data = super().as_dict()
        data["volume_per_atom"] = self.volume_per_atom
        data["formation_energy_per_atom"] = self.formation_energy_per_atom
        data["temperature"] = self.temperature
        return data

    @classmethod
    def from_dict(cls, d) -> "GibbsComputedEntry":
        """
        :param d: Dict representation.
        :return: GibbsComputedEntry
        """
        dec = MontyDecoder()
        entry = cls(composition=d["composition"],
                    formation_energy_per_atom=d["formation_energy_per_atom"],
                    volume_per_atom=d["volume_per_atom"],
                    temperature=d["temperature"],
                    energy_adjustments=dec.process_decoded(d["energy_adjustments"]),
                    parameters=d["parameters"],
                    data=d["data"],
                    entry_id=d["entry_id"])
        return entry

    def __repr__(self):
        output = [
            f"GibbsComputedEntry | {self.entry_id} | {self.composition.formula}",
            f"Gibbs Energy ({self.temperature} K) = {self.energy:.4f}",
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
