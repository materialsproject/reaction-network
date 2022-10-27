"""
This module contains functions for plotting experimental reaction pathway data.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas
from monty.json import MSONable
from pymatgen.analysis.phase_diagram import PhaseDiagram
from scipy.ndimage.filters import median_filter
from tqdm import tqdm

from rxn_network.core.composition import Composition
from rxn_network.entries.entry_set import GibbsEntrySet


class PathwayPlotter(MSONable):
    """
    Helper class for plotting a reaction pathway and the corresponding energy cascade.

    Note:
        This class is far from complete and is not intended yet for general use.
    """

    def __init__(
        self,
        phase_amounts: Dict[str, List[float]],
        temps: List[float],
        apply_smoothing: bool = True,
    ):
        """
        Args:
            phase_amounts: Dicts with format {phase: [amounts]}
            temps: list of temperatures
            apply_smoothing: Whether to smooth the data. Default is True.
        """
        self._phase_amounts = phase_amounts
        self._temps = temps
        self._apply_smoothing = apply_smoothing

        self._formulas = list(phase_amounts.keys())
        self._df = self._get_phase_df()

        self._num_atoms_df = self._get_num_atoms_df()

    def plot_pathway(self):
        """
        Returns a plot of the pathway by calling DataFrame.plot().
        """
        return self.df.plot()

    def plot_energy_cascade(self, entries):
        """
        Returns a plot of the energy cascade given a list of entries.

        Args:
            entries: List of entries or GibbsEntrySet.

        """
        energies = self._get_energies(entries)
        energies_df = pandas.DataFrame(energies).T
        ground_state_energies = energies_df.pop("ground_state")

        gibbs_arr = (
            self.df.values
            * energies_df.values
            / self.num_atoms_df.sum(axis=1).values.reshape(-1, 1)
        )

        g_df = pandas.DataFrame(gibbs_arr, columns=self.df.columns, index=self.df.index)

        total_g = g_df.sum(axis=1)
        total_g = total_g - ground_state_energies

        plot = total_g.plot(backend="matplotlib", style="o")
        plt.xlabel("Temperature (K)", {"size": 11})
        plt.ylabel(r"Gibbs Free Energy, $G$ (eV/atom)", {"size": 11})

        return plot

    def _get_energies(self, entries):
        """
        Interal method: returns a list of energies for each phase
        """
        all_energies = {}
        formulas = self.df.columns.to_list()

        compositions = self.compositions

        for temp in tqdm(self.df.index):
            all_energies[temp] = {}

            gibbs_entries = GibbsEntrySet.from_computed_entries(entries, temp)
            pd = PhaseDiagram(gibbs_entries)

            ground_state_energy = pd.get_hull_energy(compositions[temp])
            all_energies[temp]["ground_state"] = ground_state_energy

            for f in formulas:
                comp = Composition(f).reduced_composition
                energy = (
                    pd.get_form_energy_per_atom(
                        gibbs_entries.get_min_entry_by_formula(comp.formula)
                    )
                    * comp.num_atoms
                )
                all_energies[temp][f] = energy

        return all_energies

    def _get_phase_df(self):
        """
        Returns a dataframe of phase amounts
        """
        phase_df = pandas.DataFrame(self._phase_amounts, index=self._temps)
        if self._apply_smoothing:
            phase_df = phase_df.apply(median_filter, axis=0, size=3)

        return phase_df

    def _get_num_atoms_df(self):
        """
        Returns a dataframe of the number of atoms in each phase
        """
        el_dict = {str(e): [] for e in self.elems}

        for idx, f in enumerate(self.df.columns):
            comp = Composition(f)
            col = self.df.iloc[:, idx]

            for el in el_dict.keys():
                el_dict[el].append((col * comp.get_el_amt_dict()[el]).fillna(0))

        for el in el_dict:
            el_dict[el] = sum(el_dict[el])

        el_df = pandas.DataFrame(el_dict)

        return el_df

    @property
    def elems(self):
        """
        Returns a list of elements in the pathway
        """
        return list({e for f in self.formulas for e in Composition(f).elements})

    @property
    def num_atoms_df(self):
        """
        Returns a dataframe of the number of atoms in each phase
        """
        return self._num_atoms_df

    @property
    def formulas(self):
        """
        Returns a list of formulas in the pathway
        """
        return self._formulas

    @property
    def df(self):
        """
        Returns a dataframe of the pathway
        """
        return self._df

    @property
    def compositions(self):
        """
        Returns the composition of the pathway
        """
        comps = [
            Composition(i).fractional_composition
            for i in self.num_atoms_df.to_dict(  # pylint: disable=not-an-iterable
                "records"
            )
        ]
        return pandas.Series(comps, index=self.num_atoms_df.index)
