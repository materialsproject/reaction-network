"""
Pathway plotting
"""
import pandas
from monty.json import MSONable
from pymatgen.core.composition import Composition

from rxn_network.entries import GibbsEntrySet, GibbsComputedEntry


class PathwayPlotter(MSONable):
    """
    Pathway plotter
    """

    def __init__(self, phase_amounts, temps):
        """
        phase_amounts: dict of {phase: amount}
        temps: list of temperatures
        """
        self._phase_amounts = phase_amounts
        self._temps = temps

        self.formulas = list(phase_amounts.keys())
        self.df = pandas.DataFrame(phase_amounts, index=temps)
        self.num_atoms_df = self._get_num_atoms_df()

    def get_pathway_plot(self):
        """
        Returns a plot of the pathway

        Args:
            entries: list of entries
        """

        return self.df.plot()

    def get_energy_cascade_plot(self, entries):
        """
        Returns a plot of the energy cascade

        """
        energies = self.get_energies(entries)
        energies_df = pandas.DataFrame(energies)

        gibbs_arr = (
            self.df.values
            * energies_df.T.values
            / self.num_atoms_df.sum(axis=1).values.reshape(-1, 1)
        )

        df = pandas.DataFrame(gibbs_arr, columns=self.df.columns, index=self.df.index)

        return df.plot()

    def get_energies(self, entries):
        """
        Returns a list of energies for each phase
        """
        all_energies = dict()

        initial_temp = self.df.index[0]
        formulas = self.df.columns.to_list()

        gibbs_entries = GibbsEntrySet.from_entries(entries, initial_temp)
        gibbs_entries = [gibbs_entries.get_min_entry_by_formula(f) for f in formulas]

        for temp in self.df.index:
            energies = {f: None for f in formulas}
            gibbs_entries = GibbsEntrySet(
                [e.get_new_temperature(temp) for e in gibbs_entries]
            )

            for f in formulas:
                comp = Composition(f).reduced_composition
                e = gibbs_entries.get_min_entry_by_formula(comp.formula)

                energy = (
                    e.energy_per_atom
                    + GibbsComputedEntry._sum_g_i(e.composition, e.temperature)
                ) * comp.num_atoms
                energies[f] = energy

            all_energies[temp] = energies

        return all_energies

    def _get_num_atoms_df(self):
        el_dict = {e: [] for e in self.elems}

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
        return list({e for f in self.formulas for e in Composition(f).elements})
