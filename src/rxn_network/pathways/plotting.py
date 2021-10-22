"""
Pathway plotting
"""
import pandas
from monty.json import MSONable
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram

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
        self.df = pandas.DataFrame(phase_amounts, index=temps)

    def get_pathway_plot(self):
        return self.df.plot()

    def get_energy_cascade_plot(self, entries):
        return

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
            gibbs_entries = GibbsEntrySet([e.get_new_temperature(temp) for e in
                                           gibbs_entries])

            for f in formulas:
                comp = Composition(f).reduced_composition
                e = gibbs_entries.get_min_entry_by_formula(comp.formula)

                energy = ((e.energy_per_atom
                           + GibbsComputedEntry._sum_g_i(e.composition, e.temperature))
                          * comp.num_atoms
                          )
                energies[f] = energy

            all_energies[temp] = energies
        return all_energies
