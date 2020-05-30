from collections import Counter
from monty.json import MSONable

import matplotlib.pyplot as plt

__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"


class PathwayAnalysis(MSONable):

    def __init__(self, rxn_network, balanced_combined_paths):
        self._balanced_combined_paths = balanced_combined_paths
        self._precursors = rxn_network.precursors
        self._targets = rxn_network.all_targets
        self._intermediate_count = self.count_intermediates()

    def count_intermediates(self):
        """
        Helper method for counting frequency of intermediates.

        Args:
        Returns:
        """
        intermediate_count = Counter()
        precursors_and_targets = {entry.composition.reduced_composition for entry in self._precursors | self._targets}
        for combined_path in self._balanced_combined_paths:
            for comp in combined_path.all_comp:
                if comp.reduced_composition not in precursors_and_targets:
                    intermediate_count[comp] += 1

        self._intermediate_count = intermediate_count
        return intermediate_count

    def plot_intermediate_freq(self):
        """
        Plot frequency of intermediates (using matplotlib).

        Args:
        Returns:
        """
        if not self._intermediate_count:
            self._intermediate_count = self.count_intermediates()

        plt.xticks(rotation='vertical')
        plt.title("Frequency of intermediates in balanced reaction pathways")
        plt.ylabel("Number of appearances")
        return plt.bar([str(k.reduced_formula) for k in self.intermediate_count.keys()], self.intermediate_count.values())

    @property
    def intermediate_count(self):
        return self._intermediate_count

    @property
    def balanced_combined_paths(self):
        return self._balanced_combined_paths