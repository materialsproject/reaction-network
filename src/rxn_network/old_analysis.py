from collections import Counter

import matplotlib.pyplot as plt
from monty.json import MSONable

__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.2"
__email__ = "mcdermott@lbl.gov"
__date__ = "December 20, 2020"


class PathwayAnalysis(MSONable):
    """
    A convenience class for performing data analytic operations on BalancedPathway
    objects generated during the reaction pathway pathfinding process.
    """

    def __init__(self, precursors, targets, balanced_combined_paths):
        """
        Args:
            rxn_network (ReactionNetwork): the ReactionNetwork object.
            balanced_combined_paths ([BalancedPathway]): list of reaction pathway
                objects which have been successfully balanced.
        """
        self._balanced_combined_paths = balanced_combined_paths
        self._precursors = set(precursors)
        self._targets = set(targets)
        self._intermediate_count = self.count_intermediates()

    def count_intermediates(self):
        """
        Method for counting the frequency of intermediate phases which show up along
        the provided pathways. May provide a metric for understanding the
        flexibility (compositionally-speaking) to which an intermediate phase can be
        used to access a target product. Note: there is no existing evidence that
        intermediate phases with higher counts have any scientific significance
        (this may just be due to bias).

        Returns:
            Counter object with counts of intermediate phases
        """
        intermediate_count = Counter()
        precursors_and_targets = {
            entry.composition.reduced_composition
            for entry in self._precursors | self._targets
        }
        for combined_path in self._balanced_combined_paths:
            for comp in combined_path.all_comp:
                if comp.reduced_composition not in precursors_and_targets:
                    intermediate_count[comp] += 1

        self._intermediate_count = intermediate_count
        return intermediate_count

    def plot_intermediate_freq(self):
        """
        Plot frequency of intermediates (using matplotlib).

        Returns:
            Bar plot object created using matplotlib.
        """
        if not self._intermediate_count:
            self._intermediate_count = self.count_intermediates()

        plt.xticks(rotation="vertical")
        plt.title("Frequency of intermediates in balanced reaction pathways")
        plt.ylabel("Number of appearances")
        return plt.bar(
            [str(k.reduced_formula) for k in self.intermediate_count.keys()],
            self.intermediate_count.values(),
        )

    @property
    def intermediate_count(self):
        return self._intermediate_count

    @property
    def balanced_combined_paths(self):
        return self._balanced_combined_paths
