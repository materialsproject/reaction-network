from typing import List

from itertools import product

import numpy as np

from rxn_network.core import Calculator
from rxn_network.reactions import ComputedReaction
from rxn_network.thermo.chempot_diagram import ChempotDiagram

from pymatgen.analysis.phase_diagram import PhaseDiagram


class ChempotDistanceCalculator(Calculator):
    """
    Calculates the chemical potential distance for a reaction (in eV/atom).
    """

    def __init__(self, cpd: ChempotDiagram, distance_type="max",
                 name="chempot_distance"):
        self.cpd = cpd
        self.name = name
        self.type = distance_type

        if distance_type=="max":
            self._mu_func = max
        elif distance_type=="mean":
            self._mu_func = np.mean

    def calculate(self, rxn: ComputedReaction) -> float:
        """

        Args:
            rxn:

        Returns:

        """
        distances = [
            self.cpd.shortest_domain_distance(
                combo[0].composition.reduced_formula,
                combo[1].composition.reduced_formula,
            )
            for combo in product(rxn.reactant_entries, rxn.product_entries)
        ]

        distance = self._mu_func(distances)
        return distance

    def decorate(self, rxn: ComputedReaction) -> ComputedReaction:
        """

        Args:
            rxn:

        Returns:

        """
        if rxn.data:
            data = rxn.data.copy()
        else:
            data = {}
        data[self.name] = self.calculate(rxn)
        return ComputedReaction(rxn.entries, rxn.coefficients, data=data,
                                lowest_num_errors=rxn.lowest_num_errors)

    @classmethod
    def from_entries(cls, entries, distance_type="max", name="chempot_distance"):
        """

        Args:
            entries:
            distance_type:
            name:

        Returns:

        """
        pd = PhaseDiagram(entries)
        cpd = ChempotDiagram(pd, default_limit=-50)
        return cls(cpd, distance_type, name)
