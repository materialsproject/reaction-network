from typing import List

from itertools import product

import numpy as np

from rxn_network.core import Calculator
from rxn_network.reactions import ComputedReaction
from rxn_network.thermo import ChempotDiagram


class ChempotDistanceCalculator(Calculator):
    """
    Calculates the chemical potential distance for a reaction (in eV/atom).
    """
    def __init__(self, cpd: ChempotDiagram, name="chempot_distance"):
        self.cpd = cpd
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        distances = [
            self.cpd.shortest_domain_distance(
                combo[0].composition.reduced_formula,
                combo[1].composition.reduced_formula,
            )
            for combo in product(rxn.reactant_entries, rxn.product_entries)
        ]

        max_distance = max(distances)
        return max_distance

    def decorate(self, rxn: ComputedReaction) -> ComputedReaction:
        if rxn.data:
            data = rxn.data.copy()
        else:
            data = {}
        data[self.name] = self.calculate(rxn)
        return ComputedReaction(rxn.entries, rxn.coefficients, data=data,
                                lowest_num_errors=rxn.lowest_num_errors)