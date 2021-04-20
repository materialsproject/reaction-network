" A calculator class for determining chemical potential distance of reactions "

from typing import List, Optional
from itertools import product
import numpy as np

from rxn_network.core import Calculator
from rxn_network.reactions import ComputedReaction
from rxn_network.thermo.chempot_diagram import ChempotDiagram

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram


class ChempotDistanceCalculator(Calculator):
    """
    Calculator for determining the "chemical potential distance" for a reaction
    (in eV/atom).
    """

    def __init__(
        self,
        cpd: ChempotDiagram,
        mu_func: Optional[str] = "max",
        name: Optional[str] = "chempot_distance",
    ):
        """
        Args:
            cpd: the chemical potential diagram
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction.
            name: the data dictionary key by which to store the calculated value.
        """
        self.cpd = cpd
        self.name = name

        if mu_func == "max":
            self._mu_func = max
        elif mu_func == "mean":
            self._mu_func = np.mean

    def calculate(self, rxn: ComputedReaction) -> float:
        """
        Calculates the chemical potential distance in eV/atom. The mu_func parameter
        determines how the distance is calculated for the overall reaction.

        Args:
            rxn: the reaction object

        Returns:
            The chemical potential distance of the reaction.
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
        Decorates the reaction (in place) with the chemical potential distance by
        storing the value within the reaction's data dictionary.

        Args:
            rxn: The reaction object.

        Returns:
            The reaction object, modified in place
        """
        if not rxn.data:
            rxn.data = {}

        rxn.data[self.name] = self.calculate(rxn)

        return rxn

    @classmethod
    def from_entries(
        cls,
        entries: List[ComputedEntry],
        mu_func: Optional[str] = "max",
        cpd_kws: Optional[dict] = {"default_limit": -50},
        name: Optional[str] = "chempot_distance",
    ):
        """
        Convenience constructor which first builds the ChempotDiagram
        object from a list of entry objects.

        Args:
            entries: entry objects used to build the ChempotDiagram
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction.
            cpd_kws: optional kwargs passed to the ChempotDiagram constructor.
                Default kwarg is default_limit = -50.
            name: the data dictionary key by which to store the calculated value,
                defaults to "chempot_distance"

        Returns:
            A ChempotDistanceCalculator object
        """
        pd = PhaseDiagram(entries)
        cpd = ChempotDiagram(pd, **cpd_kws)
        return cls(cpd, mu_func, name)

    @property
    def mu_func(self):
        " Returns the function used to process the interfacial mu distances "
        return self._mu_func
