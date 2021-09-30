"""
A calculator class for determining chemical potential distance of reactions
"""
from itertools import chain, combinations, product
from typing import List, Optional

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry

from rxn_network.core.calculator import Calculator
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram


class ChempotDistanceCalculator(Calculator):
    """
    Calculator for determining the "chemical potential distance" for a reaction
    (in eV/atom).

    For more information on this specific implementation of the algorithm,
    please cite/reference the paper below:

    Todd, Paul K., McDermott, M.J., et al. “Selectivity in yttrium manganese oxide
    synthesis via local chemical potentials in hyperdimensional phase space.”
    ArXiv:2104.05986 [Cond-Mat], Apr. 2021. arXiv.org, http://arxiv.org/abs/2104.05986
    """

    def __init__(
        self,
        cpd: ChemicalPotentialDiagram,
        mu_func: str = "sum",
        name: str = "chempot_distance",
    ):
        """
        Args:
            cpd: the chemical potential diagram
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction.
            name: the data dictionary key with which to store the calculated value.
        """
        self.cpd = cpd
        self.name = name

        if mu_func == "max":
            self._mu_func = np.max  # type: ignore
        elif mu_func == "mean":
            self._mu_func = np.mean  # type: ignore
        elif mu_func == "sum":
            self._mu_func = np.sum  # type: ignore

    def calculate(self, rxn: ComputedReaction) -> float:
        """
        Calculates the chemical potential distance in eV/atom. The mu_func parameter
        determines how the distance is calculated for the overall reaction.

        Args:
            rxn: the reaction object

        Returns:
            The chemical potential distance of the reaction.
        """
        combos = chain(
            product(rxn.reactant_entries, rxn.product_entries),
            combinations(rxn.product_entries, 2),
        )
        distances = [
            self.cpd.shortest_domain_distance(
                combo[0].composition.reduced_formula,
                combo[1].composition.reduced_formula,
            )
            for combo in combos
        ]

        distance = float(self._mu_func(distances))
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
        entries: List[PDEntry],
        mu_func: str = "sum",
        name: str = "chempot_distance",
        **kwargs
    ) -> "ChempotDistanceCalculator":
        """
        Convenience constructor which first builds the ChemicalPotentialDiagram
        object from a list of entry objects.

        Args:
            entries: entry objects used to build the ChemicalPotentialDiagram
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction.
            name: the data dictionary key by which to store the calculated value,
                defaults to "chempot_distance"
            **kwargs: optional kwargs passed to ChemicalPotentialDiagram; defaults to
                "default_min_limit"=-50

        Returns:
            A ChempotDistanceCalculator object
        """
        if not kwargs.get("default_min_limit"):
            kwargs["default_min_limit"] = -50

        cpd = ChemicalPotentialDiagram(entries=entries, **kwargs)
        return cls(cpd, mu_func, name)

    @property
    def mu_func(self):
        """Returns the function used to process the interfacial mu distances"""
        return self._mu_func
