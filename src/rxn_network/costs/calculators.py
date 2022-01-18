"""
A calculator class for determining chemical potential distance of reactions
"""
from itertools import chain, combinations, product
from typing import Callable, List

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
    please reference the paper below:

    Todd, P. K.; McDermott, M. J.; Rom, C. L.; Corrao, A. A.; Denney, J. J.; Dwaraknath,
    S. S.; Khalifah, P. G.; Persson, K. A.;  Neilson, J. R. Selectivity in Yttrium
    Manganese Oxide Synthesis via Local Chemical Potentials in Hyperdimensional Phase
    Space. J. Am. Chem. Soc. 2021, 143 (37), 15185–15194.
    https://doi.org/10.1021/jacs.1c06229.

    """

    def __init__(
        self,
        cpd: ChemicalPotentialDiagram,
        mu_func: str = "sum",
        name: str = "chempot_distance",
    ):
        """
        Args:
            cpd: the chemical potential diagram object
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction. Current options are 1) max, 2) mean, and 3) sum (default).
            name: the data dictionary key with which to store the calculated value.
                Defaults to "chempot_distance".
        """
        self.cpd = cpd
        self._name = name

        if mu_func == "max":
            self._mu_func = max  # type: ignore
        elif mu_func == "mean":
            self._mu_func = np.mean  # type: ignore
        elif mu_func == "sum":
            self._mu_func = sum  # type: ignore

    def calculate(self, rxn: ComputedReaction) -> float:
        """
        Calculates the (aggregate) chemical potential distance in eV/atom. The mu_func parameter
        determines how the individual pairwise interface distances are aggregated into a
        single value describing the overall reaction.

        Args:
            rxn: the ComputedReaction object

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

    @classmethod
    def from_entries(
        cls,
        entries: List[PDEntry],
        mu_func: str = "sum",
        name: str = "chempot_distance",
        **kwargs,
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
            **kwargs: optional kwargs passed to ChemicalPotentialDiagram. By default, passes
                {"default_min_limit": -50}.

        Returns:
            A ChempotDistanceCalculator object
        """
        if not kwargs.get("default_min_limit"):
            kwargs["default_min_limit"] = -50

        cpd = ChemicalPotentialDiagram(entries=entries, **kwargs)
        return cls(cpd, mu_func, name)

    @property
    def mu_func(self) -> Callable:
        """Returns the function used to process the chemical potential distances into a
        single metric."""
        return self._mu_func

    @property
    def name(self) -> str:
        """Returns the name of the data dictionary key where the value is stored"""
        return self._name
