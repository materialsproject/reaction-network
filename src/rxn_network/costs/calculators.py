"""A calculator class for determining chemical potential distance of reactions."""
from __future__ import annotations

from itertools import chain, combinations, product
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from rxn_network.costs.base import Calculator
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram

if TYPE_CHECKING:
    from pymatgen.analysis.phase_diagram import PDEntry

    from rxn_network.reactions.computed import ComputedReaction
    from rxn_network.reactions.hull import InterfaceReactionHull


class ChempotDistanceCalculator(Calculator):
    """Calculator for determining the aggregated "chemical potential distance" for a
    reaction, in units of eV/atom. This is a reaction selectivity metric based on an
    aggregation function applied to the chemical potential differences of reactant-
    product and product-product interfaces in a reaction.

    If you use this cost metric, please cite the following work:

        Todd, P. K.; McDermott, M. J.; Rom, C. L.; Corrao, A. A.; Denney, J. J.;
        Dwaraknath, S. S.; Khalifah, P. G.; Persson, K. A.;  Neilson, J. R. Selectivity
        in Yttrium Manganese Oxide Synthesis via Local Chemical Potentials in
        Hyperdimensional Phase Space. J. Am. Chem. Soc. 2021, 143 (37), 15185-15194.
        https://doi.org/10.1021/jacs.1c06229.
    """

    def __init__(
        self,
        cpd: ChemicalPotentialDiagram,
        mu_func: Callable | str = "sum",
        name: str = "chempot_distance",
    ):
        """
        Args:
            cpd: the chemical potential diagram for the phase space in which the
                reaction(s) exist
            mu_func: the function (or string name of the function) used to aggregate the
                interfacial chemical potential distances into a single value describing
                the whole reaction. Current options are 1) max, 2) mean, and 3) sum
                (default).
            name: the data dictionary key with which to store the calculated value.
                Defaults to "chempot_distance".
        """
        self.cpd = cpd
        self.name = name

        if mu_func == "max":
            self.mu_func = max
        elif mu_func == "mean":
            self.mu_func = np.mean  # type: ignore
        elif mu_func == "sum":
            self.mu_func = sum  # type: ignore
        elif isinstance(mu_func, str):
            raise ValueError(
                "Provided mu_func name is not a known function; please provide the"
                " function directly."
            )

        self._open_elems = set()
        if cpd.entries[0].__class__.__name__ == "GrandPotPDEntry":
            self._open_elems = set(cpd.entries[0].chempots.keys())

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates the aggregate chemical potential distance in eV/atom. The mu_func
        parameter determines how the individual pairwise interface distances are
        aggregated into a single value describing the overall reaction. When mu_func =
        "sum" (i.e., the default setting), the total chemical potential distance is
        returned.

        Args:
            rxn: the ComputedReaction object

        Returns:
            The aggregate chemical potential distance of the reaction in eV/atom.
        """
        reactant_entries = rxn.reactant_entries
        product_entries = rxn.product_entries

        if hasattr(rxn, "grand_entries"):
            reactant_entries = [
                e
                for e, c in zip(rxn.grand_entries, rxn.coefficients)
                if c < 0 and e.__class__.__name__ == "GrandPotPDEntry"
            ]
            product_entries = [
                e
                for e, c in zip(rxn.grand_entries, rxn.coefficients)
                if c > 0 and e.__class__.__name__ == "GrandPotPDEntry"
            ]
        combos = chain(
            product(reactant_entries, product_entries),
            combinations(product_entries, 2),
        )
        distances = [
            self.cpd.shortest_domain_distance(
                combo[0].composition.reduced_formula,
                combo[1].composition.reduced_formula,
                offset=self.cpd.get_offset(combo[0]) + self.cpd.get_offset(combo[1]),
            )
            for combo in combos
        ]

        distance = round(float(self.mu_func(distances)), 5)
        return distance

    @classmethod
    def from_entries(
        cls,
        entries: list[PDEntry],
        mu_func: Callable[[Iterable[float]], float] | str = "sum",
        name: str = "chempot_distance",
        **kwargs,
    ) -> ChempotDistanceCalculator:
        """Convenience constructor which first builds the ChemicalPotentialDiagram
        object from a list of entry objects.

        Args:
            entries: entry objects used to build the ChemicalPotentialDiagram
            mu_func: the name of the function used to process the interfacial
                chemical potential distances into a single value describing the whole
                reaction.
            name: the data dictionary key by which to store the calculated value,
                defaults to "chempot_distance"
            **kwargs: optional kwargs passed to ChemicalPotentialDiagram.

        Returns:
            A ChempotDistanceCalculator object
        """

        cpd = ChemicalPotentialDiagram(entries=entries, **kwargs)
        return cls(cpd, mu_func, name)


class PrimaryCompetitionCalculator(Calculator):
    """Calculator for determining the primary competition, C_1, for a reaction (units:
    eV/atom).

    If you use this selectivity metric in your work, please cite the following work:

        McDermott, M. J.; McBride, B. C.; Regier, C.; Tran, G. T.; Chen, Y.; Corrao, A.
        A.; Gallant, M. C.; Kamm, G. E.; Bartel, C. J.; Chapman, K. W.; Khalifah, P. G.;
        Ceder, G.; Neilson, J. R.; Persson, K. A. Assessing Thermodynamic Selectivity of
        Solid-State Reactions for the Predictive Synthesis of Inorganic Materials. arXiv
        August 22, 2023. https://doi.org/10.48550/arXiv.2308.11816.
    """

    def __init__(
        self,
        irh: InterfaceReactionHull,
        name: str = "primary_competition",
    ):
        """
        Args:
            irh: the interface reaction hull containing the target reaction and all
                competing reactions.
            name: the data dictionary key with which to store the calculated value.
                Defaults to "primary_competition".
        """
        self.irh = irh
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates the competitiveness score for a given reaction by enumerating
        competing reactions, evaluating their cost with the supplied cost function, and
        then using the c-score formula, i.e. the _get_c_score() method, to determine the
        competitiveness score.

        Args:
            rxn: the ComputedReaction object to be evaluated

        Returns:
            The C1 score
        """
        return self.irh.get_primary_competition(rxn)


class SecondaryCompetitionCalculator(Calculator):
    """Calculator for determining the secondary competition, C_2, for a reaction (in
    eV/atom).

    If you use this selectivity metric in your work, please cite the following work:

        McDermott, M. J.; McBride, B. C.; Regier, C.; Tran, G. T.; Chen, Y.; Corrao, A.
        A.; Gallant, M. C.; Kamm, G. E.; Bartel, C. J.; Chapman, K. W.; Khalifah, P. G.;
        Ceder, G.; Neilson, J. R.; Persson, K. A. Assessing Thermodynamic Selectivity of
        Solid-State Reactions for the Predictive Synthesis of Inorganic Materials. arXiv
        August 22, 2023. https://doi.org/10.48550/arXiv.2308.11816.
    """

    def __init__(
        self,
        irh: InterfaceReactionHull,
        name: str = "secondary_competition",
    ):
        """
        Args:
            irh: the interface reaction hull containing the target reaction and all
                competing reactions.
            name: the data dictionary key with which to store the calculated value.
                Defaults to "secondary_competition".
        """
        self.irh = irh
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates the secondary competition per its implementation in the
        InterfaceReactionHull class.

        Args:
            rxn: the ComputedReaction object to be evaluated

        Returns:
            The C2 score
        """
        return self.irh.get_secondary_competition(rxn)


class SecondaryCompetitionWithEhullCalculator(Calculator):
    """
    WARNING: this is an alternative calculator for secondary competition (C_2) that
    includes the energy above hull of the target reaciton. It should only be used for
    testing purposes.
    """

    def __init__(
        self,
        irh: InterfaceReactionHull,
        name: str = "secondary_competition_with_ehull",
    ):
        """
        Args:
            irh: the interface reaction hull containing the target reaction and all
                competing reactions.
            name: the data dictionary key with which to store the calculated value.
                Defaults to "secondary_competition_with_ehull".
        """
        self.irh = irh
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates the secondary competition with e_hull per its implementation in
        the InterfaceReactionHull class.

        Args:
            rxn: the ComputedReaction object to be evaluated

        Returns:
            The C2 + e_hull score
        """
        return self.irh.get_secondary_competition(rxn, include_e_hull=True)


class SecondaryCompetitionMaxCalculator(Calculator):
    """
    WARNING: this is an alternative calculator for secondary competition (C_2) that
    defaults to calculation of the maximum secondary reaction energy. It should only be
    used for testing purposes.
    """

    def __init__(
        self,
        irh: InterfaceReactionHull,
        name: str = "secondary_competition_max",
    ):
        """
        Args:
            irh: the interface reaction hull containing the target reaction and all
                competing reactions.
            name: the data dictionary key with which to store the calculated value.
                Defaults to "secondary_competition_max".
        """
        self.irh = irh
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates the secondary competition with max energies.

        Args:
            rxn: the ComputedReaction object to be evaluated

        Returns:
            The C2 score using max energies.
        """
        return self.irh.get_secondary_competition_max_energy(rxn)


class SecondaryCompetitionAreaCalculator(Calculator):
    """
    WARNING: this is an alternative calculator for secondary competition (C_2) that
    defaults to calculation of the area of the enclosed hull. It should only be used for
    testing purposes and is quite unstable.
    """

    def __init__(
        self,
        irh: InterfaceReactionHull,
        name: str = "secondary_competition_area",
    ):
        """
        Args:
            irh: the interface reaction hull containing the target reaction and all
                competing reactions.
            name: the data dictionary key with which to store the calculated value.
                Defaults to "secondary_competition_area".
        """
        self.irh = irh
        self.name = name

    def calculate(self, rxn: ComputedReaction) -> float:
        """Calculates an area for secondary competition.

        Args:
            rxn: the ComputedReaction object to be evaluated

        Returns:
            Secondary competition as represented by area.
        """
        return self.irh.get_secondary_competition_area(rxn)
