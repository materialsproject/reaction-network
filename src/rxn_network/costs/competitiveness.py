"""
A calculator class for determining competitiveness score of a reaction.
"""
import warnings
from typing import List, Optional, Iterable, Dict, Union
from functools import lru_cache

import numpy as np
from pymatgen.core.composition import Composition, Element

from rxn_network.core.calculator import Calculator
from rxn_network.core.cost_function import CostFunction
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)


class CompetitivenessScoreCalculator(Calculator):
    """
    Calculator for determining the competitiveness score (c-score) for a reaction
    (in eV/atom).

    For more information on this specific implementation of the algorithm,
    please cite/reference the paper below:

    Todd, Paul K., McDermott, M.J., et al. “Selectivity in yttrium manganese oxide
    synthesis via local chemical potentials in hyperdimensional phase space.”
    ArXiv:2104.05986 [Cond-Mat], Apr. 2021. arXiv.org, http://arxiv.org/abs/2104.05986
    """

    def __init__(
        self,
        entries: GibbsEntrySet,
        cost_function: CostFunction,
        open_phases: Optional[Iterable[str]] = None,
        open_elem: Optional[Union[str, Element]] = None,
        chempot: float = 0.0,
        use_basic=True,
        use_minimize=False,
        basic_enumerator_kwargs: Optional[Dict] = None,
        minimize_enumerator_kwargs: Optional[Dict] = None,
        name: str = "c_score",
    ):
        """
        Args:
            entries: Iterable of entries to be used for reaction enumeration in
                determining c-score
            cost_function: The cost function used to determine the c-score
            name: the data dictionary key with which to store the calculated value.
        """
        self.entries = entries
        self.cost_function = cost_function
        self.open_phases = open_phases
        self.open_elem = open_elem
        self.chempot = chempot
        self.use_basic = use_basic
        self.use_minimize = use_minimize
        self._name = name
        self.basic_enumerator_kwargs = (
            basic_enumerator_kwargs if basic_enumerator_kwargs else {}
        )
        self.minimize_enumerator_kwargs = (
            minimize_enumerator_kwargs if minimize_enumerator_kwargs else {}
        )

        calcs = ["ChempotDistanceCalculator"]
        if not self.basic_enumerator_kwargs.get("calculators"):
            self.basic_enumerator_kwargs["calculators"] = calcs
        if not self.minimize_enumerator_kwargs.get("calculators"):
            self.minimize_enumerator_kwargs["calculators"] = calcs

    def calculate(self, rxn: ComputedReaction) -> float:
        """
        Calculates the chemical potential distance in eV/atom. The mu_func parameter
        determines how the distance is calculated for the overall reaction.

        Args:
            rxn: the reaction object

        Returns:
            The chemical potential distance of the reaction.
        """
        cost = self.cost_function.evaluate(rxn)

        competing_rxns = self.get_competing_rxns(rxn)
        competing_costs = [self.cost_function.evaluate(r) for r in competing_rxns]

        c_score = self._get_c_score(cost, competing_costs)

        return c_score

    @lru_cache(maxsize=1)
    def get_competing_rxns(self, rxn: ComputedReaction) -> List[ComputedReaction]:
        """
        Returns a list of competing reactions for the given reaction. These are
        enumerated given the settings in the constructor.

        Args:
            rxn: the reaction object

        Returns:
            A list of competing reactions

        """
        precursors = [r.reduced_formula for r in rxn.reactants]

        open_phases = (
            [Composition(p).reduced_formula for p in self.open_phases]
            if self.open_phases
            else None
        )
        if open_phases:
            precursors = list(set(precursors) - set(open_phases))

        enumerators = []
        if self.use_basic:
            kwargs = self.basic_enumerator_kwargs.copy()
            kwargs["precursors"] = precursors
            be = BasicEnumerator(**kwargs)
            enumerators.append(be)

            if open_phases:
                kwargs["open_phases"] = open_phases
                boe = BasicOpenEnumerator(**kwargs)
                enumerators.append(boe)

        if self.use_minimize:
            kwargs = self.minimize_enumerator_kwargs.copy()
            kwargs["precursors"] = precursors
            mge = MinimizeGibbsEnumerator(**kwargs)
            enumerators.append(mge)

            open_elem = self.open_elem
            if not open_elem and open_phases:
                open_comp = Composition(open_phases[0])
                if open_comp.is_element:
                    open_elem = open_comp.elements[0]
                    warnings.warn(f"Using open phase element {open_elem}")

            if open_elem:
                kwargs["open_elem"] = open_elem
                kwargs["mu"] = self.chempot

                mgpe = MinimizeGrandPotentialEnumerator(**kwargs)
                enumerators.append(mgpe)

        rxns = set()
        for e in enumerators:
            rxns.update(e.enumerate(self.entries))

        if rxn in rxns:
            rxns.remove(rxn)

        rxns_updated = ReactionSet.from_rxns(
            rxns, open_elem=open_elem, chempot=self.chempot
        ).get_rxns()

        return rxns_updated

    def _get_c_score(self, cost, competing_costs, scale=10):
        """
        Calculates the c-score for a given reaction.

        Args:
            cost: the cost of the selected reaction
            competing_costs: the costs of all other competing reactions

        Returns:
            The c-score for the reaction
        """
        return np.sum([np.log(1 + np.exp(scale * (cost - c))) for c in competing_costs])

    @property
    def name(self):
        return self._name