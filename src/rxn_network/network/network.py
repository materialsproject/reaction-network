from itertools import chain, combinations

import numpy as np

from typing import List
from monty.json import MSONable
from pymatgen.core.composition import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram

from rxn_network.costs.softplus import Softplus
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.reactions.computed import ComputedReaction

from rxn_network.enumerators.utils import (
    initialize_entry,
    initialize_calculators,
    apply_calculators,
    get_total_chemsys,
    group_by_chemsys,
    filter_entries_by_chemsys,
    get_entry_by_comp,
    get_computed_rxn,
    get_open_computed_rxn,
)

from rxn_network.utils import limited_powerset


class Synthesis(MSONable):
    def __init__(
        self, precursors, initial_amounts, entries, open_formulas=None, threshold=0.03,
            cost_func="softplus", calculators=["ChempotDistanceCalculator"]
    ):
        self.precursors = precursors
        self.initial_amounts = initial_amounts
        self.entries = GibbsEntrySet(entries)
        self.open_formulas = open_formulas
        self.threshold = threshold
        self.cost_func = cost_func
        self.calculators=calculators

        if "ChempotDistanceCalculator" in self.calculators:
            self.entries = self.entries.filter_by_stability(e_above_hull=0.0)
            self.calculators = initialize_calculators(calculators, entries)

        self.combos = list(limited_powerset(entries, 2))
        if open_formulas:
            self.open_entries = [initialize_entry(f, self.entries) for f in
                                 self.open_formulas]

        self.phases = None
        self.amounts = None
        self.interfaces = None
        self.history = {}

    def simulate(self, final_time, dt, temp, max_cost=1.0, dn=0.001):
        self.phases = np.array([e.composition.reduced_formula for e in self.entries])
        self.amounts = np.zeros(len(self.phases))

        for precursor, initial_amt in zip(self.precursors, self.initial_amounts):
            for idx, (p, a) in enumerate(zip(self.phases, self.amounts)):
                if p==precursor:
                    self.amounts[idx] = initial_amt

        current_rxns = {}
        for t in np.arange(0, final_time + dt, dt):
            print(t)
            interfaces = self.get_interfaces()
            cost_func = Softplus(
                temp, ["energy_per_atom", "chempot_distance"], [0.5, 0.5]
            )

            for interface in interfaces:
                if interface.name in current_rxns:
                    continue

                rxns = np.array(interface.react(self.combos))
                costs = np.array([cost_func.evaluate(rxn) for rxn in rxns])

                ind = costs.argsort()
                rxns = rxns[ind]
                costs = costs[ind]

                possible_rxns, possible_costs = rxns[costs<max_cost], \
                                                costs[costs<max_cost]
                if possible_rxns.any():
                    weights = np.exp(-possible_costs)
                    weights = weights / weights.sum()
                    rxn = np.random.choice(possible_rxns, size=1, p=weights)[0]
                    current_rxns[interface.name] = rxn
                    print(rxn)

            self.history[t] = self.state
            self.update_phases(current_rxns.values(), dn)

    def get_interfaces(self):
        return [Interface(get_entry_by_comp(Composition(r1), self.entries),
                          get_entry_by_comp(Composition(r2), self.entries),
                          self.calculators) for r1, r2 in combinations(self.phases_nonzero, 2)]

    def update_phases(self, rxns, dn):
        for rxn in rxns:
            if not rxn:
                continue
            factor = min([self.state[r.reduced_formula] for r in rxn.reactants])
            for comp, coeff in zip(rxn.compositions, rxn.coefficients):
                idx = np.argwhere(self.phases == comp.reduced_formula)
                new_amount = self.amounts[idx] + coeff * factor * dn * np.exp(
                    -rxn.energy_per_atom)
                if new_amount < 0:
                    new_amount = 0
                self.amounts[idx] = new_amount

    @property
    def phases_nonzero(self):
        return [p for p, amt in zip(self.phases, self.amounts) if amt > self.threshold]

    @property
    def state(self):
        return {phase: amt for phase, amt in zip(self.phases, self.amounts)}


class Interface(MSONable):
    def __init__(self, r1, r2, calculators):
        self.r1 = r1
        self.r2 = r2
        self.reactants = [r1, r2]
        self.calculators = calculators

        self.reactant_formulas = sorted([self.r1.composition.reduced_formula,
                                         self.r2.composition.reduced_formula])

    def react(self, combos, open_entries=None):
        #open_entries = set(open_entries)
        reactants = set(self.reactants)

        rxns = []
        for products in combos:
            for r in limited_powerset(reactants, 2):
                r = set(r)
                p = set(products)

                if r & p:  # do not allow repeated phases
                    continue

                rxn = ComputedReaction.balance(r, p)

                if not rxn.balanced or rxn.lowest_num_errors != 0:
                    continue
                rxn = apply_calculators(rxn, self.calculators)
                rxns.append(rxn)

        return rxns

    @property
    def name(self):
        return "-".join(self.reactant_formulas)
