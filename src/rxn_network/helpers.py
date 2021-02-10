"""
This module implements several helper classes for storing and parsing reaction pathway
info in the Reaction Network core module.
"""

import os
from functools import cached_property
from itertools import chain, combinations, zip_longest

import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from rxn_network.reaction import ComputedReaction, Reaction, ReactionError

__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.2"
__email__ = "mcdermott@lbl.gov"
__date__ = "December 20, 2020"


class RxnPathway(MSONable):
    """
    Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes cost of each
    reaction.
    """

    def __init__(self, rxns, costs):
        """
        Args:
            rxns ([ComputedReaction]): list of ComputedReaction objects in pymatgen
                which occur along path.
            costs ([float]): list of corresponding costs for each reaction.
        """
        self._rxns = list(rxns)
        self._costs = list(costs)

        self.total_cost = sum(self._costs)
        self._dg_per_atom = [
            rxn.calculated_reaction_energy
            / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
            for rxn in self._rxns
        ]

    @property
    def rxns(self):
        return self._rxns

    @property
    def costs(self):
        return self._costs

    @property
    def dg_per_atom(self):
        return self._dg_per_atom

    def __repr__(self):
        path_info = ""
        for rxn, dg in zip(self._rxns, self._dg_per_atom):
            path_info += f"{rxn} (dG = {round(dg, 3)} eV/atom) \n"

        path_info += f"Total Cost: {round(self.total_cost,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._rxns))


class BalancedPathway(MSONable):
    """
    Helper class for combining multiple reactions which stoichiometrically balance to
    form a net reaction.
    """

    def __init__(self, rxn_dict, net_rxn, balance=True):
        """
        Args:
            rxn_dict (dict): dictionary of ComputedReaction objects (keys) and their
                associated costs (values).
            net_rxn (ComputedReaction): net reaction to use for stoichiometric
                constraints.
            balance (bool): whether to solve for multiplicities on initialization.
                You might want this to be False if you're balancing the pathways first
                and then initializing the object later, as is done in the pathfinding
                methods.
        """
        self.rxn_dict = rxn_dict
        self.net_rxn = net_rxn
        self.balance = balance
        self.all_rxns = list(self.rxn_dict.keys())
        self.is_balanced = False
        self.multiplicities = None

        self.all_reactants = {
            reactants for rxn in self.rxn_dict.keys() for reactants in rxn.reactants
        }
        self.all_products = {
            products for rxn in self.rxn_dict.keys() for products in rxn.products
        }

        self.all_comp = list(
            self.all_reactants | self.all_products | set(self.net_rxn.all_comp)
        )
        self.net_coeffs = self._get_net_coeffs(net_rxn, self.all_comp)
        self.comp_matrix = self._get_comp_matrix(self.all_comp, self.all_rxns)

        if balance:
            self.is_balanced, multiplicities = self._balance_rxns(
                self.comp_matrix, self.net_coeffs
            )
            self.set_multiplicities(multiplicities)

    def set_multiplicities(self, multiplicities):
        """
        Stores the provided multiplicities (e.g. if solved for outside of object
        initialization).
        Args:
            multiplicities ([float]): list of multiplicities in same order as list of
                all rxns (see self.all_rxns).
        """
        self.multiplicities = {
            rxn: multiplicity
            for (rxn, multiplicity) in zip(self.all_rxns, multiplicities)
        }

    @cached_property
    def total_cost(self):
        if self.is_balanced:
            return sum(
                [self.multiplicities[r] * self.rxn_dict[r] for r in self.rxn_dict]
            ) / sum(list(self.multiplicities.values()))

    @cached_property
    def average_cost(self):
        if self.is_balanced:
            return (
                sum([self.multiplicities[r] * self.rxn_dict[r] for r in self.rxn_dict])
                / sum(list(self.multiplicities.values()))
                / len(self.rxn_dict)
            )

    @staticmethod
    def _balance_rxns(comp_matrix, net_coeffs, tol=1e-6):
        """
        Internal method for balancing a set of reactions to achieve the same
        stoichiometry as a net reaction. Solves for multiplicities of reactions by
        using matrix psuedoinverse and checks to see if solution works.
        Args:
            comp_matrix (np.array): Matrix of stoichiometric coeffs for each reaction.
            net_coeffs (np.array): Vector of stoichiometric coeffs for net reaction.
            tol (float): Numerical tolerance for checking solution.
        Returns:
        """
        comp_pseudo_inverse = np.linalg.pinv(comp_matrix).T
        multiplicities = comp_pseudo_inverse @ net_coeffs

        is_balanced = False

        if (multiplicities < tol).any():
            is_balanced = False
        elif np.allclose(comp_matrix.T @ multiplicities, net_coeffs):
            is_balanced = True

        return is_balanced, multiplicities

    @staticmethod
    def _get_net_coeffs(net_rxn, all_comp):
        """
        Internal method for getting the net reaction coefficients vector.
        Args:
            net_rxn (ComputedReaction): net reaction object.
            all_comp ([Composition]): list of compositions in system of reactions.
        Returns:
            Numpy array which is a vector of the stoichiometric coeffs of net
            reaction and zeros for all intermediate phases.
        """
        return np.array(
            [
                net_rxn.get_coeff(comp) if comp in net_rxn.all_comp else 0
                for comp in all_comp
            ]
        )

    @staticmethod
    def _get_comp_matrix(all_comp, all_rxns):
        """
        Internal method for getting the composition matrix used in the balancing
        procedure.
        Args:
            all_comp ([Composition]): list of compositions in system of reactions.
            all_rxns ([ComputedReaction]): list of all reaction objects.
        Returns:
            Numpy array which is a matrix of the stoichiometric coeffs of each
            reaction in the system of reactions.
        """
        return np.array(
            [
                [
                    rxn.get_coeff(comp) if comp in rxn.all_comp else 0
                    for comp in all_comp
                ]
                for rxn in all_rxns
            ]
        )

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "rxn_dict": self.rxn_dict,
            "net_rxn": self.net_rxn,
            "balance": self.balance,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["rxn_dict"], d["net_rxn"], d["balance"])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return set(self.all_rxns) == set(other.all_rxns)
        else:
            return False

    def __repr__(self):
        rxn_info = ""
        for rxn, cost in self.rxn_dict.items():
            dg_per_atom = rxn.calculated_reaction_energy / sum(
                [rxn.get_el_amount(elem) for elem in rxn.elements]
            )
            rxn_info += f"{rxn} (dG = {round(dg_per_atom,3)} eV/atom) \n"
        rxn_info += (
            f"\nTotal Cost: {round(self.total_cost,3)} | Average Cost: "
            f"{round(self.average_cost,3)}\n\n"
        )

        return rxn_info

    def __hash__(self):
        return hash(frozenset(self.all_rxns))


class CombinedPathway(BalancedPathway):
    """
    Extends the BalancedPathway object to allow for combining of multiple RxnPathway
    objects (instead of ComputedReaction objects themselves).
    """

    def __init__(self, paths, net_rxn):
        """
        Args:
            paths ([RxnPathway]): list of reaction pathway objects.
            net_rxn (ComputedReaction): net reaction object.
        """
        self._paths = paths
        rxn_dict = {
            rxn: cost
            for path in self._paths
            for (rxn, cost) in zip(path.rxns, path.costs)
        }

        super().__init__(rxn_dict, net_rxn)

    @property
    def paths(self):
        return self._paths

    def __repr__(self):
        path_info = ""
        for path in self._paths:
            path_info += f"{str(path)} \n\n"
        path_info += (
            f"Average Cost: {round(self.average_cost,3)} \n"
            f"Total Cost: {round(self.total_cost,3)}"
        )

        return path_info


def powerset(entries, max_num_combos):
    """
    Helper static method for generating combination sets ranging from singular
    length to maximum length specified by max_num_combos.

    Args:
        entries (list/set): list/set of all entry objects to combine
        max_num_combos (int): upper limit for size of combinations of entries

    Returns:
        list: all combination sets
    """
    return chain.from_iterable(
        [
            combinations(entries, num_combos)
            for num_combos in range(1, max_num_combos + 1)
        ]
    )


def react_interface(r1, r2, pd, num_entries, grand_pd=None):
    if grand_pd:
        interface = InterfacialReactivity(
            r1,
            r2,
            grand_pd,
            norm=True,
            include_no_mixing_energy=False,
            pd_non_grand=pd,
            use_hull_energy=True,
        )
    else:
        interface = InterfacialReactivity(
            r1,
            r2,
            pd,
            norm=False,
            include_no_mixing_energy=False,
            pd_non_grand=None,
            use_hull_energy=True,
        )

    entries = pd.all_entries
    rxns = {
        get_computed_rxn(rxn, entries, num_entries)
        for _, _, _, rxn, _ in interface.get_kinks()
    }

    return rxns


def get_computed_rxn(rxn, entries, num_entries):
    reactants = [
        r.reduced_composition
        for r in rxn.reactants
        if not np.isclose(rxn.get_coeff(r), 0)
    ]
    products = [
        p.reduced_composition
        for p in rxn.products
        if not np.isclose(rxn.get_coeff(p), 0)
    ]
    reactant_entries = [get_entry_by_comp(r, entries) for r in reactants]
    product_entries = [get_entry_by_comp(p, entries) for p in products]
    return ComputedReaction(reactant_entries, product_entries, num_entries=num_entries)


def get_entry_by_comp(comp, entry_set):
    possible_entries = filter(
        lambda x: x.composition.reduced_composition == comp, entry_set
    )
    return sorted(possible_entries, key=lambda x: x.energy_per_atom)[0]


def expand_pd(entries):
    """
    Helper method for expanding a single PhaseDiagram into a set of smaller phase
    diagrams, indexed by chemical subsystem. This is an absolutely necessary
    approach when considering chemical systems which contain > ~10 elements,
    due to limitations of the ConvexHull algorithm.
    Args:
        entries ([ComputedEntry]): list of ComputedEntry-like objects for building
            phase diagram.
    Returns:
        Dictionary of PhaseDiagram objects indexed by chemical subsystem string;
        e.g. {"Li-Mn-O": <PhaseDiagram object>, "C-Y": <PhaseDiagram object>, ...}
    """

    pd_dict = dict()

    sorted_entries = sorted(
        entries, key=lambda x: len(x.composition.elements), reverse=True
    )

    for e in sorted_entries:
        for chemsys in pd_dict.keys():
            if set(e.composition.chemical_system.split("-")).issubset(
                chemsys.split("-")
            ):
                break
        else:
            pd_dict[e.composition.chemical_system] = PhaseDiagram(
                list(
                    filter(
                        lambda x: set(x.composition.elements).issubset(
                            e.composition.elements
                        ),
                        entries,
                    )
                )
            )

    return pd_dict


def find_interdependent_rxns(path, precursors, verbose=True):
    precursors = set(precursors)
    interdependent = False
    combined_rxn = None

    rxns = set(path.all_rxns)
    num_rxns = len(rxns)

    if num_rxns == 1:
        return False, None

    for combo in powerset(rxns, num_rxns):
        size = len(combo)
        if any([set(rxn.reactants).issubset(precursors) for rxn in combo]) or size == 1:
            continue
        other_comp = {c for rxn in (rxns - set(combo)) for c in rxn.all_comp}

        unique_reactants = []
        unique_products = []
        for rxn in combo:
            unique_reactants.append(set(rxn.reactants) - precursors)
            unique_products.append(set(rxn.products) - precursors)

        overlap = [False] * size
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                overlapping_phases = unique_reactants[i] & unique_products[j]
                if overlapping_phases and (overlapping_phases not in other_comp):
                    overlap[i] = True

        if all(overlap):
            interdependent = True

            combined_reactants = {c for p in combo for c in p.reactants}
            combined_products = {c for p in combo for c in p.products}
            shared = combined_reactants & combined_products

            combined_reactants = combined_reactants - shared
            combined_products = combined_products - shared
            try:
                combined_rxn = Reaction(
                    list(combined_reactants), list(combined_products)
                )
                if verbose:
                    print(combined_rxn)
            except ReactionError:
                print("Could not combine interdependent reactions!")

    return interdependent, combined_rxn


def softplus(params, weights, t=273):
    """
    Cost function (softplus).

    Args:
        params: list of cost function parameters (e.g. energy)
        weights: list of weights corresponds to parameters of the cost function
        t: temperature (K)

    Returns:
        float: cost (in a.u.)
    """
    weighted_params = np.dot(np.array(params), np.array(weights))
    return np.log(1 + (273 / t) * np.exp(weighted_params))


def get_rxn_cost(
    rxn, cost_function="softplus", temp=273, max_mu_diff=None, most_negative_rxn=None
):
    """Helper method which determines reaction cost/weight.

    Args:
        rxn (CalculatedReaction): the pymatgen CalculatedReaction object.

    Returns:
        float: cost/weight of individual reaction edge
    """
    total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
    energy = rxn.calculated_reaction_energy / total_num_atoms

    if cost_function == "softplus":
        if max_mu_diff:
            params = [energy, max_mu_diff]
            weights = [1, 0.1]
        else:
            params = [energy]
            weights = [1.0]
        weight = softplus(params, weights, t=temp)
    elif cost_function == "piecewise":
        weight = energy
        if weight < most_negative_rxn:
            most_negative_rxn = weight
        if weight >= 0:
            weight = 2 * weight + 1
    elif cost_function == "relu":
        weight = energy
        if weight < 0:
            weight = 0
    else:
        weight = 0

    return weight


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def find_rxn_edges(combos, cost_function, rxn_e_filter, temp, num_entries):
    edges = []
    for combo in combos:
        if not combo:
            continue
        entry = combo[0][0]
        v = combo[0][1]
        other_entry = combo[1][0]
        other_v = combo[1][1]

        phases = entry.entries
        other_phases = other_entry.entries

        if other_phases == phases:
            continue  # do not consider identity-like reactions (e.g. A + B -> A
            # + B)

        # max_mu_diff = None
        # if self._include_chempot_restriction:
        #     product_mu_ranges = [
        #         self._entry_mu_ranges[e] for e in other_phases
        #     ]
        #     product_mu_span = {
        #         elem: (
        #             min(
        #                 [p[elem][0] for p in product_mu_ranges if elem in p]
        #             ),
        #             max(
        #                 [p[elem][1] for p in product_mu_ranges if elem in p]
        #             ),
        #         )
        #         for elem in elems
        #     }
        #
        #     max_mu_diff = -np.inf
        #     for elem in product_mu_span:
        #         mu_diff = (product_mu_span[elem][0]
        #                    - reactant_mu_span[elem][1])
        #         if mu_diff > max_mu_diff:
        #             max_mu_diff = mu_diff

        rxn = ComputedReaction(
            list(phases), list(other_phases), num_entries=num_entries
        )
        if not rxn._balanced:
            continue

        if rxn._lowest_num_errors != 0:
            continue  # remove reaction which has components that
            # change sides or disappear

        total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
        rxn_energy = rxn.calculated_reaction_energy / total_num_atoms

        if rxn_e_filter and rxn_energy > rxn_e_filter:
            continue

        weight = get_rxn_cost(
            rxn, cost_function=cost_function, temp=temp, max_mu_diff=None
        )
        edges.append([v, other_v, weight, rxn, True, False])

    return edges
