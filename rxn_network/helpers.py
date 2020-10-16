"""
This module implements several helper classes for storing and parsing reaction pathway
info in the Reaction Network core module.
"""

import os
from itertools import combinations

import numpy as np
import json

from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction

from monty.json import MSONable


__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.1"
__email__ = "mcdermott@lbl.gov"
__date__ = "July 20, 2020"


class CustomEntry(PDEntry):
    def __init__(self, composition, energy_dict, temp=None, name=None, attribute=None):
        composition = Composition(composition)

        if not temp:
            temp = 300

        super().__init__(composition, energy_dict[str(temp)], name=name,
                         attribute=attribute)
        self.temp = temp
        self.energy_dict = energy_dict

    def set_temp(self, temp):
        super().__init__(self.composition, self.energy_dict[str(temp)], name=self.name,
                         attribute=self.attribute)

    def __repr__(self):
        return super().__repr__() + f" (T={self.temp} K)"


class RxnEntries(MSONable):
    """
    Helper class for describing combinations of ComputedEntry-like objects in context
    of a reaction network. Necessary for implementation in NetworkX (and useful
    for other network packages!)
    """

    def __init__(self, entries, description):
        """
        Args:
            entries [ComputedEntry]: list of ComputedEntry-like objects
            description (str): Node type, as selected from:
                "R" (reactants), "P" (products),
                "S" (starters/precursors), "T" (target),
                "D" (dummy)
        """
        self._entries = set(entries) if entries else None
        self._chemsys = (
            "-".join(
                sorted(
                    {
                        str(el)
                        for entry in self._entries
                        for el in entry.composition.elements
                    }
                )
            )
            if entries
            else None
        )

        if description in ["r", "R", "reactants", "Reactants"]:
            self._description = "R"
        elif description in ["p", "P", "products", "Products"]:
            self._description = "P"
        elif description in [
            "s",
            "S",
            "precursors",
            "Precursors",
            "starters",
            "Starters",
        ]:
            self._description = "S"
        elif description in ["t", "T", "target", "Target"]:
            self._description = "T"
        elif description in ["d", "D", "dummy", "Dummy"]:
            self._description = "D"
        else:
            self._description = description

    @property
    def entries(self):
        return self._entries

    @property
    def description(self):
        return self._description

    @property
    def chemsys(self):
        return self._chemsys

    def __repr__(self):
        if self._description == "D":
            return "Dummy Node"

        formulas = [entry.composition.reduced_formula for entry in self._entries]
        formulas.sort()
        if not self._description:
            return f"{','.join(formulas)}"
        else:
            return f"{self._description}: {','.join(formulas)}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.description == other.description:
                if self.chemsys == other.chemsys:
                    return self.entries == other.entries
        else:
            return False

    def __hash__(self):
        if not self._description or self._description == "D":
            return hash(self._description)
        else:
            return hash((self._description, frozenset(self._entries)))


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
        self.all_rxns = list(self.rxn_dict.keys())
        self.net_rxn = net_rxn
        self.all_reactants = set()
        self.all_products = set()
        self.is_balanced = False
        self.multiplicities = None
        self.total_cost = None
        self.average_cost = None

        for rxn in self.rxn_dict.keys():
            self.all_reactants.update(rxn.reactants)
            self.all_products.update(rxn.products)

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

        if self.is_balanced:
            self.calculate_costs()

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

    def calculate_costs(self):
        """
        Calculates and sets total and average cost of all pathways using the reaction
        dict.
        """
        self.total_cost = sum(
            [mult * self.rxn_dict[rxn] for (rxn, mult) in self.multiplicities.items()]
        )
        self.average_cost = self.total_cost / len(self.rxn_dict)

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
        rxn_info += f"\nAverage Cost: {round(self.average_cost,3)} \n" \
                    f"Total Cost: {round(self.total_cost,3)}"

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

    for e in sorted(entries, key=lambda x: len(x.composition.elements), reverse=True):
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
