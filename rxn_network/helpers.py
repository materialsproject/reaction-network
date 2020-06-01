"""
This module implements several helper classes for storing and parsing reaction pathway info
in the Reaction Network module.
"""

import os
from itertools import combinations

import numpy as np
import json

from pymatgen import Composition
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.entries.computed_entries import ComputedStructureEntry

from monty.json import MSONable, MontyDecoder


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"
__date__ = "February 25, 2020"


with open(os.path.join(os.path.dirname(__file__), "g_els.json")) as f:
    G_ELEMS = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "nist_gas_gf.json")) as f:
    G_GASES = json.load(f)


class GibbsComputedStructureEntry(ComputedStructureEntry):
    """
    An extension to ComputedStructureEntry which includes the estimated Gibbs free energy of formation.
    """

    def __init__(self, structure, formation_enthalpy, temp=300, gibbs_model="SISSO", correction=None, parameters=None,
                 data=None, entry_id=None):
        """
        Initializes a GibbsComputedStructureEntry.

        Args:
            structure (Structure): The actual structure of an entry.
            formation_enthalpy (float): Formation enthalpy of the entry, calculated using phase diagram construction (eV)
            temp (int): Temperature in Kelvin.
            gibbs_model (str): Model for Gibbs Free energy. Currently supported options: ["SISSO"]
            correction (float): A correction to be applied to the energy. Defaults to 0
            parameters (dict): An optional dict of parameters associated with
                the entry. Defaults to None.
            data (dict): An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id (obj): An optional id to uniquely identify the entry.
        """
        self.structure = structure
        self.formation_enthalpy = formation_enthalpy
        self.temp = temp

        super().__init__(structure, energy=self.gf_sisso(), correction=correction,
                         parameters=parameters, data=data, entry_id=entry_id)

        self._gibbs_model = gibbs_model

    def gf_sisso(self):
        """
        Returns:
            Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel et al. (2018) [eV]
                (not normalized)
        """
        comp = self.structure.composition
        if comp.is_element:
            return self.formation_enthalpy
        elif comp.reduced_formula in G_GASES.keys():
            return G_GASES[comp.reduced_formula][str(self.temp)]*comp.get_reduced_formula_and_factor()[1]

        num_atoms = self.structure.num_sites
        vol_per_atom = self.structure.volume / num_atoms
        reduced_mass = self.reduced_mass()

        return self.formation_enthalpy + num_atoms*self.g_delta(vol_per_atom, reduced_mass, self.temp) - self._sum_g_i()

    def _sum_g_i(self):
        """
        Returns:
            Sum of the stoichiometrically weighted chemical potentials of the elements at T found in "g_els.json"
             (float) [eV]
        """
        elems = self.structure.composition.get_el_amt_dict()
        return sum([amt*G_ELEMS[str(self.temp)][elem] for elem, amt in elems.items()])

    def reduced_mass(self):
        """
        Returns:
            Reduced mass calculated via Eq. 6 in Bartel et al. (2018)
        """
        reduced_comp = self.structure.composition.reduced_composition
        num_elems = len(reduced_comp.elements)
        elem_dict = reduced_comp.get_el_amt_dict()

        denominator = (num_elems - 1)*reduced_comp.num_atoms

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j)*(m_i*m_j)/(m_i + m_j)

        reduced_mass = (1/denominator)*mass_sum

        return reduced_mass

    @staticmethod
    def g_delta(vol_per_atom, reduced_mass, temp):
        """
        Args:
            vol_per_atom: volume per atom [Å^3/atom]
            reduced_mass (float) - reduced mass as calculated with pair-wise sum formula [amu]
            temp (float) - Temperature [K]
        Returns:
            G^delta as predicted by SISSO-learned descriptor from Bartel et al. (2018) (float) [eV/atom]
        """

        return (-2.48e-4*np.log(vol_per_atom) - 8.94e-5*reduced_mass/vol_per_atom)*temp + 0.181*np.log(temp) - 0.882

    @staticmethod
    def from_pd(pd, temp=300, gibbs_model="SISSO"):
        gibbs_entries = []
        for entry in pd.all_entries:
            if entry in pd.el_refs.values() or not entry.structure.composition.is_element:
                gibbs_entries.append(GibbsComputedStructureEntry(entry.structure,
                                                                 formation_enthalpy=pd.get_form_energy(entry),
                                                                 temp=temp, correction=0, gibbs_model=gibbs_model,
                                                                 data=entry.data, entry_id=entry.entry_id))
        return gibbs_entries

    def __repr__(self):
        output = ["GibbsComputedStructureEntry {} - {}".format(
            self.entry_id, self.composition.formula),
            "Gibbs Free Energy (Formation) = {:.4f}".format(self.energy)]
        return "\n".join(output)


class RxnEntries(MSONable):
    """
    Helper class for describing combinations of ComputedEntry-like objects in context of a reaction network.
        Necessary for implementation in NetworkX (and useful for other network packages!)
    """

    def __init__(self, entries, description):
        """
        Args:
            entries: [ComputedEntry] list of ComputedEntry-like objects
            description: Node type, selected from "R" (reactants), "P" (products), "S" (starters/precursors),
                "T" (target), or "D" (dummy)
        """
        self._entries = set(entries) if entries else None
        self._chemsys = "-".join(sorted({str(el) for entry in self._entries
                                         for el in entry.composition.elements})) if entries else None

        if description in ["r", "R", "reactants", "Reactants"]:
            self._description = "R"
        elif description in ["p", "P", "products", "Products"]:
            self._description = "P"
        elif description in ["s", "S", "precursors", "Precursors", "starters", "Starters"]:
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
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        if not self._description or self._description == "D":
            return hash(self._description)
        else:
            return hash((self._description, frozenset(self._entries)))


class RxnPathway(MSONable):
    """
    Helper class for storing multiple ComputedReaction objects which form a single reaction pathway.
    """

    def __init__(self, rxns, costs):
        self._rxns = list(rxns)
        self._costs = list(costs)

        self.total_cost = sum(self._costs)
        self._dG_per_atom = [rxn.calculated_reaction_energy / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                             for rxn in self._rxns]

    @property
    def rxns(self):
        return self._rxns

    @property
    def costs(self):
        return self._costs

    @property
    def dG_per_atom(self):
        return self._dG_per_atom

    def __repr__(self):
        path_info = ""
        for rxn, dG in zip(self._rxns, self._dG_per_atom):
            path_info += f"{rxn} (dG = {round(dG,3)} eV/atom) \n"

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
    Helper class for combining multiple reactions which stoichiometrically balance to form a net reaction.
    """
    def __init__(self, rxn_dict, net_rxn, multiplicities=None, is_balanced=False):
        self.rxn_dict = rxn_dict
        self.all_rxns = list(self.rxn_dict.keys())
        self.net_rxn = net_rxn
        self.all_reactants = set()
        self.all_products = set()
        self.is_balanced = is_balanced

        for rxn in self.rxn_dict.keys():
            self.all_reactants.update(rxn.reactants)
            self.all_products.update(rxn.products)

        self.all_comp = list(self.all_reactants | self.all_products | set(self.net_rxn.all_comp))

        if not multiplicities:
            net_coeffs = self._get_net_coeffs(net_rxn, self.all_comp)
            comp_matrix = self._get_comp_matrix(self.all_comp, self.all_rxns)
            self.is_balanced, multiplicities = self._balance_rxns(comp_matrix, net_coeffs)
            self.multiplicities = {rxn: multiplicity for (rxn, multiplicity) in zip(self.all_rxns, multiplicities)}

        if self.is_balanced:
            self.total_cost = sum([mult*self.rxn_dict[rxn] for (rxn, mult) in self.multiplicities.items()])
            self.average_cost = self.total_cost / len(self.rxn_dict)

    @staticmethod
    def _balance_rxns(comp_matrix, net_coeffs, tol=1e-6):
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
        return np.array([net_rxn.get_coeff(comp) if comp in net_rxn.all_comp else 0
                         for comp in all_comp])

    @staticmethod
    def _get_comp_matrix(all_comp, all_rxns):
        return np.array([[rxn.get_coeff(comp) if comp in rxn.all_comp else 0 for comp in all_comp]
                         for rxn in all_rxns])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return set(self.all_rxns) == set(other.all_rxns)
        else:
            return False

    def __repr__(self):
        rxn_info = ""
        for rxn, cost in self.rxn_dict.items():
            dg_per_atom = rxn.calculated_reaction_energy / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
            rxn_info += f"{rxn} (dG = {round(dg_per_atom,3)} eV/atom) \n"
        rxn_info += f"\nAverage Cost: {round(self.average_cost,3)} \nTotal Cost: {round(self.total_cost,3)}"

        return rxn_info

    def __hash__(self):
        return hash(frozenset(self.all_rxns))


class CombinedPathway(BalancedPathway):
    """
    Helper class for combining multiple RxnPathway objects in series/parallel to form a "net" pathway
        from a set of initial reactants to final products.
    """

    def __init__(self, paths, net_rxn):
        self._paths = paths
        rxn_dict = {rxn: cost for path in self._paths for (rxn, cost) in zip(path.rxns, path.costs)}

        super().__init__(rxn_dict, net_rxn)

    @property
    def paths(self):
        return self._paths

    def __repr__(self):
        path_info = ""
        for path in self._paths:
            path_info += f"{str(path)} \n\n"
        path_info += f"Average Cost: {round(self.average_cost,3)} \nTotal Cost: {round(self.total_cost,3)}"

        return path_info
