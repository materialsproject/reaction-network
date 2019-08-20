from itertools import combinations

import numpy as np
import json

from pymatgen import Composition
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.entries.computed_entries import ComputedStructureEntry

from monty.json import MSONable, MontyDecoder


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"

with open("g_els.json") as f:
    G_ELEMS = json.load(f)


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
        self._temp = temp

    def gf_sisso(self):
        """
        Returns:
            Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel et al. (2018) [eV]
                (not normalized)
        """
        if self.structure.composition.is_element:
            return 0

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

        denominator = ((num_elems - 1)*reduced_comp.num_atoms)

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j)*(m_i*m_j)/(m_i+m_j)

        reduced_mass = (1 / denominator) * mass_sum

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
                                                                 data=entry.data,entry_id=entry.entry_id))
        return gibbs_entries

    def __repr__(self):
        output = ["GibbsComputedStructureEntry {} - {}".format(
            self.entry_id, self.composition.formula),
            "Gibbs Free Energy (Formation) = {:.4f}".format(self.energy)]
        return "\n".join(output)


class RxnEntries(MSONable):
    """
    Helper class for describing ComputedEntry-like objects in context of a reaction network.
        Necessary for implementation in NetworkX.
    """

    def __init__(self, entries, description):
        self._entries = entries

        if description in ["r", "reactants", "Reactants"]:
            self._description = "Reactants"
        elif description in ["p", "products", "Products"]:
            self._description = "Products"
        elif description in ["s", "starters", "Starters"]:
            self._description = "Starters"
        elif description in ["t", "target", "Target"]:
            self._description = "Target"

    @property
    def entries(self):
        return self._entries

    @property
    def description(self):
        return self._description

    def __repr__(self):
        return f"{self._description}: {str([entry.composition.reduced_formula for entry in self._entries])}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash((self._description, frozenset(self._entries)))


class RxnPathway(MSONable):
    """
    Helper class for storing multiple ComputedReaction objects which form a single reaction pathway.
    """

    def __init__(self, rxns, weights):
        self._rxns = list(rxns)
        self._weights = list(weights)

        self.total_weight = sum(self._weights)
        self._dH_per_atom = [rxn.calculated_reaction_energy / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                             for rxn in self._rxns]

    @property
    def rxns(self):
        return self._rxns

    @property
    def weights(self):
        return self._weights

    @property
    def dH_per_atom(self):
        return self._dH_per_atom

    def __repr__(self):
        path_info = ""
        for rxn, dH in zip(self._rxns, self._dH_per_atom):
            path_info += f"{rxn} (dH = {round(dH,3)} eV/atom) \n"

        path_info += f"Total Cost: {round(self.total_weight,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._rxns))


class CombinedPathway(MSONable):
    """
    Helper class for combining multiple RxnPathway objects in series/parallel to form a "net" pathway
        from a set of initial reactants to final products.
    """

    def __init__(self, paths, starters, targets):
        self._paths = list(paths)
        self._starters = list(starters)
        self._targets = list(targets)
        self.average_weight = np.mean([path.total_weight for path in self._paths])
        self.total_weight = sum([path.total_weight for path in self._paths])
        self.net_rxn = ComputedReaction(self._starters, self._targets)

        self.all_rxns = list({step for path in self._paths for step in path.rxns})

        reactants = set()
        products = set()

        for path in self._paths:
            for step in path.rxns:
                reactants.update(step.reactants)
                products.update(step.products)

        self.all_reactants = reactants
        self.all_products = products
        self.all_comp = list(self.all_reactants | self.all_products)

        self.multiplicities = None
        self.is_balanced = False
        self._balance_pathways()

    def _balance_pathways(self):
        net_coeffs = [self.net_rxn.get_coeff(comp) if comp in self.net_rxn.all_comp else 0
                      for comp in self.all_comp]
        comp_matrix = np.array([[rxn.get_coeff(comp) if comp in rxn.all_comp else 0 for comp in self.all_comp]
                                for rxn in self.all_rxns])
        comp_pseudo_inverse = np.linalg.pinv(comp_matrix).transpose()
        self.multiplicities = np.matmul(comp_pseudo_inverse, net_coeffs)

        if (self.multiplicities < self.net_rxn.TOLERANCE).any():
            return
        elif np.allclose(np.matmul(comp_matrix.transpose(), self.multiplicities), net_coeffs):
            self.is_balanced = True

        return

    @property
    def paths(self):
        return self._paths

    def __repr__(self):
        path_info = ""
        for path in self._paths:
            path_info += f"{str(path)} \n\n"
        path_info += f"Average Cost: {round(self.average_weight,3)} \nTotal Cost: {round(self.total_weight,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._paths))