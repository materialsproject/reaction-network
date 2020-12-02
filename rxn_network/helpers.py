"""
This module implements several helper classes for storing and parsing reaction pathway
info in the Reaction Network core module.
"""

import os
from itertools import chain, combinations

import numpy as np
import json
import hashlib

from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram, \
    PDEntry
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError, ComputedReaction
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.interface_reactions import InterfacialReactivity
from scipy.interpolate import interp1d

from monty.json import MSONable, MontyEncoder, MontyDecoder


__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.1"
__email__ = "mcdermott@lbl.gov"
__date__ = "July 20, 2020"


with open(os.path.join(os.path.dirname(__file__), "g_els.json")) as f:
    G_ELEMS = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "nist_gas_gf.json")) as f:
    G_GASES = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "compounds.json")) as f:
    G_COMPOUNDS = json.load(f)


def _new_pdentry_hash(self):  # necessary fix, will be updated in pymatgen in future
    data_md5 = hashlib.md5(f"{self.composition.formula}_"
                           f"{self.energy}".encode('utf-8')).hexdigest()
    return int(data_md5, 16)


PDEntry.__hash__ = _new_pdentry_hash


class GibbsComputedStructureEntry(ComputedStructureEntry):
    """
    An extension to ComputedStructureEntry which includes the estimated Gibbs
    free energy of formation via a machine-learned model.
    """

    def __init__(
        self,
        structure: Structure,
        formation_enthalpy: float,
        temp: float = 300,
        gibbs_model: str = "SISSO",
        correction: float = 0.0,
        energy_adjustments: list = None,
        parameters: dict = None,
        data: dict = None,
        entry_id: object = None,
    ):
        """
        Args:
            structure (Structure): The pymatgen Structure object of an entry.
            formation_enthalpy (float): Formation enthalpy of the entry, calculated
                using phase diagram construction (eV)
            temp (float): Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will
                be interpolated. Defaults to 300 K.
            gibbs_model (str): Model for Gibbs Free energy. Currently the default (and
                only supported) option is "SISSO", the descriptor created by Bartel et
                al. (2018).
            correction (float): A correction to be applied to the energy. Defaults to 0
            parameters (dict): An optional dict of parameters associated with
                the entry. Defaults to None.
            data (dict): An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        self._structure = structure
        self.formation_enthalpy = formation_enthalpy
        self.temp = temp
        self.interpolated = False

        if self.temp < 300 or self.temp > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        if self.temp % 100:
            self.interpolated = True

        if gibbs_model.lower() == "sisso":
            gibbs_energy = self.gf_sisso()
        else:
            raise ValueError(
                f"{gibbs_model} not a valid model. Please select from [" f"'SISSO']"
            )

        self.gibbs_model = gibbs_model

        super().__init__(
            structure,
            energy=gibbs_energy,
            correction=correction,
            energy_adjustments=energy_adjustments,
            parameters=parameters,
            data=data,
            entry_id=entry_id,
        )

    def gf_sisso(self) -> float:
        """
        Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel
        et al. (2018). Units: eV (not normalized)

        WARNING: This descriptor only applies to solids. The implementation here
        attempts to detect and use downloaded NIST-JANAF data for common gases (e.g.
        CO2) where possible.

        Reference: Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R.,
        Tumas, W., Weimer, A. W., … Holder, A. M. (2018). Physical descriptor for
        the Gibbs energy of inorganic crystalline solids and
        temperature-dependent materials chemistry. Nature Communications, 9(1),
        4168. https://doi.org/10.1038/s41467-018-06682-4

        Returns:
            float: Gibbs free energy of formation (eV)
        """
        comp = self.structure.composition

        if comp.is_element:
            return self.formation_enthalpy

        exp_data = False
        if comp.reduced_formula in G_GASES.keys():
            exp_data = True
            data = G_GASES[comp.reduced_formula]
            factor = comp.get_reduced_formula_and_factor()[1]
        elif comp.reduced_formula in G_COMPOUNDS.keys():
            exp_data = True
            data = G_COMPOUNDS[comp.reduced_formula]
            factor = comp.get_reduced_formula_and_factor()[1]

        if exp_data:
            if self.interpolated:
                g_interp = interp1d([int(t) for t in data.keys()], list(data.values()))
                return g_interp(self.temp) * factor
            else:
                return data[str(self.temp)] * factor

        num_atoms = self.structure.num_sites
        vol_per_atom = self.structure.volume / num_atoms
        reduced_mass = self._reduced_mass()

        return (
            self.formation_enthalpy
            + num_atoms * self._g_delta_sisso(vol_per_atom, reduced_mass, self.temp)
            - self._sum_g_i()
        )

    def _sum_g_i(self) -> float:
        """
        Sum of the stoichiometrically weighted chemical potentials of the elements
        at specified temperature, as acquired from "g_els.json".

        Returns:
             float: sum of weighted chemical potentials [eV]
        """
        elems = self.structure.composition.get_el_amt_dict()

        if self.interpolated:
            sum_g_i = 0
            for elem, amt in elems.items():
                g_interp = interp1d(
                    [float(t) for t in G_ELEMS.keys()],
                    [g_dict[elem] for g_dict in G_ELEMS.values()],
                )
                sum_g_i += amt * g_interp(self.temp)
        else:
            sum_g_i = sum(
                [amt * G_ELEMS[str(self.temp)][elem] for elem, amt in elems.items()]
            )

        return sum_g_i

    def _reduced_mass(self) -> float:
        """
        Reduced mass as calculated via Eq. 6 in Bartel et al. (2018)

        Returns:
            float: reduced mass (amu)
        """
        reduced_comp = self.structure.composition.reduced_composition
        num_elems = len(reduced_comp.elements)
        elem_dict = reduced_comp.get_el_amt_dict()

        denominator = (num_elems - 1) * reduced_comp.num_atoms

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j)

        reduced_mass = (1 / denominator) * mass_sum

        return reduced_mass

    @staticmethod
    def _g_delta_sisso(vol_per_atom, reduced_mass, temp) -> float:
        """
        G^delta as predicted by SISSO-learned descriptor from Eq. (4) in
        Bartel et al. (2018).

        Args:
            vol_per_atom (float): volume per atom [Å^3/atom]
            reduced_mass (float) - reduced mass as calculated with pair-wise sum formula
                [amu]
            temp (float) - Temperature [K]

        Returns:
            float: G^delta
        """

        return (
            (-2.48e-4 * np.log(vol_per_atom) - 8.94e-5 * reduced_mass / vol_per_atom)
            * temp
            + 0.181 * np.log(temp)
            - 0.882
        )

    @classmethod
    def from_pd(
        cls, pd, temp=300, gibbs_model="SISSO"
    ):
        """
        Constructor method for initializing a list of GibbsComputedStructureEntry
        objects from an existing T = 0 K phase diagram composed of
        ComputedStructureEntry objects, as acquired from a thermochemical database;
        e.g. The Materials Project.

        Args:
            pd (PhaseDiagram): T = 0 K phase diagram as created in pymatgen. Must
                contain ComputedStructureEntry objects.
            temp (int): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """
        gibbs_entries = []
        for entry in pd.all_entries:
            if (
                entry in pd.el_refs.values()
                or not entry.structure.composition.is_element
            ):
                gibbs_entries.append(
                    cls(
                        entry.structure,
                        formation_enthalpy=pd.get_form_energy(entry),
                        temp=temp,
                        correction=0,
                        gibbs_model=gibbs_model,
                        data=entry.data,
                        entry_id=entry.entry_id,
                    )
                )
        return gibbs_entries

    @classmethod
    def from_entries(
        cls, entries, temp=300, gibbs_model="SISSO"
    ):
        """
        Constructor method for initializing GibbsComputedStructureEntry objects from
        T = 0 K ComputedStructureEntry objects, as acquired from a thermochemical
        database e.g. The Materials Project.

        Args:
            entries ([ComputedStructureEntry]): List of ComputedStructureEntry objects,
                as downloaded from The Materials Project API.
            temp (int): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        pd = PhaseDiagram(entries)
        return cls.from_pd(pd, temp, gibbs_model)

    def to_pd_entry(self):
        data = {"entry_id": self.entry_id}
        data.update(self.data)
        return PDEntry(self.composition, self.energy, self.name, data)

    def as_dict(self) -> dict:
        """
        :return: MSONAble dict.
        """
        d = super().as_dict()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["formation_enthalpy"] = self.formation_enthalpy
        d["temp"] = self.temp
        d["gibbs_model"] = self.gibbs_model
        d["interpolated"] = self.interpolated
        return d

    @classmethod
    def from_dict(cls, d) -> "GibbsComputedStructureEntry":
        """
        :param d: Dict representation.
        :return: GibbsComputedStructureEntry
        """
        dec = MontyDecoder()
        return cls(
            dec.process_decoded(d["structure"]),
            d["formation_enthalpy"],
            d["temp"],
            d["gibbs_model"],
            correction=d["correction"],
            energy_adjustments=[
                dec.process_decoded(e) for e in d.get("energy_adjustments", {})
            ],
            parameters={
                k: dec.process_decoded(v) for k, v in d.get("parameters", {}).items()
            },
            data={k: dec.process_decoded(v) for k, v in d.get("data", {}).items()},
            entry_id=d.get("entry_id", None),
        )

    def __repr__(self):
        output = [
            "GibbsComputedStructureEntry {} - {}".format(
                self.entry_id, self.composition.formula
            ),
            "Gibbs Free Energy (Formation) = {:.4f}".format(self.energy),
        ]
        return "\n".join(output)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.composition == other.composition) and \
                   (self.formation_enthalpy == other.formation_enthalpy) and \
                   (self.entry_id == other.entry_id) and (self.temp == other.temp)
        else:
            return False

    def __hash__(self):
        data_md5 = hashlib.md5(f"{self.composition}_"
                               f"{self.formation_enthalpy}_{self.entry_id}_"
                               f"{self.temp}".encode('utf-8')).hexdigest()
        return int(data_md5, 16)


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
        self.is_balanced = False
        self.multiplicities = None
        self.total_cost = None
        self.average_cost = None

        self.all_reactants = {reactants for rxn in self.rxn_dict.keys() for reactants in
                              rxn.reactants}
        self.all_products = {products for rxn in self.rxn_dict.keys() for products in
                              rxn.products}

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
        rxn_info += f"\nTotal Cost: {round(self.total_cost,3)} | Average Cost: " \
                    f"{round(self.average_cost,3)}\n\n"

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


def generate_all_combos(entries, max_num_combos):
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


def react_interface(r1, r2, pd, grand_pd=None):
    if grand_pd:
        interface = InterfacialReactivity(
            r1,
            r2,
            grand_pd,
            norm=True,
            include_no_mixing_energy=False,
            pd_non_grand=pd,
            use_hull_energy=True
        )
    else:
        interface = InterfacialReactivity(r1,
                                          r2,
                                          pd,
                                          norm=False,
                                          include_no_mixing_energy=False,
                                          pd_non_grand=None,
                                          use_hull_energy=True)

    entries = pd.all_entries
    rxns = {get_computed_rxn(rxn, entries) for _, _, _, rxn,
                                                _ in interface.get_kinks()}

    return rxns


def get_computed_rxn(rxn, entries):
    reactants = [r.reduced_composition for r in rxn.reactants if not np.isclose(
        rxn.get_coeff(r), 0)]
    products = [p.reduced_composition for p in rxn.products if not np.isclose(
        rxn.get_coeff(p), 0)]
    reactant_entries = [get_entry_by_comp(r, entries) for r in reactants]
    product_entries = [get_entry_by_comp(p, entries) for p in products]
    return ComputedReaction(reactant_entries, product_entries)


def get_entry_by_comp(comp, entry_set):
    possible_entries = filter(lambda x: x.composition.reduced_composition
                                                   == comp, entry_set)
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


def find_interdependent_rxns(path, precursors, verbose=True):
    precursors = set(precursors)
    interdependent = False
    combined_rxn = None

    rxns = set(path.all_rxns)
    num_rxns = len(rxns)

    if num_rxns == 1:
        return False, None

    for combo in generate_all_combos(rxns, num_rxns):
        size = len(combo)
        if any([set(rxn.reactants).issubset(precursors) for rxn in combo]) or size==1:
            continue
        other_comp = {c for rxn in (rxns - set(combo)) for c in rxn.all_comp}

        unique_reactants = []
        unique_products = []
        for rxn in combo:
            unique_reactants.append(set(rxn.reactants) - precursors)
            unique_products.append(set(rxn.products) - precursors)

        overlap = [False]*size
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
                combined_rxn = Reaction(list(combined_reactants), list(combined_products))
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


def get_rxn_cost(rxn, cost_function="softplus", temp=273, max_mu_diff=None,
                      most_negative_rxn=None):
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

