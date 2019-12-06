import logging
from itertools import combinations, chain
from collections import Counter

import numpy as np
from scipy.constants import physical_constants

import networkx as nx

from pymatgen import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError
from matminer.featurizers.structure import StructuralComplexity

from rxn_network.helpers import RxnEntries, RxnPathway, CombinedPathway
from rxn_network.analysis import PathwayAnalysis


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph (implemented in NetworkX) which enumerates and
        explores all possible chemical reactions (edges) between reactant/product combinations (nodes) in a
        chemical system.
    """
    def __init__(self, entries, max_num_components=2, include_metastable=False,
                 include_polymorphs=False, include_info_entropy=False):
        """
        Constructs a ReactionNetwork object with necessary initialization steps of finding combinations
        (but does not generate the actual network).

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-like objects to consider in network
            max_num_components (int): maximum number of components allowed on each side of the reaction (default 2)
            include_metastable (float or bool): either the specified cutoff for energy above hull, or False if
                considering only stable entries
            include_polymorphs (bool): Whether or not to consider non-ground state polymorphs (defaults to False)
        """
        self._all_entries = entries
        self._pd = PhaseDiagram(entries)
        self._max_num_components = max_num_components
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs

        self._starters = None
        self._all_targets = None
        self._selected_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._temp = None
        self._redox = None
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        self._rxn_network = None
        #self._vis = None

        self.logger = logging.getLogger('ReactionNetwork')
        self.logger.setLevel("INFO")

        self._filtered_entries = self.filter_entries(self._pd, include_metastable, include_polymorphs)
        if include_info_entropy:
            for entry in self._filtered_entries:
                info_entropy = StructuralComplexity().featurize(entry.structure)[0]
                entry.parameters = {"info_entropy": info_entropy}

        filtered_entries_str = ', '.join([entry.composition.reduced_formula for entry in self._filtered_entries])

        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} entries: \n{filtered_entries_str}")
        #self.logger.info(f"Found {len(self._all_combos)} combinations of entries (size <= {self._max_num_components}).")

    def generate_rxn_network(self, starters, targets, cost_function="softplus", temp=300, complex_loopback=True, redox=False):
        """
        Generates the actual reaction network (weighted, directed graph) using NetworkX.

        Args:
            starters (list of ComputedEntries): entries for all phases which serve as the main reactants
            targets (list of ComputedEntries): entries for all phases which are the final products
            cost_function (str): name of cost function to use for entire network (e.g. "softplus")
            complex_loopback (bool): whether or not to add edges looping back to
                combinations of intermediates and initial reactants

        Returns:
            None
        """
        self._starters = set(starters) if starters else None
        self._all_targets = set(targets) if targets else None
        self._selected_target = {targets[0]}  # take first entry to be first designated target
        self._cost_function = cost_function
        self._complex_loopback = complex_loopback
        self._starter_entries = None
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions
        self._temp = temp
        self._redox = redox

        if not self._starters:
            starter_entries = RxnEntries(None, "d")  # use dummy starter node
            if self._complex_loopback:
                raise ValueError("Complex loopback cannot be enabled when using a dummy starters node!")
        elif False in [isinstance(starter, ComputedEntry) for starter in self._starters] or False in [
                isinstance(target, ComputedEntry) for target in self._all_targets]:
            raise TypeError("Starters and target must be (or inherit from) ComputedEntry objects.")
        else:
            starter_entries = RxnEntries(starters, "s")
            self._starter_entries = starter_entries

        g = nx.DiGraph()
        #vis = nx.DiGraph()  # simpler graph for visualization

        target_entry = RxnEntries(self._selected_target, "t")

        g.add_nodes_from([starter_entries, target_entry])
        self.logger.info("Generating reactions...")

        for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
            reactants = set(r)
            reactant_entries = RxnEntries(reactants, "r")
            g.add_node(reactant_entries)

            #vis.add_node(RxnEntries(reactants, None), path=0)

            # connect starters to reactants
            if starter_entries.description == "D" or (self._starters.issubset(reactants) and redox) \
                    or (reactants.issubset(self._starters) and not redox):
                g.add_edge(starter_entries, reactant_entries, weight=0)

        for p in self.generate_all_combos(self._filtered_entries, self._max_num_components):
            products = set(p)
            product_entries = RxnEntries(products, "p")
            g.add_node(product_entries)

            # link back to any node which contains starters or these intermediate products, p
            if complex_loopback:
                for c in self.generate_all_combos(products.union(self._starters), self._max_num_components):
                    combo = set(c)
                    if not combo.issubset(self._starters):  # must contain at least one intermediate
                        g.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)
                    #if combo != products:
                        #vis.add_edge(RxnEntries(products, None), RxnEntries(combo, None), weight=0, path=0)
            else:
                g.add_edge(product_entries, RxnEntries(products, "r"), weight=0)

            if self._selected_target.issubset(products):
                g.add_edge(product_entries, target_entry, weight=0)  # add edges connecting to target

            # find all the reactions now
            for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                reactants = set(r)
                if products == reactants:
                    continue  # do not consider identity-like reactions (A + B -> A + B)

                reactants_elems = set([elem for entry in reactants for elem in entry.composition.elements])
                products_elems = set([elem for entry in products for elem in entry.composition.elements])

                if reactants_elems != products_elems:
                    continue  # do not consider reaction which changes chemical systems

                reactant_entries = RxnEntries(reactants, "r")

                try:
                    rxn_forwards = ComputedReaction(list(reactants), list(products))
                    #rxn_backwards = ComputedReaction(list(products), list(reactants))
                except ReactionError:
                    continue

                reactants_comps = {reactant.composition.reduced_composition for reactant in reactants}
                products_comps = {product.composition.reduced_composition for product in products}

                if (True in (abs(rxn_forwards.coeffs) < rxn_forwards.TOLERANCE)) or (reactants_comps != set(rxn_forwards.reactants)) or (
                        products_comps != set(rxn_forwards.products)):
                    continue  # remove reaction which has components that either change sides or disappear

                total_num_atoms = sum([rxn_forwards.get_el_amount(elem) for elem in rxn_forwards.elements])
                dg_per_atom_f = rxn_forwards.calculated_reaction_energy / total_num_atoms
                #dg_per_atom_b = rxn_backwards.calculated_reaction_energy / total_num_atoms

                # densities = np.array([entry.structure.density for entry in rxn_forwards.all_entries])
                # info_entropies = np.array([entry.parameters["info_entropy"] for entry in rxn_forwards.all_entries])
                # total_coeffs = sum([abs(coeff) for coeff in rxn_forwards.coeffs])

                # d_density = np.dot(rxn_forwards.coeffs, densities) / total_coeffs
                # d_info_entropy = np.dot(rxn_forwards.coeffs, info_entropies) / total_coeffs

                weight_f = self.get_rxn_cost(dg_per_atom_f, cost_function, temp) # d_info_entropy)  # d_density)
                #weight_b = self.get_rxn_cost(dg_per_atom_b, cost_function, temp)
                g.add_edge(reactant_entries, product_entries, weight=weight_f, rxn=rxn_forwards)

                #if dg_per_atom_f < 1:
                    #vis.add_edge(RxnEntries(reactants, None), RxnEntries(products, None), weight=weight_f, rxn=rxn_forwards, path=0)
                #if dg_per_atom_b < 1:
                    #vis.add_edge(RxnEntries(products, None), RxnEntries(reactants, None), weight=weight_b, rxn=rxn_backwards, path=0)

        self.logger.info(f"Complete: Created graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

        if cost_function in ["enthalpies_positive", "bipartite"]:
            self.logger.info(f"Adjusting enthalpies up by {-round(self._most_negative_rxn, 3)} eV")
            for (u, v, rxn) in g.edges.data(data='rxn'):
                if rxn is None:
                    continue
                elif cost_function == "enthalpies_positive":
                    g[u][v]["weight"] -= self._most_negative_rxn
                elif cost_function == "bipartite" and (g[u][v]["weight"] < 0):
                    g[u][v]["weight"] = 1 - (g[u][v]["weight"] / self._most_negative_rxn)

        self._rxn_network = g
        #self._vis = vis

    def find_k_shortest_paths(self, k, target=None):
        """
        Finds k shortest paths to designated target using Yen's Algorithm (as defined in Networkx)

        Args:
            k (int): desired number of shortest pathways (ranked by cost)

        Returns:
            [RxnPathway]: list of RxnPathway objects
        """

        if self._starters:
            starters = RxnEntries(self._starters, "s")
        else:
            starters = RxnEntries(None, "d")

        if target is None:
            target = self._selected_target

        target = RxnEntries(target, "t")

        paths = []
        num_found = 0

        try:
            for num, path in enumerate(nx.shortest_simple_paths(self._rxn_network, starters, target, weight="weight")):
                if num_found == k:
                    break

                path_reactants = None
                path_products = None
                rxns = []
                weights = []

                for step in path:
                    if step.description == "R":
                        path_reactants = step
                        #if path_products:
                            #vis_reactants = RxnEntries(path_reactants.entries, None)
                            #vis_products = RxnEntries(path_products.entries, None)
                            #nx.set_edge_attributes(self._vis, {(vis_products, vis_reactants): {"path": {"path": 10}}})
                    elif step.description == "P":
                        path_products = step
                        rxn_data = self._rxn_network.get_edge_data(path_reactants, path_products)

                        #vis_reactants = RxnEntries(path_reactants.entries, None)
                        #vis_products = RxnEntries(path_products.entries, None)
                        #nx.set_edge_attributes(self._vis, {(vis_reactants, vis_products): {"path": 10}})
                        #nx.set_node_attributes(self._vis, {vis_reactants: {"path": 10}})
                        #nx.set_node_attributes(self._vis, {vis_products: {"path": 10}})

                        rxn = rxn_data["rxn"]
                        weight = rxn_data["weight"]

                        rxns.append(rxn)
                        weights.append(weight)

                rxn_pathway = RxnPathway(rxns, weights)
                paths.append(rxn_pathway)
                print(rxn_pathway)
                print("\n")

                num_found += 1
        except nx.NetworkXNoPath:
            print("No paths found!")

        return paths

    def find_redox_paths(self, entry, reduced_entry, k=5, max_num_combos=4, consider_remnant_rxns=True):
        """

        Args:
        Returns:
        """
        hydrogen_entry = self._pd.el_refs[Element("H")]
        oxygen_entry = self._pd.el_refs[Element("O")]
        water_entry = None

        for e in self._filtered_entries:
            if e.composition.reduced_composition == Composition("H2O"):
                water_entry = e

        all_paths = set()

        self.set_starters([entry])
        self.set_target(reduced_entry)
        print(f"PATHS to {reduced_entry.composition.reduced_formula} \n")
        all_paths.update(self.find_k_shortest_paths(k))

        self.set_starters([reduced_entry])
        self.set_target(entry)
        print(f"PATHS to {entry.composition.reduced_formula} \n")
        all_paths.update(self.find_k_shortest_paths(k))

        self.set_starters(None)
        self.set_target(hydrogen_entry)
        print(f"PATHS to {hydrogen_entry.composition.reduced_formula} \n")
        all_paths.update(self.find_k_shortest_paths(k))

        self.set_starters(None)
        self.set_target(oxygen_entry)
        print(f"PATHS to {oxygen_entry.composition.reduced_formula} \n")
        all_paths.update(self.find_k_shortest_paths(k))

        self.set_starters([water_entry])
        self.set_target(entry, dummy=True)
        print(f"PATHS from {water_entry.composition.reduced_formula} \n")
        all_paths.update(self.find_k_shortest_paths(k))

        if consider_remnant_rxns:
            intermediates = {entry for path in all_paths for rxn in path.rxns for entry in rxn.all_entries} \
                            - {entry, reduced_entry}
            remnant_paths = self.find_remnant_paths(intermediates, {entry, reduced_entry})
            if remnant_paths:
                print("Remnant Reactions \n")
                print(remnant_paths, "\n")
                all_paths.update(remnant_paths)

        balanced_total_paths = set()
        for combo in self.generate_all_combos(all_paths, max_num_combos):
            combined_pathway = CombinedPathway(combo, [water_entry], [hydrogen_entry, oxygen_entry])
            if combined_pathway.is_balanced:
                balanced_total_paths.add(combined_pathway)

        return sorted(list(balanced_total_paths), key=lambda combined_path: combined_path.total_weight)

    def find_combined_paths(self, k, targets=None, max_num_combos=4, consider_remnant_rxns=True):
        """
        Builds k shortest paths to provided targets and then seeks to combine them to achieve a "net reaction"
            with balanced stoichiometry. In other words, the full conversion of all intermediates to final products.

        Args:
            k (int): calculated free energy of reaction
            targets ([ComputedEntries]): list of all target ComputedEntry objects
            max_num_combos (int): upper limit on how many pathways to consider at a time (default 3).

        Returns:
            [CombinedPathway]: list of CombinedPathway objects, sorted by average cost

        """
        paths_to_all_targets = set()
        if not targets:
            targets = self._all_targets
        else:
            targets = set(targets)

        net_rxn = ComputedReaction(list(self._starters), list(targets))
        print(f"NET RXN: {net_rxn} \n")

        for target in targets:
            print(f"PATHS to {target.composition.reduced_formula} \n")
            self.set_target(target)
            paths_to_all_targets.update(self.find_k_shortest_paths(k))

        if consider_remnant_rxns:
            starters_and_targets = targets | self._starters
            intermediates = {entry for path in paths_to_all_targets
                                 for rxn in path.rxns for entry in rxn.all_entries} - starters_and_targets
            remnant_paths = self.find_remnant_paths(intermediates, targets)
            if remnant_paths:
                print("Remnant Reactions \n")
                print(remnant_paths, "\n")
                paths_to_all_targets.update(remnant_paths)

        balanced_total_paths = set()
        for combo in self.generate_all_combos(paths_to_all_targets, max_num_combos):
            combined_pathway = CombinedPathway(combo, self._starters, targets)
            if combined_pathway.is_balanced:
                balanced_total_paths.add(combined_pathway)

        analysis = PathwayAnalysis(self, balanced_total_paths)

        return sorted(list(balanced_total_paths), key=lambda combined_path: combined_path.total_weight), analysis

    def find_remnant_paths(self, intermediates, targets, max_num_combos=2):
        """
        Helper method which looks for potential pathways between found intermediates.

        Args:
        Returns:

        """
        remnant_paths = set()

        for reactants_combo in self.generate_all_combos(intermediates, max_num_combos):
            for products_combo in self.generate_all_combos(targets, max_num_combos):
                try:
                    rxn = ComputedReaction(list(reactants_combo), list(products_combo))
                except ReactionError:
                    continue

                reactants_comps = {reactant.composition.reduced_composition for reactant in reactants_combo}
                products_comps = {product.composition.reduced_composition for product in products_combo}

                if (True in (abs(rxn.coeffs) < rxn.TOLERANCE)) or (reactants_comps != set(rxn.reactants)) or (
                        products_comps != set(rxn.products)):
                    continue

                path = RxnPathway([rxn], [self.get_rxn_cost(rxn.calculated_reaction_energy, self._cost_function)])
                remnant_paths.add(path)

        return remnant_paths

    def get_rxn_cost(self, energy, cost_function, temp=300, d_info_entropy=0, d_density=0):
        """
        Helper method which determines reaction cost/weight.

        Args:
            energy (float): calculated free energy of reaction
            cost_function (str): name of cost function (e.g. "softplus")
            temp (int): temperature parameter for scaling cost function [Kelvin]

        Returns:
            float: cost/weight of individual reaction edge
        """

        if cost_function == "softplus":
            weight = self._softplus(energy, temp, d_info_entropy, d_density)
        elif cost_function == "bipartite":
            weight = energy
            if weight < self._most_negative_rxn:
                self._most_negative_rxn = weight
            if weight >= 0:
                weight = 2 * weight + 1
        elif cost_function == "rectified":
            weight = energy
            if weight < 0:
                weight = 0
        elif cost_function == "arrhenius":
            weight = self._arrhenius(energy, temp)
        elif cost_function == "enthalpies_positive":
            weight = energy
            if weight < self._most_negative_rxn:
                self._most_negative_rxn = weight
        else:
            weight = 0

        return weight

    def set_cost_function(self, cost_function, temp=300):
        """
        Replaces network's current cost function with provided new function by recomputing edge weights.

        Args:
            cost_function (str): name of cost function (e.g. "softplus")

        Returns:
            None
        """
        for (u, v, rxn) in self._rxn_network.edges.data(data='rxn'):
            if rxn:
                total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                dg_per_atom = rxn.calculated_reaction_energy / total_num_atoms

                weight = self.get_rxn_cost(dg_per_atom, cost_function, temp)
                self._rxn_network[u][v]["weight"] = weight

    def set_starters(self, starters, connect_direct = False, dummy_exclusive = False):
        """
        Replaces network's previous starter nodes with provided new starters.
            Recreates edges that link products back to reactants.

        Args:
            starters ([ComputedEntry]): list of new starter entries

        Returns:
            None
        """

        for node in self._rxn_network.nodes():
            if node.description == "S" or node.description == "D":
                self._rxn_network.remove_node(node)

                if not starters or dummy_exclusive:
                    starter_entries = RxnEntries(None, "d")
                else:
                    starters = set(starters)
                    starter_entries = RxnEntries(starters, "s")

                self._rxn_network.add_node(starter_entries)

                if connect_direct:
                    self._rxn_network.add_edge(starter_entries, RxnEntries(starters, "r"), weight=0)
                elif self._redox:
                    for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                        reactants = set(r)
                        if (not starters or (dummy_exclusive) or starters.issubset(reactants)):
                            reactant_entries = RxnEntries(reactants, "r")
                            self._rxn_network.add_edge(starter_entries, reactant_entries, weight=0)
                else:
                    for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                        reactants = set(r)
                        if reactants.issubset(starters):  # link starting node to reactant nodes
                            reactant_entries = RxnEntries(reactants, "r")
                            self._rxn_network.add_edge(starter_entries, reactant_entries, weight=0)
                break

        if not self._complex_loopback:
            self._starters = starters
            return

        for products in self.generate_all_combos(self._filtered_entries, self._max_num_components):
            product_entries = RxnEntries(products, "p")

            old_loopbacks = self.generate_all_combos(list(products.union(self._starters)), self._max_num_components)
            for combo in old_loopbacks:
                # delete old edge linking back
                self._rxn_network.remove_edge(product_entries, RxnEntries(combo, "r"))

            new_loopbacks = self.generate_all_combos(list(products.union(starters)), self._max_num_components)
            for combo in new_loopbacks:
                # add new edges linking back
                self._rxn_network.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)

        self._starters = starters

    def set_target(self, target, dummy=False):
        """
        Replaces network's previous target node with provided new target.

        Args:
            target (ComputedEntry): entry of new target

        Returns:
            None
        """
        if target in self._selected_target:
            return

        self._selected_target = {target}

        for node in self._rxn_network.nodes():
            if node.description == "T":
                self._rxn_network.remove_node(node)

                target_entry = RxnEntries(self._selected_target, "t")

                for p in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                    products = set(p)
                    if self._selected_target.issubset(products) or dummy:
                        product_entries = RxnEntries(products, "p")
                        self._rxn_network.add_edge(product_entries, target_entry, weight=0) # link product node to target node

                break

    @staticmethod
    def filter_entries(pd, e_above_hull, include_polymorphs):
        """
        Helper method for filtering entries by specified energy above hull

        Args:
            e_above_hull (float): cutoff for energy above hull (** eV **)
            include_polymorphs (bool): whether to include higher energy polymorphs of existing structures

        Returns:
            list: all entries less than or equal to specified energy above hull
        """

        if e_above_hull == 0:
            filtered_entries = list(pd.stable_entries)
        else:
            filtered_entries = [entry for entry in pd.all_entries if
                                pd.get_e_above_hull(entry) <= e_above_hull]
            if not include_polymorphs:
                filtered_entries_no_polymorphs = []
                all_comp = {entry.composition.reduced_composition for entry in filtered_entries}
                for comp in all_comp:
                    polymorphs = [entry for entry in filtered_entries if entry.composition.reduced_composition == comp]
                    min_entry = min(polymorphs, key=lambda entry: entry.energy_per_atom)
                    filtered_entries_no_polymorphs.append(min_entry)

                return filtered_entries_no_polymorphs

        return filtered_entries

    @staticmethod
    def generate_all_combos(entries, max_num_combos):
        """
        Static helper method for generating combination sets ranging from singular length to maximum length
            specified by max_num_combos.

        Args:
            entries (list/set): list/set of all entry objects to combine
            max_num_combos (int): upper limit for size of combinations of entries

        Returns:
            list: all combination sets
        """

        return chain.from_iterable([combinations(entries, num_combos) for num_combos in range(1, max_num_combos + 1)])

    @staticmethod
    def _arrhenius(energy, t):
        ''' Simple Arrenhius relation involving energy and temperature '''
        kb = physical_constants["Boltzmann constant in eV/K"][0]
        return np.exp(energy / (kb * t))

    @staticmethod
    def _softplus(energy, t=1.0, d_info_entropy=0, d_density=0):
        weighted_params = energy - 0.25*d_info_entropy - 0.0*d_density
        return np.log(1 + (273 / t) * np.exp(weighted_params))

    @property
    def all_entries(self):
        return self._all_entries

    @property
    def filtered_entries(self):
        return self._filtered_entries

    @property
    def rxn_network(self):
        return self._rxn_network

    @property
    def starters(self):
        return self._starters

    @property
    def all_targets(self):
        return self._all_targets

    def __repr__(self):
        return str(self._rxn_network)
