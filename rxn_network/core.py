import logging
from itertools import combinations, chain

import numpy as np
from scipy.constants import physical_constants

import networkx as nx

from pymatgen import Element
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError
from matminer.featurizers.structure import StructuralComplexity

from rxn_network.helpers import RxnEntries, RxnPathway, CombinedPathway, GibbsComputedStructureEntry


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph (implemented in NetworkX) which enumerates and
        explores all possible chemical reactions (edges) between reactant/product combinations (nodes) in a
        chemical system.
    """
    def __init__(self, entries, max_num_components=2, include_metastable=False,
                 include_polymorphs=False, include_info_entropy=True):
        """
        Constructs a ReactionNetwork object with necessary initialization steps of finding combinations
        (but does not generate the actual network).

        Args:
            entries ([ComputedEntry]): list of ComputedEntry objects to consider in network
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
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        self._rxn_network = None
        self._vis = None

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

    def generate_rxn_network(self, starters, targets, cost_function="softplus", temp=300, complex_loopback=True):
        """
        Generates the actual reaction network (weighted, directed graph) using Networkx.

        Args:
            starters (list of ComputedEntries): entries for all phases which serve as the main reactants
            targets (list of ComputedEntries): entries for all phases which are the final products
            cost_function (str): name of cost function to use for entire network (e.g. "softplus")
            complex_loopback (bool): whether or not to add edges looping back to
                combinations of intermediates and initial reactants

        Returns:
            None
        """
        self._starters = starters
        self._all_targets = targets
        self._selected_target = [targets[0]]  # take first entry to be first designated target
        self._cost_function = cost_function
        self._complex_loopback = complex_loopback
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions
        self._temp = temp

        if False in [isinstance(starter, ComputedEntry) for starter in starters] or False in [
                isinstance(target, ComputedEntry) for target in targets]:
            raise TypeError("Starters and target must be (or inherit from) ComputedEntry objects.")

        g = nx.DiGraph()
        vis = nx.DiGraph()  # simpler graph for visualization

        starters = set(self._starters)
        target = set(self._selected_target)

        starter_entries = RxnEntries(starters, "s")
        target_entry = RxnEntries(target, "t")

        g.add_nodes_from([starter_entries, target_entry])

        self.logger.info("Generating reactions...")

        for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
            reactants = set(r)
            reactant_entries = RxnEntries(reactants, "r")
            g.add_node(reactant_entries)

            vis.add_node(RxnEntries(reactants, None), path=0)

            if reactants.issubset(starters):
                g.add_edge(starter_entries, reactant_entries, weight=0)

        for p in self.generate_all_combos(self._filtered_entries, self._max_num_components):
            products = set(p)
            product_entries = RxnEntries(products, "p")
            g.add_node(product_entries)

            if complex_loopback:
                for c in self.generate_all_combos(products.union(starters), self._max_num_components):
                    combo = set(c)
                    g.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)
                    if combo != products:
                        vis.add_edge(RxnEntries(products, None), RxnEntries(combo, None), weight=0, path=0)
            else:
                g.add_edge(product_entries, RxnEntries(products, "r"), weight=0)

            if target.issubset(products):
                g.add_edge(product_entries, target_entry, weight=0)  # add edge connecting to target

            for r in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                reactants = set(r)
                if products == reactants:
                    continue  # removes many identity-like reactions

                reactants_elems = set([elem for entry in reactants for elem in entry.composition.elements])
                products_elems = set([elem for entry in products for elem in entry.composition.elements])

                if reactants_elems != products_elems:
                    continue  # removes reactions which change chemical systems

                reactant_entries = RxnEntries(reactants, "r")

                try:
                    rxn_forwards = ComputedReaction(list(reactants), list(products))
                    rxn_backwards = ComputedReaction(list(products), list(reactants))
                except ReactionError:
                    continue

                reactants_comps = {reactant.composition.reduced_composition for reactant in reactants}
                products_comps = {product.composition.reduced_composition for product in products}

                if (True in (abs(rxn_forwards.coeffs) < rxn_forwards.TOLERANCE)) or (reactants_comps != set(rxn_forwards.reactants)) or (
                        products_comps != set(rxn_forwards.products)):
                    continue  # removes reactions which have components that either change sides or disappear

                total_num_atoms = sum([rxn_forwards.get_el_amount(elem) for elem in rxn_forwards.elements])
                dg_per_atom_f = rxn_forwards.calculated_reaction_energy / total_num_atoms
                dg_per_atom_b = rxn_backwards.calculated_reaction_energy / total_num_atoms

                densities = np.array([entry.structure.density for entry in rxn_forwards.all_entries])
                info_entropies = np.array([entry.parameters["info_entropy"] for entry in rxn_forwards.all_entries])
                total_coeffs = sum([abs(coeff) for coeff in rxn_forwards.coeffs])

                #d_density = np.dot(rxn_forwards.coeffs, densities) / total_coeffs
                d_info_entropy = np.dot(rxn_forwards.coeffs, info_entropies) / total_coeffs

                weight_f = self.get_rxn_cost(dg_per_atom_f, cost_function, temp, d_info_entropy) #d_density)
                weight_b = self.get_rxn_cost(dg_per_atom_b, cost_function, temp)
                g.add_edge(reactant_entries, product_entries, weight=weight_f, rxn=rxn_forwards)

                if dg_per_atom_f < 1:
                    vis.add_edge(RxnEntries(reactants, None), RxnEntries(products, None), weight=weight_f, rxn=rxn_forwards, path=0)
                if dg_per_atom_b < 1:
                    vis.add_edge(RxnEntries(products, None), RxnEntries(reactants, None), weight=weight_b, rxn=rxn_backwards, path=-0)

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
        self._vis = vis

    def find_k_shortest_paths(self, k, target=None):
        """
        Finds k shortest paths to designated target using Yen's Algorithm (as defined in Networkx)

        Args:
            k (int): desired number of shortest pathways (ranked by cost)

        Returns:
            [RxnPathway]: list of RxnPathway objects
        """
        starters = RxnEntries(set(self._starters), "s")

        if target is None:
            target = self._selected_target

        target = RxnEntries(set(target), "t")

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
                        if path_products:
                            vis_reactants = RxnEntries(path_reactants.entries, None)
                            vis_products = RxnEntries(path_products.entries, None)
                            nx.set_edge_attributes(self._vis, {(vis_products, vis_reactants): {"path": {"path": 10}}})
                    elif step.description == "P":
                        path_products = step
                        rxn_data = self._rxn_network.get_edge_data(path_reactants, path_products)

                        vis_reactants = RxnEntries(path_reactants.entries, None)
                        vis_products = RxnEntries(path_products.entries, None)
                        nx.set_edge_attributes(self._vis, {(vis_reactants, vis_products): {"path": 10}})
                        nx.set_node_attributes(self._vis, {vis_reactants: {"path": 10}})
                        nx.set_node_attributes(self._vis, {vis_products: {"path": 10}})

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

    def find_redox_paths(self, num_paths_to_consider=5):
        """
        Iterates through possible starting phases (+ O2) to find shortest paths to some desired target
        (e.g. an oxidized material). Useful for finding thermochemical redox pathways.

        Args:
            num_paths_to_consider (int): how many shortest paths to consider when trying to get to desired target
        Returns:
            [RxnPathway]: Sorted list of all reaction pathways
        """

        paths_from_all_starters = set()
        oxygen_entry = self._pd.el_refs[Element("O")]
        filtered_entries = self._filtered_entries.copy()
        filtered_entries.remove(self._selected_target[0])
        filtered_entries.remove(oxygen_entry)

        for s in self.generate_all_combos(filtered_entries, self._max_num_components):
            starters = set(s)
            starters.add(oxygen_entry)
            try:
                print(f"PATHS from {[starter.composition.reduced_formula for starter in starters]} \n")
                self.set_starters(starters, connect_direct=True)
                paths_from_all_starters.update(self.find_k_shortest_paths(num_paths_to_consider))
            except Exception as e:
                print(e)
                continue

        return sorted(paths_from_all_starters, key=lambda path: path.total_weight)

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

        net_rxn = ComputedReaction(self._starters, targets)
        print(f"NET RXN: {net_rxn} \n")

        for target in targets:
            print(f"PATHS to {target.composition.reduced_formula} \n")
            self.set_target(target)
            paths_to_all_targets.update(self.find_k_shortest_paths(k))

        if consider_remnant_rxns:
            starters_and_targets = set(targets + self._starters)
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

        return sorted(list(balanced_total_paths), key=lambda combined_path: combined_path.total_weight)

    def find_remnant_paths(self, intermediates, targets, max_num_combos=2):
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

    def set_starters(self, starters, connect_direct = False):
        """
        Replaces network's previous starter nodes with provided new starters.
            Recreates edges that link products back to reactants.

        Args:
            starters ([ComputedEntry]): list of new starter entries

        Returns:
            None
        """
        for node in self._rxn_network.nodes():
            if node.description == "S":
                self._rxn_network.remove_node(node)

                starters = set(starters)
                starter_entries = RxnEntries(starters, "s")

                self._rxn_network.add_node(starter_entries)

                if connect_direct:
                    self._rxn_network.add_edge(starter_entries, RxnEntries(starters, "r"), weight=0)
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

    def set_target(self, target):
        """
        Replaces network's previous target node with provided new target.

        Args:
            target (ComputedEntry): entry of new target

        Returns:
            None
        """
        if target in self._selected_target:
            return

        target = [target]
        self._selected_target = target

        for node in self._rxn_network.nodes():
            if node.description == "T":
                self._rxn_network.remove_node(node)

                target = set(target)
                target_entry = RxnEntries(target, "t")

                for p in self.generate_all_combos(self._filtered_entries, self._max_num_components):
                    products = set(p)
                    if target.issubset(products):
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
        weighted_params = energy - d_info_entropy - 0.0*d_density
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

    def __repr__(self):
        return str(self._rxn_network)
