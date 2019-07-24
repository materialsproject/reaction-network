import logging
from itertools import combinations, chain

import numpy as np
from scipy.constants import physical_constants

import networkx as nx

from rxn_network.helpers import RxnEntries, RxnPathway, CombinedPathway

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction


class ReactionNetwork():
    def __init__(self, entries, max_num_components=2, max_e_above_hull=0.0):
        self._all_entries = entries
        self._max_num_components = max_num_components
        self._max_e_above_hull = max_e_above_hull

        self.logger = logging.getLogger('ReactionNetwork')
        self.logger.setLevel("INFO")

        self._filtered_entries = self._filter_entries(self._all_entries, self._max_e_above_hull)

        self.logger.info("Generating combinations...")
        self._all_combos = self.generate_all_combos(self._filtered_entries, self._max_num_components)
        self.logger.info(f"Found {len(self._all_combos)} combinations of entries (size <= {self._max_num_components}).")

    def generate_rxn_network(self, starters, target, cost_function):
        self._starters = starters
        self._target = target
        self._cost_function = cost_function
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        if False in [isinstance(starter, ComputedEntry) for starter in starters] or not isinstance(target, ComputedEntry):
            raise TypeError("Starters and target must be ComputedEntries.")

        G = nx.DiGraph()

        starters = set(starters)
        target = {target}

        starter_entries = RxnEntries(starters, "s")
        target_entries = RxnEntries(target, "t")

        G.add_nodes_from([starter_entries, target_entries])

        self.logger.info("Generating reactions...")

        for reactants in self._all_combos:
            reactant_entries = RxnEntries(reactants, "r")
            G.add_node(reactant_entries)

            if reactants.issubset(starters):  # link starting node to reactant nodes (no cost)
                G.add_edge(starter_entries, reactant_entries, weight=0)

        for products in self._all_combos:
            product_entries = RxnEntries(products, "p")
            G.add_node(product_entries)  # add node for products

            linking_combos = self.generate_all_combos(list(products.union(starters)), self._max_num_components)

            for combo in linking_combos:
                G.add_edge(product_entries, RxnEntries(combo, "r"),
                           weight=0)  # add edge linking products back to reactants

            if target.issubset(products):
                G.add_edge(product_entries, target_entries, weight=0)  # add edge connecting to target

            for reactants in self._all_combos:
                if products == reactants:
                    # removes many identity-like reactions
                    continue

                reactants_elems = set([elem for entry in reactants for elem in entry.composition.elements])
                products_elems = set([elem for entry in products for elem in entry.composition.elements])

                if reactants_elems != products_elems:
                    # removes reactions which change chemical systems
                    continue

                reactant_entries = RxnEntries(reactants, "r")

                try:
                    rxn = ComputedReaction(list(reactants), list(products))

                    reactants_comps = {reactant.composition.reduced_composition for reactant in reactants}
                    products_comps = {product.composition.reduced_composition for product in products}

                    if (True in (abs(rxn.coeffs) < rxn.TOLERANCE)) or (reactants_comps != set(rxn.reactants)) or (
                            products_comps != set(rxn.products)):
                        # removes reactions which have components that either change sides or disappear
                        continue
                except:
                    continue

                total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                dH_per_atom = rxn.calculated_reaction_energy / total_num_atoms

                weight = self.determine_rxn_weight(dH_per_atom, cost_function)

                G.add_edge(reactant_entries, product_entries, weight=weight, rxn=rxn)

        self.logger.info(f"Complete: Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        if cost_function == "enthalpies_positive":
            self.logger.info(f"Adjusting enthalpies up by {-round(self._most_negative_rxn,3)} eV")
            # shift all energies to be on a positive scale
            for (u, v, rxn) in G.edges.data(data='rxn'):
                if rxn != None:
                    G[u][v]["weight"] -= self._most_negative_rxn
        elif cost_function == "bipartite":
            self.logger.info(f"Adjusting enthalpies using {-round(self._most_negative_rxn,3)} eV")
            # shift all exothermic energies to be on a positive scale between 0-1
            for (u, v, rxn) in G.edges.data(data='rxn'):
                if rxn != None and (G[u][v]["weight"] < 0):
                    G[u][v]["weight"] = 1 - (G[u][v]["weight"] / self._most_negative_rxn)

        self._rxn_network = G

    def find_k_shortest_pathways(self, k):

        starters = RxnEntries(set(self._starters), "s")
        target = RxnEntries(set(self._target), "t")

        pathways = []
        num_found = 0

        for path in nx.shortest_simple_paths(self._rxn_network, starters, target, weight="weight"):

            if num_found == k:
                break

            path_reactants = None
            path_products = None
            rxns = []
            weights = []

            for step in path:
                if step.description == "Reactants":
                    path_reactants = step
                elif step.description == "Products":
                    path_products = step
                    rxn_data = self._rxn_network.get_edge_data(path_reactants, path_products)
                    rxn = rxn_data["rxn"]
                    weight = rxn_data["weight"]

                    rxns.append(rxn)
                    weights.append(weight)

            rxn_pathway = RxnPathway(rxns, weights)
            pathways.append(rxn_pathway)
            print(rxn_pathway)

            num_found += 1

            print("\n")

        return pathways

    def find_combined_paths(self, k, targets=None, max_num_combos=3):
        paths_to_all_targets = []

        if targets is None:
            targets = self._all_entries

        for target in targets:
            try:
                print(f"PATHS to {target.composition.reduced_formula} \n")
                self.set_target([target])
                paths_to_all_targets.extend(self.find_k_shortest_pathways(k))
            except:
                print(f"No (more) pathways found! \n")

        balanced_total_paths = []
        for combo in chain.from_iterable(
                [combinations(paths_to_all_targets, num) for num in range(1, max_num_combos + 1)]):
            combined_pathway = CombinedPathway(combo, targets)

            if combined_pathway.net_rxn:
                reactants_set = set(combined_pathway.net_rxn._reactant_entries)
                products_set = set(combined_pathway.net_rxn._product_entries)

                common_entries = reactants_set.intersection(products_set)
                reactants_set = reactants_set - common_entries
                products_set = products_set - common_entries

                if (set(targets).issubset(products_set)) and (set(self._starters) == reactants_set):
                    balanced_total_paths.append(combined_pathway)

                leftover_products = products_set - set(targets)

                leftover_rxns = set()

                for leftover_reactants in chain.from_iterable(
                        [combinations(leftover_products, num) for num in range(1, len(leftover_products)+1)]):
                    for target_combos in chain.from_iterable(
                            [combinations(targets, num) for num in range(1, len(targets) + 1)]):
                        try:
                            leftover_rxns.add(ComputedReaction(list(leftover_reactants),list(target_combos)))
                        except:
                            continue

                for leftover_rxn in leftover_rxns:
                    leftover_path = RxnPathway([leftover_rxn], [0])
                    all_paths = combined_pathway.paths.copy()
                    all_paths.append(leftover_path)

                    appended_combined_path = CombinedPathway(all_paths,targets)
                    print(appended_combined_path)
                    #print([e.composition.reduced_formula for e in appended_combined_path.all_reactants])
                    #print([e.composition.reduced_formula for e in appended_combined_path.all_products])

                    if appended_combined_path.net_rxn:
                        balanced_total_paths.append(combined_pathway)

        return sorted(balanced_total_paths, key=lambda combined_pathway: combined_pathway.average_weight)

    def determine_rxn_weight(self, energy, cost_function):
        if cost_function == "softplus":
            weight = self._softplus(energy, T=500)
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
            weight = self._arrhenius(energy, 100)
        elif cost_function == "enthalpies_positive":
            weight = energy
            if weight < self._most_negative_rxn:
                self._most_negative_rxn = weight
        else:
            weight = 0

        return weight

    def set_cost_function(self, cost_function):
        for (u, v, rxn) in self._rxn_network.edges.data(data='rxn'):
            if rxn != None:
                total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                dH_per_atom = rxn.calculated_reaction_energy / total_num_atoms

                weight = self.determine_rxn_weight(dH_per_atom, cost_function)
                self._rxn_network[u][v]["weight"] = weight

    def set_starters(self, starters):
        # remove previous starters node
        for node in self._rxn_network.nodes():
            if node.description == "Starters":
                self._rxn_network.remove_node(node)

                starters = set(starters)
                starter_entries = RxnEntries(starters, "s")

                for reactants in self._all_combos:
                    if reactants.issubset(starters):  # link starting node to reactant nodes
                        reactant_entries = RxnEntries(reactants, "r")
                        self._rxn_network.add_edge(starter_entries, reactant_entries, weight=0)
                break

        for products in self._all_combos:
            product_entries = RxnEntries(products, "p")

            old_linking_combos = self.generate_all_combos(list(products.union(self._starters)),
                                                          self._max_num_components)

            for combo in old_linking_combos:
                # delete old edge linking back
                self._rxn_network.remove_edge(product_entries, RxnEntries(combo, "r"))

            new_linking_combos = self.generate_all_combos(list(products.union(starters)), self._max_num_components)
            for combo in new_linking_combos:
                # add new edges linking back
                self._rxn_network.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)

        self._starters = starters

    def set_target(self, target):
        for node in self._rxn_network.nodes():
            if node.description == "Target":
                self._rxn_network.remove_node(node)

                target = set(target)
                target_entries = RxnEntries(target, "t")

                for products in self._all_combos:
                    if target.issubset(products):  # link starting node to reactant nodes
                        product_entries = RxnEntries(products, "p")
                        self._rxn_network.add_edge(product_entries, target_entries, weight=0)

                break

        self._target = target

    def _filter_entries(self, entries, e_above_hull):
        pd_of_all_entries = PhaseDiagram(entries)

        if e_above_hull == 0:
            logging.info("Using stable entries only!")
            return list(pd_of_all_entries.stable_entries)
        else:
            filtered_list = [entry for entry in entries if pd_of_all_entries.get_e_above_hull(entry) <= e_above_hull]
            return filtered_list

    def generate_all_combos(self, entries, max_num_components):
        all_combos = []
        for num_components in range(1, max_num_components + 1):
            # add combinations for all sizes between 1 --> max_num_components
            all_combos.extend([set(combo) for combo in combinations(entries, num_components)])

        return all_combos

    @staticmethod
    def _arrhenius(energy, T):
        kb = physical_constants["Boltzmann constant in eV/K"][0]
        return np.exp(energy / (kb * T))

    @staticmethod
    def _softplus(energy, T=1.0):
        return np.log(1 + (273/T)*np.exp(energy))

    @property
    def all_entries(self):
        return self._all_entries

    @property
    def filtered_entries(self):
        return self._filtered_entries

    @property
    def rxn_network(self):
        return self._rxn_network

#     def __str__(self):
#         return f"{self._description}: {str([entry.composition.reduced_formula for entry in self._entries])}"

#     def __eq__(self, other):
#         if isinstance(other, self.__class__):
#             return self.as_dict() == other.as_dict()
#         else:
#             return False

#     def __hash__(self):
#         return hash((self._description, frozenset(self._entries)))

#     __repr__ = __str__

