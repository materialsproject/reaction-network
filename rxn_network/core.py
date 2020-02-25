import logging
from itertools import combinations, chain

import numpy as np
from scipy.constants import physical_constants

import graph_tool.all as gt
import queue

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError
from matminer.featurizers.structure import StructuralComplexity

from rxn_network.helpers import RxnEntries, RxnPathway, CombinedPathway
from rxn_network.analysis import PathwayAnalysis


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"
__date__ = "February 25, 2020"


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph in graph-tool which enumerates and
        explores all possible chemical reactions (edges) between phase combinations (nodes) in a
        chemical system.
    """
    def __init__(self, entries, max_num_phases=2, include_metastable=False,
                 include_polymorphs=False, include_info_entropy=False):
        """
        Constructs a ReactionNetwork object with necessary initialization steps. This does not yet compute the graph.

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-like objects to consider in network
            max_num_phases (int): maximum number of phases allowed on each side of the reaction (default 2)
            include_metastable (float or bool): either the specified cutoff for energy above hull, or False if
                considering only stable entries
            include_polymorphs (bool): Whether or not to consider non-ground state polymorphs (defaults to False)
        """

        self.logger = logging.getLogger('ReactionNetwork')
        self.logger.setLevel("INFO")

        self._all_entries = entries
        self._pd = PhaseDiagram(entries)
        self._max_num_phases = max_num_phases
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs

        self._starters = None
        self._starters_v = None
        self._all_targets = None
        self._target_v = None
        self._current_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._temp = None
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        self._rxn_network = None

        self._filtered_entries = self.filter_entries(self._pd, include_metastable, include_polymorphs)
        self._all_entry_combos = [set(combo) for combo in self.generate_all_combos(self._filtered_entries,
                                                                                   self._max_num_phases)]

        if include_info_entropy:
            for entry in self._filtered_entries:
                info_entropy = StructuralComplexity().featurize(entry.structure)[0]
                entry.parameters = {"info_entropy": info_entropy}

        filtered_entries_str = ', '.join([entry.composition.reduced_formula for entry in self._filtered_entries])

        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} entries: \n{filtered_entries_str}")

    def generate_rxn_network(self, starters, targets, cost_function="softplus", temp=300, complex_loopback=True):
        """
        Generates the actual reaction network (weighted, directed graph) using graph-tool.

        Args:
            starters (list of ComputedEntries): entries for all phases which serve as the main reactants
            targets (list of ComputedEntries): entries for all phases which are the final products
            cost_function (str): name of cost function to use for entire network (e.g. "softplus")
            temp (int): temperature used for scaling cost functions
            complex_loopback (bool): whether or not to add edges looping back to
                combinations of intermediates and initial reactants

        Returns:
            None
        """
        self._starters = set(starters) if starters else None
        self._all_targets = set(targets) if targets else None
        self._current_target = {targets[0]} if targets else None  # take first entry to be first designated target
        self._cost_function = cost_function
        self._complex_loopback = complex_loopback
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions
        self._temp = temp

        if not self._starters:
            starter_entries = RxnEntries(None, "d")  # use dummy starter node
            if self._complex_loopback:
                raise ValueError("Complex loopback cannot be enabled when using a dummy starters node!")
        elif False in [isinstance(starter, ComputedEntry) for starter in self._starters] or False in [
                isinstance(target, ComputedEntry) for target in self._all_targets]:
            raise TypeError("Starters and target must be (or inherit from) ComputedEntry objects.")
        else:
            starter_entries = RxnEntries(starters, "s")

        g = gt.Graph()

        entries_dict = {}
        v_map = g.new_vertex_property("object")
        type_map = g.new_vertex_property("int")  # 0: starters, 1: reactants, 2: products, 3: target
        filter_v_map = g.new_vertex_property("bool")
        weight_map = g.new_edge_property("double")
        rxn_map = g.new_edge_property("object")
        filter_e_map = g.new_edge_property("bool")

        g.vertex_properties["entries"] = v_map
        g.vertex_properties["type"] = type_map
        g.vertex_properties["bool"] = filter_v_map
        g.edge_properties["weight"] = weight_map
        g.edge_properties["rxn"] = rxn_map
        g.edge_properties["bool"] = filter_e_map

        starters_v = g.add_vertex()
        v_map[starters_v] = starter_entries
        type_map[starters_v] = 0
        filter_v_map[starters_v] = True
        self._starters_v = starters_v

        for num, entries in enumerate(self._all_entry_combos):
            idx = 2*num + 1
            reactants = RxnEntries(entries, "R")
            products = RxnEntries(entries, "P")

            entries_dict[reactants] = idx
            type_map[idx] = 1
            v_map[idx] = reactants
            filter_v_map[idx] = True

            entries_dict[products] = idx+1
            type_map[idx+1] = 2
            v_map[idx+1] = products
            filter_v_map[idx + 1] = True

        edge_list = []

        g.add_vertex(len(entries_dict) - 1)

        target_v = g.add_vertex()
        target_entries = RxnEntries(self._current_target, "t")
        v_map[target_v] = target_entries
        type_map[target_v] = 3
        filter_v_map[target_v] = True
        self._target_v = target_v

        self.logger.info("Generating reactions...")

        for entry, v in entries_dict.items():
            phases = entry.entries

            v = g.vertex(v)

            if type_map[v] == 1:
                if phases.issubset(self._starters):
                    edge_list.append([starters_v, v, 0, None, True])
            elif type_map[v] == 2:
                if self._current_target.issubset(phases):
                    edge_list.append([v, target_v, 0, None, True])

                if complex_loopback:
                    for c in self.generate_all_combos(phases.union(self._starters), self._max_num_phases):
                        combo_phases = set(c)
                        combo_entry = RxnEntries(combo_phases, "R")
                        loopback_v = g.vertex(entries_dict[combo_entry])
                        if not combo_phases.issubset(self._starters):  # must contain at least one intermediate
                            edge_list.append([v, loopback_v, 0, None, True])

                for other_entry, other_v in entries_dict.items():
                    if not other_entry.description == "R":
                        continue

                    other_phases = other_entry.entries

                    if other_phases == phases:
                        continue  # do not consider identity-like reactions (A + B -> A + B)

                    reactants_elems = set([elem for entry in other_phases for elem in entry.composition.elements])
                    products_elems = set([elem for entry in phases for elem in entry.composition.elements])

                    if reactants_elems != products_elems:
                        continue  # do not consider reaction which changes chemical systems

                    try:
                        rxn = ComputedReaction(list(other_phases), list(phases))
                    except ReactionError:
                        continue

                    if rxn._lowest_num_errors != 0:
                        continue  # remove reaction which has components that either change sides or disappear

                    total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                    dg_per_atom = rxn.calculated_reaction_energy / total_num_atoms
                    weight = self.get_rxn_cost(dg_per_atom, cost_function, temp)

                    other_v = g.vertex(other_v)

                    edge_list.append([other_v, v, weight, rxn, True])

        g.add_edge_list(edge_list, eprops=[weight_map, rxn_map, filter_e_map])

        self.logger.info(f"Complete: Created graph with {g.num_vertices()} nodes and {g.num_edges()} edges.")
        self._rxn_network = g

    def find_k_shortest_paths(self, k):
        """
        Finds k shortest paths to designated target using Yen's Algorithm.

        Args:
            k (int): desired number of shortest pathways (ranked by cost)
            target (ComputedEntry-like object): desired target, defaults to first target when network was created

        Returns:
            [RxnPathway]: list of RxnPathway objects
        """

        paths = []
        num_found = 0

        type_map = self._rxn_network.vp["type"]
        weight_map = self._rxn_network.ep["weight"]
        rxn_map = self._rxn_network.ep["rxn"]

        for num, path in enumerate(self.yens_ksp(self._rxn_network, k, self._starters_v, self._target_v)):
            if num_found == k:
                break

            rxns = []
            weights = []

            for step, v in enumerate(path):
                if type_map[v] == 2:  # add rxn step if current node in path is a product
                    e = self._rxn_network.edge(path[step-1], v)
                    rxns.append(rxn_map[e])
                    weights.append(weight_map[e])

            rxn_pathway = RxnPathway(rxns, weights)
            paths.append(rxn_pathway)
            print(rxn_pathway, "\n")

            num_found += 1

        return paths

    def find_combined_paths(self, k, targets=None, max_num_combos=4, consider_remnant_rxns=True):
        """
        Builds k shortest paths to provided targets and then seeks to combine them to achieve a "net reaction"
            with balanced stoichiometry. In other words, the full conversion of all intermediates to final products.

        Args:
            k (int): calculated free energy of reaction
            targets ([ComputedEntries]): list of all target ComputedEntry objects, defaults to targets provided when network was created
            max_num_combos (int): upper limit on how many pathways to consider at a time (default 3).

        Returns:
            [CombinedPathway]: list of CombinedPathway objects, sorted by total cost

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
        Helper method which looks for potential reactions between found intermediates.

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

                if rxn._lowest_num_errors > 0:
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
                else:
                    for r in self.generate_all_combos(self._filtered_entries, self._max_num_phases):
                        reactants = set(r)
                        if reactants.issubset(starters):  # link starting node to reactant nodes
                            reactant_entries = RxnEntries(reactants, "r")
                            self._rxn_network.add_edge(starter_entries, reactant_entries, weight=0)
                break

        if not self._complex_loopback:
            self._starters = starters
            return

        for p in self.generate_all_combos(self._filtered_entries, self._max_num_phases):
            products = set(p)
            product_entries = RxnEntries(products, "p")

            old_loopbacks = self.generate_all_combos(list(products.union(self._starters)), self._max_num_phases)
            for combo in old_loopbacks:
                # delete old edge linking back
                self._rxn_network.remove_edge(product_entries, RxnEntries(combo, "r"))

            new_loopbacks = self.generate_all_combos(list(products.union(starters)), self._max_num_phases)
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
        g = self._rxn_network

        if target in self._current_target:
            return
        else:
            self._current_target = {target}

        v_map = g.vp["entries"]
        type_map = g.vp["type"]
        filter_v_map = g.vp["bool"]

        g.remove_vertex(self._target_v)

        new_target_entry = RxnEntries(self._current_target, "t")
        new_target_v = self._rxn_network.add_vertex()
        v_map[new_target_v] = new_target_entry
        type_map[new_target_v] = 3
        filter_v_map[new_target_v] = True
        self._target_v = new_target_v

        new_edges = []
        all_vertices = g.get_vertices(vprops=[g.vertex_index, type_map])

        for v in all_vertices[all_vertices[:, 2] == 2]:  # search for all products
            vertex = g.vertex(v[1])
            if self._current_target.issubset(v_map[vertex].entries):
                new_edges.append([vertex, new_target_v, 0, None, True])  # link all products to new target

        weight_map = g.ep["weight"]
        rxn_map = g.ep["rxn"]
        filter_e_map = g.ep["bool"]

        g.add_edge_list(new_edges, eprops=[weight_map, rxn_map, filter_e_map])

    @staticmethod
    def yens_ksp(g, num_k, starter_v, target_v, weight_prop="weight"):
        """
        Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by Antonin Lenfant.

        Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths in a Network",
        Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.
        """
        g = g.copy()

        filter_v_map = g.vp["bool"]
        filter_e_map = g.ep["bool"]
        weights = g.ep[weight_prop]

        def path_cost(vertices):
            cost = 0
            for j in range(len(vertices)-1):
                cost += g.ep["weight"][g.edge(vertices[j], vertices[j+1])]
            return cost

        path = gt.shortest_path(g, starter_v, target_v, weights=weights)[0]

        a = [path]
        a_costs = [path_cost(path)]

        b = queue.PriorityQueue()

        for k in range(1, num_k):
            for i in range(len(a[k - 1]) - 1):
                spur_v = a[k - 1][i]
                root_path = a[k - 1][:i]

                filtered_edges = []

                for path in a:
                    if len(path) - 1 > i and root_path == path[:i]:
                        e = g.edge(path[i], path[i + 1])
                        if not e:
                            continue
                        filter_e_map[e] = False
                        filtered_edges.append(e)

                g.set_edge_filter(filter_e_map)

                spur_path = gt.shortest_path(g, spur_v, target_v, weights=weights)[0]

                for e in filtered_edges:
                    filter_e_map[e] = True

                if spur_path:
                    total_path = root_path + spur_path
                    total_path_cost = path_cost(total_path)
                    b.put((total_path_cost, total_path))

            while True:
                try:
                    cost_, path_ = b.get(block=False)
                except queue.Empty:
                    break
                if path_ not in a:
                    a.append(path_)
                    a_costs.append(cost_)
                    break

        return a

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
