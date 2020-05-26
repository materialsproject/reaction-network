import logging
import os
from itertools import combinations, chain, groupby

import numpy as np
from scipy.constants import physical_constants
import pandas as pd

import graph_tool.all as gt
import queue

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError
from matminer.featurizers.structure import StructuralComplexity

from rxn_network.helpers import *
from rxn_network.analysis import *

__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"
__date__ = "May 21, 2020"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph in graph-tool which enumerates and
        traverses possible chemical reactions (edges) between phase combinations (vertices) in a
        chemical system.
    """
    def __init__(self, entries, max_num_phases=2, include_metastable=False,
                 include_polymorphs=False, include_info_entropy=False, include_literature=False):
        """
        Constructs a ReactionNetwork object with necessary initialization steps. This does not yet compute the graph.

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-like objects to consider in network
            max_num_phases (int): maximum number of phases allowed on each side of the reaction (default 2)
            include_metastable (float or bool): either the specified cutoff for energy above hull, or False if
                considering only stable entries
            include_polymorphs (bool): Whether or not to consider non-ground state polymorphs (defaults to False)
            include_info_entropy (bool): Whether or not to consider Shannon information entropy as a cost metric
        """

        self.logger = logging.getLogger('ReactionNetwork')
        self.logger.setLevel("INFO")

        self._all_entries = entries
        self._pd = PhaseDiagram(entries)
        self._max_num_phases = max_num_phases
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs
        self._include_literature = include_literature

        self._starters = None
        self._starters_v = None
        self._all_targets = None
        self._target_v = None
        self._current_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._temp = None
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        self.g = None

        self._filtered_entries = self.filter_entries(self._pd, include_metastable, include_polymorphs)
        self._all_entry_combos = [set(combo) for combo in self.generate_all_combos(self._filtered_entries,
                                                                                   self._max_num_phases)]

        if include_info_entropy:
            for entry in self._filtered_entries:
                info_entropy = StructuralComplexity().featurize(entry.structure)[0]
                entry.parameters = {"info_entropy": info_entropy}

        if include_literature:
            all_chemsyses = []
            for i in range(len(self._pd.elements)):
                for els in combinations([str(el) for el in self._pd.elements], i + 1):
                    all_chemsyses.append('-'.join(sorted(els)))

            lit_dfs = []
            for df in include_literature:
                lit_dfs.append(pd.read_pickle(os.path.join(MODULE_DIR, df)))
            literature_df = pd.concat(lit_dfs)
            self.literature_df = literature_df[literature_df["chemsys"].isin(all_chemsyses)]

        filtered_entries_str = ', '.join([entry.composition.reduced_formula for entry in self._filtered_entries])

        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} entries: \n{filtered_entries_str}")

    def generate_rxn_network(self, starters, targets, cost_function="softplus", temp=300, complex_loopback=True):
        """
        Generates and stores the actual reaction network (weighted, directed graph) using graph-tool.

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

        g.vp["entries"] = g.new_vertex_property("object")
        g.vp["type"] = g.new_vertex_property("int")  # 0: starters, 1: reactants, 2: products, 3: target
        g.vp["bool"] = g.new_vertex_property("bool")
        g.vp["path"] = g.new_vertex_property("bool")
        g.vp["chemsys"] = g.new_vertex_property("string")

        g.ep["weight"] = g.new_edge_property("double")
        g.ep["rxn"] = g.new_edge_property("object")
        g.ep["bool"] = g.new_edge_property("bool")
        g.ep["path"] = g.new_edge_property("bool")
        g.ep["doi"] = g.new_edge_property("object")

        starters_v = g.add_vertex()

        g.vp["entries"][starters_v] = starter_entries
        g.vp["type"][starters_v] = 0
        g.vp["bool"][starters_v] = True
        g.vp["path"][starters_v] = True
        g.vp["chemsys"][starters_v] = starter_entries.chemical_system
        self._starters_v = starters_v

        idx = 0
        for num, entries in enumerate(self._all_entry_combos):
            idx = 2*num + 1
            reactants = RxnEntries(entries, "R")
            products = RxnEntries(entries, "P")
            chemsys = reactants.chemical_system
            if chemsys not in entries_dict:
                entries_dict[chemsys] = dict({"R": {}, "P": {}})

            entries_dict[chemsys]["R"][reactants] = idx
            g.vp["entries"][idx] = reactants
            g.vp["type"][idx] = 1
            g.vp["bool"][idx] = True
            g.vp["path"][idx] = False
            g.vp["chemsys"][idx] = chemsys

            entries_dict[chemsys]["P"][products] = idx+1
            g.vp["entries"][idx+1] = products
            g.vp["type"][idx+1] = 2
            g.vp["bool"][idx + 1] = True
            g.vp["path"][idx+1] = False
            g.vp["chemsys"][idx+1] = chemsys

        g.add_vertex(idx+1)

        target_v = g.add_vertex()

        target_entries = RxnEntries(self._current_target, "t")
        g.vp["entries"][target_v] = target_entries
        g.vp["type"][target_v] = 3
        g.vp["bool"][target_v] = True
        g.vp["path"][target_v] = True
        g.vp["chemsys"][target_v] = target_entries.chemical_system
        self._target_v = target_v

        self.logger.info("Generating reactions...")

        edge_list = []
        for chemsys, vertices in entries_dict.items():
            for entry, v in vertices["R"].items():
                phases = entry.entries

                v = g.vertex(v)

                if starter_entries.description == "D" or phases.issubset(self._starters):
                    edge_list.append([starters_v, v, 0, None, True, False, None])

            for entry, v in vertices["P"].items():
                phases = entry.entries
                if self._current_target.issubset(phases):
                    edge_list.append([v, target_v, 0, None, True, False, None])

                if complex_loopback:
                    for c in self.generate_all_combos(phases.union(self._starters), self._max_num_phases):
                        combo_phases = set(c)
                        combo_entry = RxnEntries(combo_phases, "R")
                        loopback_v = g.vertex(entries_dict[combo_entry.chemical_system]["R"][combo_entry])
                        if not combo_phases.issubset(self._starters):  # must contain at least one intermediate
                            edge_list.append([v, loopback_v, 0, None, True, False, None])

                for other_entry, other_v in vertices["R"].items():
                    other_phases = other_entry.entries

                    if other_phases == phases:
                        continue  # do not consider identity-like reactions (A + B -> A + B)

                    try:
                        rxn = ComputedReaction(list(other_phases), list(phases))
                    except ReactionError:
                        continue

                    if rxn._lowest_num_errors != 0:
                        continue  # remove reaction which has components that either change sides or disappear

                    doi = None
                    if self._include_literature:
                        doi_df = self.literature_df[self.literature_df["rxn"] == rxn]
                        if not doi_df.empty:
                            doi = doi_df["doi"].values[0]

                    total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                    dg_per_atom = rxn.calculated_reaction_energy / total_num_atoms
                    weight = self.get_rxn_cost(dg_per_atom, cost_function, temp)

                    other_v = g.vertex(other_v)

                    edge_list.append([other_v, v, weight, rxn, True, False, doi])

        g.add_edge_list(edge_list, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"], g.ep["doi"]])

        self.logger.info(f"Created graph with {g.num_vertices()} nodes and {g.num_edges()} edges.")
        self.g = g

    def find_k_shortest_rxn_pathways(self, k):
        """
        Finds k shortest paths to designated target using Yen's Algorithm.

        Args:
            k (int): desired number of shortest pathways (ranked by cost)

        Returns:
            [RxnPathway]: list of RxnPathway objects
        """
        g = self.g

        paths = []
        num_found = 0

        for num, path in enumerate(self.yens_ksp(g, k, self._starters_v, self._target_v)):
            if num_found == k:
                break

            rxns = []
            weights = []

            for step, v in enumerate(path):
                g.vp["path"][v] = True

                if g.vp["type"][v] == 2:  # add rxn step if current node in path is a product
                    e = g.edge(path[step-1], v)
                    g.ep["path"][e] = True
                    rxns.append(g.ep["rxn"][e])
                    weights.append(g.ep["weight"][e])

            rxn_pathway = RxnPathway(rxns, weights)
            paths.append(rxn_pathway)
            print(rxn_pathway, "\n")

            num_found += 1

        return paths

    def find_combined_paths(self, k, targets=None, max_num_combos=4, rxns_only=False, consider_remnant_rxns=True):
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
        if rxns_only:
            paths_to_all_targets = dict()
        else:
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
            paths = self.find_k_shortest_rxn_pathways(k)
            if rxns_only:
                paths = {rxn: cost for path in paths for (rxn, cost) in zip(path.rxns, path.costs)}
            paths_to_all_targets.update(paths)

        if consider_remnant_rxns:
            starters_and_targets = targets | self._starters

            if rxns_only:
                intermediates = {entry for rxn in paths_to_all_targets for entry in rxn.all_entries} - starters_and_targets
            else:
                intermediates = {entry for path in paths_to_all_targets
                                 for rxn in path.all_rxns for entry in rxn.all_entries} - starters_and_targets
            remnant_paths = self.find_remnant_paths(intermediates, targets)

            if remnant_paths:
                print("Remnant Reactions \n")
                print(remnant_paths, "\n")
                if rxns_only:
                    remnant_paths = {rxn: cost for path in remnant_paths for (rxn, cost) in zip(path.rxns, path.costs)}
                paths_to_all_targets.update(remnant_paths)

        balanced_total_paths = set()
        if rxns_only:
            paths_to_try = list(paths_to_all_targets.keys())
        else:
            paths_to_try = paths_to_all_targets

        for combo in self.generate_all_combos(paths_to_try, max_num_combos):
            if rxns_only:
                combined_pathway = BalancedPathway({p: paths_to_all_targets[p] for p in combo}, net_rxn)
            else:
                combined_pathway = CombinedPathway(combo, net_rxn)

            if combined_pathway.is_balanced:
                balanced_total_paths.add(combined_pathway)

        analysis = PathwayAnalysis(self, balanced_total_paths)
        return sorted(list(balanced_total_paths), key=lambda x: x.total_cost), analysis

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
        for (u, v, rxn) in self.g.edges.data(data='rxn'):
            if rxn:
                total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                dg_per_atom = rxn.calculated_reaction_energy / total_num_atoms

                weight = self.get_rxn_cost(dg_per_atom, cost_function, temp)
                self.g[u][v]["weight"] = weight

    def set_starters(self, starters, connect_direct = False, dummy_exclusive = False):
        """
        Replaces network's previous starter nodes with provided new starters.
            Recreates edges that link products back to reactants.

        Args:
            starters ([ComputedEntry]): list of new starter entries

        Returns:
            None
        """

        for node in self.g.nodes():
            if node.description == "S" or node.description == "D":
                self.g.remove_node(node)

                if not starters or dummy_exclusive:
                    starter_entries = RxnEntries(None, "d")
                else:
                    starters = set(starters)
                    starter_entries = RxnEntries(starters, "s")

                self.g.add_node(starter_entries)

                if connect_direct:
                    self.g.add_edge(starter_entries, RxnEntries(starters, "r"), weight=0)
                else:
                    for r in self.generate_all_combos(self._filtered_entries, self._max_num_phases):
                        reactants = set(r)
                        if reactants.issubset(starters):  # link starting node to reactant nodes
                            reactant_entries = RxnEntries(reactants, "r")
                            self.g.add_edge(starter_entries, reactant_entries, weight=0)
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
                self.g.remove_edge(product_entries, RxnEntries(combo, "r"))

            new_loopbacks = self.generate_all_combos(list(products.union(starters)), self._max_num_phases)
            for combo in new_loopbacks:
                # add new edges linking back
                self.g.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)

        self._starters = starters

    def set_target(self, target):
        """
        Replaces network's previous target node with provided new target.

        Args:
            target (ComputedEntry): entry of new target

        Returns:
            None
        """
        g = self.g

        if target in self._current_target:
            return
        else:
            self._current_target = {target}

        g.remove_vertex(self._target_v)

        new_target_entry = RxnEntries(self._current_target, "t")
        new_target_v = g.add_vertex()
        g.vp["entries"][new_target_v] = new_target_entry
        g.vp["type"][new_target_v] = 3
        g.vp["bool"][new_target_v] = True
        self._target_v = new_target_v

        new_edges = []
        all_vertices = g.get_vertices(vprops=[g.vertex_index, g.vp["type"]])

        for v in all_vertices[all_vertices[:, 2] == 2]:  # search for all products
            vertex = g.vertex(v[1])
            if self._current_target.issubset(g.vp["entries"][vertex].entries):
                new_edges.append([vertex, new_target_v, 0, None, True])  # link all products to new target

        g.add_edge_list(new_edges, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"]])

    @staticmethod
    def yens_ksp(g, num_k, starter_v, target_v, edge_prop="bool", weight_prop="weight"):
        """
        Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by Antonin Lenfant.

        Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths in a Network",
        Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.
        """

        def path_cost(vertices):
            cost = 0
            for j in range(len(vertices)-1):
                cost += g.ep["weight"][g.edge(vertices[j], vertices[j+1])]
            return cost

        path = gt.shortest_path(g, starter_v, target_v, weights=g.ep[weight_prop])[0]

        a = [path]
        a_costs = [path_cost(path)]

        b = queue.PriorityQueue()  # automatically sorts by path cost (priority)

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
                        g.ep[edge_prop][e] = False
                        filtered_edges.append(e)

                gv = gt.GraphView(g, efilt=g.ep[edge_prop])

                spur_path = gt.shortest_path(gv, spur_v, target_v, weights=g.ep[weight_prop])[0]

                for e in filtered_edges:
                    g.ep[edge_prop][e] = True

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
    def starters(self):
        return self._starters

    @property
    def all_targets(self):
        return self._all_targets

    def __repr__(self):
        return str(self.g)
