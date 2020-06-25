import logging
from itertools import combinations, chain, groupby
from tqdm import tqdm

import numpy as np
from scipy.constants import physical_constants

import graph_tool.all as gt
import queue

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError

from matminer.featurizers.structure import StructuralComplexity
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

from rxn_network.helpers import *
from rxn_network.analysis import *


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"
__date__ = "May 21, 2020"


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph in graph-tool that is a dense network of
        all possible chemical reactions (edges) between phase combinations (vertices) in a
        chemical system. Reaction pathway hypotheses are generated using pathfinding methods.
    """
    def __init__(self, entries, max_num_phases=2, temp=300, extend_entries=None, include_metastable=False,
                 include_struct_similarity=False, include_polymorphs=False, include_info_entropy=False):
        """
        Initializes ReactionNetwork object with necessary preprocessing steps. This does not yet compute the graph.

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-like objects to consider in network.
                These can be acquired from Materials Project (using MPRester) or created manually in pymatgen.
            max_num_phases (int): maximum number of phases allowed on each side of the reaction (default 2).
                Note that n > 2 leads to significant (and often prohibitive) combinatorial explosion.
            extend entries([ComputedStructureEntry]): list of ComputedStructureEntry-like objects which will
                be included in the network even after filtering for thermodynamic stability.
            include_metastable (float or bool): either a) the specified cutoff for energy per atom (eV/atom)
                above hull, or b) True/False if considering only stable vs. all entries.
                An energy cutoff of 0.1 eV/atom is a reasonable starting threshold for thermodynamic stability.
            include_struct_similarity (bool): Whether or not to include structural similarity metrics in the
                cost function
            include_polymorphs (bool): Whether or not to consider non-ground state polymorphs.
            include_info_entropy (bool): (BETA) -- Whether or not to consider Shannon information entropy as a
                cost metric.
        """

        self.logger = logging.getLogger('ReactionNetwork')
        self.logger.setLevel("INFO")

        # Chemical system / phase diagram variables
        self._all_entries = entries
        self._max_num_phases = max_num_phases
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs
        self._temp = temp
        self._elements = {elem for entry in self.all_entries for elem in entry.composition.elements}
        self._gibbs_entries = GibbsComputedStructureEntry.from_entries(self._all_entries, self._temp)
        self._pd_dict, self._filtered_entries = self._filter_entries(self._gibbs_entries,
                                                                     include_metastable, include_polymorphs)
        self.fingerprints = dict()

        if extend_entries:
            self._filtered_entries.extend(extend_entries)

        if include_struct_similarity:
            ssf = SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
                stats=('mean', 'std_dev', 'minimum', 'maximum'))
            for entry in self._filtered_entries:
                self.fingerprints[entry] = np.array(ssf.featurize(entry.structure))

        self._all_entry_combos = [set(combo) for combo in self._generate_all_combos(self._filtered_entries,
                                                                                    self._max_num_phases)]

        # Graph variables used during graph creation
        self._precursors = None
        self._precursors_v = None
        self._all_targets = None
        self._target_v = None
        self._current_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._entries_dict = None
        self._most_negative_rxn = float("inf")  # used for shifting reaction energies in some cost functions

        self._g = None  # Graph object in graph-tool

        if include_info_entropy:  # This is in BETA -- use with caution (no evidence this is realistic!)
            for entry in self._filtered_entries:
                info_entropy = StructuralComplexity().featurize(entry.structure)[0]
                entry.parameters = {"info_entropy": info_entropy}

        filtered_entries_str = ', '.join([entry.composition.reduced_formula for entry in self._filtered_entries])
        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} entries: \n{filtered_entries_str}")

    def generate_rxn_network(self, precursors=None, targets=None, cost_function="softplus", complex_loopback=True):
        """
        Generates and stores the reaction network (weighted, directed graph) using graph-tool. In practice,
        the main iterative loop will start taking a considerable amount of time when using >40-50 phases. As
        of now, temperature must be provided in the range [300 K, 2000 K] in steps of 100 K.

        Args:
            precursors (list of ComputedEntries): entries for all phases which serve as the main reactants;
                if None,a "dummy" node is used to represent any possible set of precursors.
            targets (list of ComputedEntries): entries for all phases which are the final products; if None,
                a "dummy" node is used to represent any possible set of targets.
            cost_function (str): name of cost function to use for entire network (e.g. "softplus").
            temp (int): temperature used for scaling cost functions.
            complex_loopback (bool): if True, adds zero-weight edges which "loop back" to allow for multi-step
                reactions, i.e. original precursors can appear many times and in different steps.
        """

        self._precursors = set(precursors) if precursors else None
        self._all_targets = set(targets) if targets else None
        self._current_target = {targets[0]} if targets else None  # take first entry to be first designated target
        self._cost_function = cost_function
        self._complex_loopback = complex_loopback

        if not self._precursors:
            precursors_entries = RxnEntries(None, "d")  # use dummy precursors node
            if self._complex_loopback:
                raise ValueError("Complex loopback cannot be enabled when using a dummy precursors node!")
        elif False in [isinstance(precursor, ComputedEntry) for precursor in self._precursors] or False in [
                isinstance(target, ComputedEntry) for target in self._all_targets]:
            raise TypeError("Precursors and targets must be ComputedEntry-like objects.")
        else:
            precursors_entries = RxnEntries(precursors, "s")

        g = gt.Graph()  # initialization of graph object in graph-tool

        g.vp["entries"] = g.new_vertex_property("object")
        g.vp["type"] = g.new_vertex_property("int")  # 0: precursors, 1: reactants, 2: products, 3: target
        g.vp["bool"] = g.new_vertex_property("bool")
        g.vp["path"] = g.new_vertex_property("bool")  # whether node is part of path
        g.vp["chemsys"] = g.new_vertex_property("string")

        g.ep["weight"] = g.new_edge_property("double")
        g.ep["rxn"] = g.new_edge_property("object")
        g.ep["bool"] = g.new_edge_property("bool")
        g.ep["path"] = g.new_edge_property("bool")  # whether edge is part of path

        precursors_v = g.add_vertex()

        g.vp["entries"][precursors_v] = precursors_entries
        g.vp["type"][precursors_v] = 0
        g.vp["bool"][precursors_v] = True
        g.vp["path"][precursors_v] = True
        g.vp["chemsys"][precursors_v] = precursors_entries.chemsys
        self._precursors_v = precursors_v

        entries_dict = {}
        idx = 0
        for num, entries in enumerate(self._all_entry_combos):
            idx = 2*num + 1
            reactants = RxnEntries(entries, "R")
            products = RxnEntries(entries, "P")
            chemsys = reactants.chemsys
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

        g.add_vertex(idx+1)  # add all precursors, reactant, and product vertices
        target_v = g.add_vertex()  # add target vertex

        target_entries = RxnEntries(self._current_target, "t")
        g.vp["entries"][target_v] = target_entries
        g.vp["type"][target_v] = 3
        g.vp["bool"][target_v] = True
        g.vp["path"][target_v] = True
        g.vp["chemsys"][target_v] = target_entries.chemsys
        self._target_v = target_v

        self.logger.info("Generating reactions by chemical subsystem...")

        edge_list = []
        for chemsys, vertices in tqdm(entries_dict.items()):
            for entry, v in vertices["R"].items():
                phases = entry.entries

                v = g.vertex(v)

                if precursors_entries.description == "D" or phases.issubset(self._precursors):
                    edge_list.append([precursors_v, v, 0, None, True, False])

            for entry, v in vertices["P"].items():
                phases = entry.entries
                if self._current_target.issubset(phases):
                    edge_list.append([v, target_v, 0, None, True, False])

                if complex_loopback:
                    for c in self._generate_all_combos(phases.union(self._precursors), self._max_num_phases):
                        combo_phases = set(c)
                        combo_entry = RxnEntries(combo_phases, "R")
                        loopback_v = g.vertex(entries_dict[combo_entry.chemsys]["R"][combo_entry])
                        if not combo_phases.issubset(self._precursors):  # must contain at least one intermediate
                            edge_list.append([v, loopback_v, 0, None, True, False])

                for other_entry, other_v in vertices["R"].items():
                    other_phases = other_entry.entries

                    if other_phases == phases:
                        continue  # do not consider identity-like reactions (e.g. A + B -> A + B)

                    try:
                        rxn = ComputedReaction(list(other_phases), list(phases))
                    except ReactionError:
                        continue

                    if rxn._lowest_num_errors != 0:
                        continue  # remove reaction which has components that change sides or disappear

                    weight = self._get_rxn_cost(rxn, cost_function, self._temp, self.fingerprints)
                    other_v = g.vertex(other_v)
                    edge_list.append([other_v, v, weight, rxn, True, False])

        g.add_edge_list(edge_list, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]])

        self._entries_dict = entries_dict
        self.logger.info(f"Created graph with {g.num_vertices()} nodes and {g.num_edges()} edges.")
        self._g = g

    def find_k_shortest_paths(self, k, verbose=True):
        """
        Finds k shortest paths to designated target using Yen's Algorithm.

        Args:
            k (int): desired number of shortest pathways (ranked by cost)
            verbose (bool): whether to print all identified pathways

        Returns:
            [RxnPathway]: list of RxnPathway objects
        """
        g = self._g

        paths = []

        for num, path in enumerate(self._yens_ksp(g, k, self._precursors_v, self._target_v)):
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

        if verbose:
            for path in paths:
                print(path,"\n")

        return paths

    def find_all_rxn_pathways(self, k, targets=None, max_num_combos=4, rxns_only=False, consider_remnant_rxns=True):
        """
        Builds k shortest paths to provided targets and then seeks to combine them to achieve a "net reaction"
            with balanced stoichiometry. In other words, the full conversion of all intermediates to final products.

        Args:
            k (int): calculated free energy of reaction.
            targets ([ComputedEntries]): list of all target ComputedEntry objects; defaults to targets provided
                when network was created.
            max_num_combos (int): upper limit on how many reactions to consider at a time (default 4).
            rxns_only (bool): Whether to consider combinations of reactions (True) or full pathways (False)
            consider_remnant_rxns (bool): Whether to consider "crossover" reactions between intermediates on
                other pathways. This can be crucial for generating realistic predictions.

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

        net_rxn = ComputedReaction(list(self._precursors), list(targets))
        print(f"NET RXN: {net_rxn} \n")

        for target in targets:
            print(f"PATHS to {target.composition.reduced_formula} \n")
            self.set_target(target)
            paths = self.find_k_shortest_paths(k)
            if rxns_only:
                paths = {rxn: cost for path in paths for (rxn, cost) in zip(path.rxns, path.costs)}
            paths_to_all_targets.update(paths)

        if consider_remnant_rxns:
            def find_remnant_rxns():
                all_remnant_rxns = set()
                for reactants_combo in self._generate_all_combos(intermediates, self._max_num_phases):
                    for products_combo in self._generate_all_combos(targets, self._max_num_phases):
                        try:
                            rxn = ComputedReaction(list(reactants_combo), list(products_combo))
                        except ReactionError:
                            continue
                        if rxn._lowest_num_errors > 0:
                            continue
                        path = RxnPathway([rxn], [self._get_rxn_cost(
                            rxn, self._cost_function, self._temp, self.fingerprints)])
                        all_remnant_rxns.add(path)
                return all_remnant_rxns

            precursors_and_targets = targets | self._precursors

            if rxns_only:
                intermediates = {entry for rxn in paths_to_all_targets for entry in rxn.all_entries} - precursors_and_targets
            else:
                intermediates = {entry for path in paths_to_all_targets
                                 for rxn in path.all_rxns for entry in rxn.all_entries} - precursors_and_targets
            remnant_rxns = find_remnant_rxns()

            if remnant_rxns:
                print("Remnant Reactions \n")
                print(remnant_rxns, "\n")
                if rxns_only:
                    remnant_rxns = {rxn: cost for path in remnant_rxns for (rxn, cost) in zip(path.rxns, path.costs)}
                paths_to_all_targets.update(remnant_rxns)

        balanced_total_paths = set()
        if rxns_only:
            paths_to_try = list(paths_to_all_targets.keys())
        else:
            paths_to_try = paths_to_all_targets

        for combo in self._generate_all_combos(paths_to_try, max_num_combos):
            if rxns_only:
                combined_pathway = BalancedPathway({p: paths_to_all_targets[p] for p in combo}, net_rxn)
            else:
                combined_pathway = CombinedPathway(combo, net_rxn)

            if combined_pathway.is_balanced:
                balanced_total_paths.add(combined_pathway)

        analysis = PathwayAnalysis(self, balanced_total_paths)
        return sorted(list(balanced_total_paths), key=lambda x: x.total_cost), analysis

    def set_precursors(self, precursors, connect_direct = False, dummy_exclusive = False):
        """
        Replaces network's previous precursor nodes with provided new precursors.
            Recreates edges that link products back to reactants.

        Args:
            precursors ([ComputedEntry]): list of new precursor entries

        Returns:
            None
        """

        for node in self._g.nodes():
            if node.description == "S" or node.description == "D":
                self._g.remove_node(node)

                if not precursors or dummy_exclusive:
                    precursors_entries = RxnEntries(None, "d")
                else:
                    precursors = set(precursors)
                    precursors_entries = RxnEntries(precursors, "s")

                self._g.add_node(precursors_entries)

                if connect_direct:
                    self._g.add_edge(precursors_entries, RxnEntries(precursors, "r"), weight=0)
                else:
                    for r in self._generate_all_combos(self._filtered_entries, self._max_num_phases):
                        reactants = set(r)
                        if reactants.issubset(precursors):  # link starting node to reactant nodes
                            reactant_entries = RxnEntries(reactants, "r")
                            self._g.add_edge(precursors_entries, reactant_entries, weight=0)
                break

        if not self._complex_loopback:
            self._precursors = precursors
            return

        for p in self._generate_all_combos(self._filtered_entries, self._max_num_phases):
            products = set(p)
            product_entries = RxnEntries(products, "p")

            old_loopbacks = self._generate_all_combos(list(products.union(self._precursors)), self._max_num_phases)
            for combo in old_loopbacks:
                # delete old edge linking back
                self._g.remove_edge(product_entries, RxnEntries(combo, "r"))

            new_loopbacks = self._generate_all_combos(list(products.union(precursors)), self._max_num_phases)
            for combo in new_loopbacks:
                # add new edges linking back
                self._g.add_edge(product_entries, RxnEntries(combo, "r"), weight=0)

        self._precursors = precursors

    def set_target(self, target):
        """
        Replaces network's previous target node with provided new target.

        Args:
            target (ComputedEntry): entry of new target

        Returns:
            None
        """
        g = self._g

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
                new_edges.append([vertex, new_target_v, 0, None, True, False])  # link all products to new target

        g.add_edge_list(new_edges, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]])

    def set_cost_function(self, cost_function, temp=300):
        """
        Replaces network's current cost function with provided new function by recomputing edge weights.

        Args:
            cost_function (str): name of cost function (e.g. "softplus")
            temp (int): temperature argument used in cost function.

        Returns:
            None
        """
        return None

    def _get_rxn_cost(self, rxn, cost_function="softplus", temp=300, struct_fingerprints=None):
        """
        Helper method which determines reaction cost/weight.

        Args:
            (CalculatedReaction): calculated free energy of reaction
            cost_function (str): name of cost function (e.g. "softplus")
            temp (int): temperature parameter for scaling cost function [Kelvin]

        Returns:
            float: cost/weight of individual reaction edge
        """

        total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
        energy = rxn.calculated_reaction_energy / total_num_atoms

        if cost_function == "softplus":
            if struct_fingerprints:
                similarity = np.linalg.norm(-np.mean([struct_fingerprints[e] for e in rxn._reactant_entries])
                                            + np.mean([struct_fingerprints[e] for e in rxn._product_entries]))
                weight = self._softplus([energy, similarity], [0.5, 0.5], t=temp)
            else:
                weight = self._softplus([energy], [1], t=temp)

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

    @staticmethod
    def _yens_ksp(g, num_k, precursors_v, target_v, edge_prop="bool", weight_prop="weight"):
        """
        Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by Antonin Lenfant.

        Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths in a Network",
        Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.
        """

        def path_cost(vertices):
            cost = 0
            for j in range(len(vertices)-1):
                cost += g.ep[weight_prop][g.edge(vertices[j], vertices[j+1])]
            return cost

        path = gt.shortest_path(g, precursors_v, target_v, weights=g.ep[weight_prop])[0]

        if not path:
            return []
        a = [path]
        a_costs = [path_cost(path)]

        b = queue.PriorityQueue()  # automatically sorts by path cost (priority)

        for k in range(1, num_k):
            try:
                prev_path = a[k - 1]
            except IndexError:
                print(f"Identified only k={k} paths before exiting. \n")
                break

            for i in range(len(prev_path) - 1):
                spur_v = prev_path[i]
                root_path = prev_path[:i]

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
    def _filter_entries(all_entries, e_above_hull, include_polymorphs):
        """
        Helper method for filtering entries by specified energy above hull

        Args:
            e_above_hull (float): cutoff for energy above hull (** eV **)
            include_polymorphs (bool): whether to include higher energy polymorphs of existing structures

        Returns:
            list: all entries less than or equal to specified energy above hull
        """
        pd_dict = expand_pd(all_entries)
        energies_above_hull = dict()

        for entry in all_entries:
            for chemsys, phase_diag in pd_dict.items():
                if set(entry.composition.chemical_system.split("-")).issubset(chemsys.split("-")):
                    energies_above_hull[entry] = phase_diag.get_e_above_hull(entry)
                    break

        if e_above_hull == 0:
            filtered_entries = [e[0] for e in energies_above_hull.items() if e[1] == 0]
        else:
            filtered_entries = [e[0] for e in energies_above_hull.items() if e[1] <= e_above_hull]

            if not include_polymorphs:
                filtered_entries_no_polymorphs = []
                all_comp = {entry.composition.reduced_composition for entry in filtered_entries}
                for comp in all_comp:
                    polymorphs = [entry for entry in filtered_entries if entry.composition.reduced_composition == comp]
                    min_entry = min(polymorphs, key=lambda x: x.energy_per_atom)
                    filtered_entries_no_polymorphs.append(min_entry)

                filtered_entries = filtered_entries_no_polymorphs

        return pd_dict, filtered_entries

    @staticmethod
    def _generate_all_combos(entries, max_num_combos):
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
    def _softplus(params, weights, t=273):
        """
        Args:
            params: list of cost function parameters (e.g. energy)
            weights: list of weights corresponds to parameters of the cost function
            t: temperature (K)

        Returns:
            float: cost (in a.u.)
        """
        weighted_params = np.dot(np.array(params), np.array(weights))
        return np.log(1 + (273 / t) * np.exp(weighted_params))

    @staticmethod
    def _arrhenius(energy, t):
        """
        Simple Arrenhius relation involving energy and temperature

        Args:
            energy:
            t:

        Returns: cost
        """
        kb = physical_constants["Boltzmann constant in eV/K"][0]
        return np.exp(energy / (kb * t))

    @property
    def g(self):
        return self._g
    @property
    def all_entries(self):
        return self._all_entries

    @property
    def filtered_entries(self):
        return self._filtered_entries

    @property
    def precursors(self):
        return self._precursors

    @property
    def all_targets(self):
        return self._all_targets

    def __repr__(self):
        return f"ReactionNetwork object, of {'-'.join(sorted([str(e) for e in self._pd.elements]))}, " \
               f"with {str(self._g)}"
