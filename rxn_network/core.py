import logging
from itertools import combinations, chain, groupby, compress
from tqdm import tqdm
from profilehooks import profile

import numpy as np
from numba import njit, prange
from numba.typed import List

import graph_tool.all as gt
import queue

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError

from rxn_network.helpers import *
from rxn_network.analysis import *


__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.1"
__email__ = "mcdermott@lbl.gov"
__date__ = "July 4, 2020"


class ReactionNetwork:
    """This class creates and stores a weighted, directed graph in graph-tool
        that is a dense network of all possible chemical reactions (edges)
        between phase combinations (vertices) in a chemical system. Reaction
        pathway hypotheses are generated using pathfinding methods.
    """

    def __init__(
        self,
        entries,
        n=2,
        temp=300,
        extend_entries=None,
        include_metastable=False,
        include_polymorphs=False,
    ):
        """Initializes ReactionNetwork object with necessary preprocessing
            steps. This does not yet compute the graph. The preprocessing
            steps currently include: generating the phase diagram and/or
            energies above hull for filtering (using estimated Gibbs free
            energies of formation at specified temperature).

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-
                like objects to consider in network. These can be acquired
                from Materials Project (using MPRester) or created manually in
                pymatgen. Entries should have same compatability (e.g.
                MPCompability) for phase diagram generation.
            n (int): maximum number of phases allowed on each side of the
                reaction (default 2). Note that n > 2 leads to significant (
                and often intractable) combinatorial explosion.
            temp (int): Temperature (in Kelvin) used for estimating Gibbs
                free energy of formation, as well as scaling the cost function
                later during network generation. Must select from [300, 400,
                500, ... 2000] K.
            extend entries([ComputedStructureEntry]): list of
                ComputedStructureEntry-like objects which will be included in
                the network even after filtering for thermodynamic stability.
                Helpful if target phase has a significantly high energy above
                the hull.
            include_metastable (float or bool): either a) the specified cutoff
                for energy per atom (eV/atom) above hull, or b) True/False
                if considering only stable vs. all entries. An energy cutoff of
                0.1 eV/atom is a reasonable starting threshold for thermodynamic
                stability. Defaults to False.
            include_polymorphs (bool): Whether or not to consider non-ground
                state polymorphs. Defaults to False. Note this is not useful
                unless structural metrics are considered in the cost function
                (to be added!)
        """
        self.logger = logging.getLogger("ReactionNetwork")
        self.logger.setLevel("INFO")

        # Chemical system / phase diagram variables
        self._all_entries = entries
        self._max_num_phases = n
        self._temp = temp
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs
        self._elements = {
            elem for entry in self.all_entries for elem in entry.composition.elements
        }
        self._gibbs_entries = GibbsComputedStructureEntry.from_entries(
            self._all_entries, self._temp
        )
        self._pd_dict, self._filtered_entries = self._filter_entries(
            self._gibbs_entries, include_metastable, include_polymorphs
        )
        self._pd = None

        if (
            len(self._elements) <= 10
        ):  # phase diagrams take considerable time to build >10 elems
            self._pd = PhaseDiagram(self._filtered_entries)

        if extend_entries:
            self._filtered_entries.extend(extend_entries)

        self._all_entry_combos = [
            set(combo)
            for combo in self._generate_all_combos(
                self._filtered_entries, self._max_num_phases
            )
        ]

        # Graph variables used during graph creation
        self._precursors = None
        self._precursors_v = None
        self._all_targets = None
        self._target_v = None
        self._current_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._most_negative_rxn = None  # used in piecewise cost function
        self._g = None  # Graph object in graph-tool

        filtered_entries_str = ", ".join(
            [entry.composition.reduced_formula for entry in self._filtered_entries]
        )
        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} "
            f"entries: \n{filtered_entries_str}"
        )

    def generate_rxn_network(
        self,
        precursors=None,
        targets=None,
        cost_function="softplus",
        complex_loopback=True,
    ):
        """Generates and stores the actual reaction network (weighted, directed graph)
            using graph-tool. In practice, the main iterative loop will start taking
            a significant amount of time (i.e. over 60 secs) when using  >50 phases.
            As of now, temperature must be selected from [300, 400, 500, ... 2000 K].

        Args:
            precursors ([ComputedEntry]): entries for all phases which serve as the
                main reactants; if None,a "dummy" node is used to represent any
                possible set of precursors.
            targets ([ComputedEntry]): entries for all phases which are the final
                products; if None, a "dummy" node is used to represent any possible
                set of targets.
            cost_function (str): name of cost function to use for entire network
                (e.g. "softplus").
            complex_loopback (bool): if True, adds zero-weight edges which "loop back"
                to allow for multi-step or autocatalytic-like reactions, i.e. original
                precursors can reappear many times and in different steps.
        """
        self._precursors = set(precursors) if precursors else None
        self._all_targets = set(targets) if targets else None
        self._current_target = (
            {targets[0]} if targets else None
        )  # take first entry to be first designated target
        self._cost_function = cost_function
        self._complex_loopback = complex_loopback

        if not self._precursors:
            precursors_entries = RxnEntries(None, "d")  # use dummy precursors node
            if self._complex_loopback:
                raise ValueError(
                    "Complex loopback can't be enabled when using a dummy precursors "
                    "node!"
                )
        elif False in [
            isinstance(precursor, ComputedEntry) for precursor in self._precursors
        ] or False in [
            isinstance(target, ComputedEntry) for target in self._all_targets
        ]:
            raise TypeError(
                "Precursors and targets must be ComputedEntry-like objects."
            )
        else:
            precursors_entries = RxnEntries(precursors, "s")

        g = gt.Graph()  # initialization of graph obj

        # Create all property maps
        g.vp["entries"] = g.new_vertex_property("object")
        g.vp["type"] = g.new_vertex_property(
            "int"
        )  # 0: precursors, 1: reactants, 2: products, 3: target
        g.vp["bool"] = g.new_vertex_property("bool")
        g.vp["path"] = g.new_vertex_property("bool")  # whether node is part of path
        g.vp["chemsys"] = g.new_vertex_property("string")

        g.ep["weight"] = g.new_edge_property("double")
        g.ep["rxn"] = g.new_edge_property("object")
        g.ep["bool"] = g.new_edge_property("bool")
        g.ep["path"] = g.new_edge_property("bool")  # whether edge is part of path

        precursors_v = g.add_vertex()
        self._update_vertex_properties(
            g,
            precursors_v,
            {
                "entries": precursors_entries,
                "type": 0,
                "bool": True,
                "path": True,
                "chemsys": precursors_entries.chemsys,
            },
        )
        self._precursors_v = precursors_v

        entries_dict = {}
        idx = 0
        for num, entries in enumerate(self._all_entry_combos):
            idx = 2 * num + 1
            reactants = RxnEntries(entries, "R")
            products = RxnEntries(entries, "P")
            chemsys = reactants.chemsys
            if chemsys not in entries_dict:
                entries_dict[chemsys] = dict({"R": {}, "P": {}})

            entries_dict[chemsys]["R"][reactants] = idx
            self._update_vertex_properties(
                g,
                idx,
                {
                    "entries": reactants,
                    "type": 1,
                    "bool": True,
                    "path": False,
                    "chemsys": chemsys,
                },
            )
            entries_dict[chemsys]["P"][products] = idx + 1
            self._update_vertex_properties(
                g,
                idx + 1,
                {
                    "entries": products,
                    "type": 2,
                    "bool": True,
                    "path": False,
                    "chemsys": chemsys,
                },
            )
        g.add_vertex(idx + 1)  # add all precursors, reactant, and product vertices
        target_v = g.add_vertex()  # add target vertex
        target_entries = RxnEntries(self._current_target, "t")
        self._update_vertex_properties(
            g,
            target_v,
            {
                "entries": target_entries,
                "type": 3,
                "bool": True,
                "path": True,
                "chemsys": target_entries.chemsys,
            },
        )
        self._target_v = target_v

        self.logger.info("Generating reactions by chemical subsystem...")

        edge_list = []
        for chemsys, vertices in tqdm(entries_dict.items()):
            for entry, v in vertices["R"].items():
                phases = entry.entries

                v = g.vertex(v)

                if precursors_entries.description == "D" or phases.issubset(
                    self._precursors
                ):
                    edge_list.append([precursors_v, v, 0, None, True, False])

            for entry, v in vertices["P"].items():
                phases = entry.entries
                if self._current_target.issubset(phases):
                    edge_list.append([v, target_v, 0, None, True, False])

                if complex_loopback:
                    combos = self._generate_all_combos(
                        phases.union(self._precursors), self._max_num_phases
                    )
                else:
                    combos = self._generate_all_combos(phases, self._max_num_phases)

                for c in combos:
                    combo_phases = set(c)
                    if combo_phases.issubset(self._precursors):
                        continue
                    combo_entry = RxnEntries(combo_phases, "R")
                    loopback_v = g.vertex(
                        entries_dict[combo_entry.chemsys]["R"][combo_entry]
                    )
                    edge_list.append([v, loopback_v, 0, None, True, False])

            for entry, v in vertices["R"].items():
                phases = entry.entries

                for other_entry, other_v in vertices["P"].items():
                    other_phases = other_entry.entries

                    if other_phases == phases:
                        continue  # do not consider identity-like reactions

                    try:
                        rxn = ComputedReaction(list(phases), list(other_phases))
                    except ReactionError:
                        continue

                    if rxn._lowest_num_errors != 0:
                        continue  # remove reaction if comps. change sides / disappear

                    weight = self._get_rxn_cost(rxn)
                    edge_list.append(
                        [g.vertex(v), g.vertex(other_v), weight, rxn, True, False]
                    )

        g.add_edge_list(
            edge_list, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]]
        )

        self.logger.info(
            f"Created graph with {g.num_vertices()} nodes and {g.num_edges()} edges."
        )
        self._g = g

    def find_k_shortest_paths(self, k, verbose=True):
        """
        Finds k shortest paths to current target using Yen's Algorithm.

        Args:
            k (int): desired number of shortest pathways (ranked by cost)
            verbose (bool): whether to print all identified pathways to the console.

        Returns:
            [RxnPathway]: list of RxnPathway objects containing reactions traversed on
                each path.
        """
        g = self._g
        paths = []

        for num, path in enumerate(
            self._yens_ksp(g, k, self._precursors_v, self._target_v)
        ):
            rxns = []
            weights = []

            for step, v in enumerate(path):
                g.vp["path"][v] = True

                if (
                    g.vp["type"][v] == 2
                ):  # add rxn step if current node in path is a product
                    e = g.edge(path[step - 1], v)
                    g.ep["path"][e] = True  # mark this edge as occurring on a path
                    rxns.append(g.ep["rxn"][e])
                    weights.append(g.ep["weight"][e])

            rxn_pathway = RxnPathway(rxns, weights)
            paths.append(rxn_pathway)

        if verbose:
            for path in paths:
                print(path, "\n")

        return paths

    def find_all_rxn_pathways(
        self, k=15, targets=None, max_num_combos=4, consider_crossover_rxns=True,
    ):
        """
        Builds the k shortest paths to provided targets and then seeks to combine
            them to achieve a "net reaction" with balanced stoichiometry. In other
            words, the full conversion of all intermediates to final products.
            Warning: this method can take a significant amount of time depending on the
            size of the network and the max_num_combos parameter. General
            recommendations are k = 15 and max_num_combos = 4, although a higher
            max_num_combos may be required to capture the full pathway.

        Args:
            k (int): Number of shortest paths to calculate to each target (i.e. if
                there are 3 targets and k=15, then 3x15 = 45 paths will be generated
                and the reactions from these will be combined.
            targets ([ComputedEntries]): list of all target ComputedEntry objects;
                defaults to targets provided when network was created.
            max_num_combos (int): upper limit on how many reactions to consider at a
                time (default 4).
            consider_crossover_rxns (bool): Whether to consider "crossover" reactions
                between intermediates in other pathways. This can be crucial for
                generating realistic predictions and it is highly recommended;
                generally the added computational cost is extremely low.

        Returns:
            ([CombinedPathway], PathwayAnalysis): Tuple containing list of
                CombinedPathway objects (sorted by total cost) and a PathwayAnalysis
                object with helpful analysis methods for hypothesized pathways.
        """
        paths_to_all_targets = dict()

        if not targets:
            targets = self._all_targets
        else:
            targets = set(targets)

        net_rxn = ComputedReaction(list(self._precursors), list(targets))
        net_rxn_all_comp = set(net_rxn.all_comp)
        print(f"NET RXN: {net_rxn} \n")

        for target in targets:
            print(f"PATHS to {target.composition.reduced_formula} \n")
            self.set_target(target)
            paths = self.find_k_shortest_paths(k)
            paths = {
                rxn: cost
                for path in paths
                for (rxn, cost) in zip(path.rxns, path.costs)
            }
            paths_to_all_targets.update(paths)

        if consider_crossover_rxns:

            def find_crossover_rxns():
                all_crossover_rxns = set()
                for reactants_combo in self._generate_all_combos(
                    intermediates, self._max_num_phases
                ):
                    for products_combo in self._generate_all_combos(
                        targets, self._max_num_phases
                    ):
                        try:
                            rxn = ComputedReaction(
                                list(reactants_combo), list(products_combo)
                            )
                        except ReactionError:
                            continue
                        if rxn._lowest_num_errors > 0:
                            continue
                        path = RxnPathway([rxn], [self._get_rxn_cost(rxn)])
                        all_crossover_rxns.add(path)
                return all_crossover_rxns

            precursors_and_targets = targets | self._precursors

            intermediates = {
                entry for rxn in paths_to_all_targets for entry in rxn.all_entries
            } - precursors_and_targets
            crossover_rxns = find_crossover_rxns()

            if crossover_rxns:
                print("Crossover Reactions \n")
                print(crossover_rxns, "\n")
                crossover_rxns = {
                    rxn: cost
                    for path in crossover_rxns
                    for (rxn, cost) in zip(path.rxns, path.costs)
                }
                paths_to_all_targets.update(crossover_rxns)

        paths_to_try = list(paths_to_all_targets.keys())

        trial_combos = list(self._generate_all_combos(paths_to_try, max_num_combos))
        unbalanced_paths = []

        for i, combo in enumerate(tqdm(trial_combos)):
            if net_rxn_all_comp.issubset(
                [comp for rxn in combo for comp in rxn.all_comp]
            ):
                unbalanced_paths.append(
                    BalancedPathway(
                        {p: paths_to_all_targets[p] for p in combo},
                        net_rxn,
                        balance=False,
                    )
                )

        comp_matrices = List([p.comp_matrix for p in unbalanced_paths])
        net_coeffs = List([p.net_coeffs for p in unbalanced_paths])

        is_balanced, multiplicities = self._balance_all_paths(comp_matrices, net_coeffs)

        balanced_total_paths = list(compress(unbalanced_paths, is_balanced))
        balanced_multiplicities = list(compress(multiplicities, is_balanced))

        for p, m in zip(balanced_total_paths, balanced_multiplicities):
            p.set_multiplicities(m)
            p.calculate_costs()

        analysis = PathwayAnalysis(self, balanced_total_paths)
        return sorted(list(balanced_total_paths), key=lambda x: x.total_cost), analysis

    def set_precursors(self, precursors=None, complex_loopback=True):
        """Replaces network's previous precursor node with provided new precursors.
            Finds new edges that link products back to reactants as dependent on the
            complex_loopback parameter.

        Args:
            precursors ([ComputedEntry]): list of new precursor entries
            complex_loopback (bool): if True, adds zero-weight edges which "loop back"
                to allow for multi-step or autocatalytic-like reactions, i.e. original
                precursors can reappear many times and in different steps.

        Returns:
            None
        """
        g = self._g
        self._precursors = set(precursors) if precursors else None

        if not self._precursors:
            precursors_entries = RxnEntries(None, "d")  # use dummy precursors node
            if self._complex_loopback:
                raise ValueError(
                    "Complex loopback can't be enabled when using a dummy precursors "
                    "node!"
                )
        elif False in [
            isinstance(precursor, ComputedEntry) for precursor in self._precursors
        ]:
            raise TypeError("Precursors must be ComputedEntry-like objects.")
        else:
            precursors_entries = RxnEntries(precursors, "s")

        g.remove_vertex(self._precursors_v)
        new_precursors_v = g.add_vertex()

        self._update_vertex_properties(
            g,
            new_precursors_v,
            {
                "entries": precursors_entries,
                "type": 0,
                "bool": True,
                "path": True,
                "chemsys": precursors_entries.chemsys,
            },
        )
        self._precursors_v = new_precursors_v

        new_edges = []
        remove_edges = []

        for v in gt.find_vertex(g, g.vp["type"], 1):  # iterate over all reactants
            phases = g.vp["entries"][v].entries

            remove_edges.extend(list(v.in_edges()))

            if precursors_entries.description == "D" or phases.issubset(
                self._precursors
            ):
                new_edges.append([new_precursors_v, v, 0, None, True, False])

        for v in gt.find_vertex(g, g.vp["type"], 2):  # iterate over all products
            phases = g.vp["entries"][v].entries

            if complex_loopback:
                combos = self._generate_all_combos(
                    phases.union(self._precursors), self._max_num_phases
                )
            else:
                combos = self._generate_all_combos(phases, self._max_num_phases)

            for c in combos:
                combo_phases = set(c)
                if combo_phases.issubset(self._precursors):
                    continue
                combo_entry = RxnEntries(combo_phases, "R")
                loopback_v = gt.find_vertex(g, g.vp["entries"], combo_entry)[0]
                new_edges.append([v, loopback_v, 0, None, True, False])

        for e in remove_edges:
            g.remove_edge(e)

        g.add_edge_list(
            new_edges, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]]
        )

    def set_target(self, target):
        """Replaces network's current target phase with new target phase.

        Args:
            target (ComputedEntry): ComputedEntry-like object for new target phase.

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
        self._update_vertex_properties(
            g,
            new_target_v,
            {
                "entries": new_target_entry,
                "type": 3,
                "bool": True,
                "path": True,
                "chemsys": new_target_entry.chemsys,
            },
        )
        self._target_v = new_target_v

        new_edges = []

        for v in gt.find_vertex(g, g.vp["type"], 2):  # search for all products
            if self._current_target.issubset(g.vp["entries"][v].entries):
                new_edges.append(
                    [v, new_target_v, 0, None, True, False]
                )  # link all products to new target

        g.add_edge_list(
            new_edges, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]]
        )

    def set_cost_function(self, cost_function):
        """Replaces network's current cost function with new function by recomputing
            edge weights.

        Args:
            cost_function (str): name of cost function. Current options are
                ["softplus", "relu", "piecewise"].

        Returns:
            None
        """
        g = self._g
        self._cost_function = cost_function

        for e in gt.find_edge_range(g, g.ep["weight"], (1e-8, 1e8)):
            g.ep["weight"][e] = self._get_rxn_cost(g.ep["rxn"][e])

    def set_temp(self, temp):
        """Sets new temperature parameter of network by recomputing
            GibbsComputedStructureEntry objects and edge weights. Does not re-filter
            for thermodynamic stability, as this would essentially
            require full initialization of a newobject.

        Args:
            temp (int): temperature in Kelvin; must be selected from
                [300, 400, 500, ... 2000] K.

        Returns:
            None
        """
        g = self._g
        self._temp = temp
        old_entries = self._filtered_entries
        self._filtered_entries = GibbsComputedStructureEntry.from_entries(
            self._filtered_entries, self._temp
        )
        mapping = dict()
        for old_e in old_entries:
            for new_e in self._filtered_entries:
                if old_e.structure == new_e.structure:
                    mapping[old_e] = new_e
                    break

        for v in g.vertices():
            current_entries = g.vp["entries"][v]
            new_phases = [mapping[e] for e in current_entries.entries]
            new_entries = RxnEntries(new_phases, current_entries.description)
            g.vp["entries"][v] = new_entries

        for e in gt.find_edge_range(g, g.ep["weight"], (1e-8, 1e8)):
            rxn = ComputedReaction(
                list(g.vp["entries"][e.source()].entries),
                list(g.vp["entries"][e.target()].entries),
            )
            g.ep["rxn"][e] = rxn
            g.ep["weight"][e] = self._get_rxn_cost(rxn)

        self._precursors = {mapping[p] for p in self._precursors}
        self._all_targets = {mapping[t] for t in self._all_targets}
        self._current_target = {mapping[ct] for ct in self._current_target}

    def _get_rxn_cost(self, rxn):
        """Helper method which determines reaction cost/weight.

        Args:
            rxn (CalculatedReaction): the pymatgen CalculatedReaction object.

        Returns:
            float: cost/weight of individual reaction edge
        """
        total_num_atoms = sum([rxn.get_el_amount(elem) for elem in rxn.elements])
        energy = rxn.calculated_reaction_energy / total_num_atoms
        cost_function = self._cost_function

        if cost_function == "softplus":
            params = [energy]
            weights = [1]
            weight = self._softplus(params, weights, t=self._temp)
        elif cost_function == "piecewise":
            weight = energy

            if weight < self._most_negative_rxn:
                self._most_negative_rxn = weight
            if weight >= 0:
                weight = 2 * weight + 1
        elif cost_function == "relu":
            weight = energy
            if weight < 0:
                weight = 0
        else:
            weight = 0

        return weight

    @staticmethod
    def _update_vertex_properties(g, v, prop_dict):
        """Helper method for updating several vertex properties at once in a graph-tool
            graph.

        Args:
            g (gt.Graph): a graph-tool Graph object.
            v (gt.Vertex or int): a graph-tool Vertex object (or its index) for a vertex
                in the provided graph.
            prop_dict (dict): a dictionary of the form {"prop": val}, where prop is the
                name of a VertexPropertyMap of the graph and val is the new updated
                value for that vertex's property.

        Returns:
            None
        """
        for prop, val in prop_dict.items():
            g.vp[prop][v] = val
        return None

    @staticmethod
    def _yens_ksp(
        g, num_k, precursors_v, target_v, edge_prop="bool", weight_prop="weight"
    ):
        """Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by
            Antonin Lenfant. Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths
            in a Network", Management Science, Vol. 17, No. 11, Theory Series (Jul.,
            1971), pp. 712-716.

        Args:
            g (gt.Graph): the graph-tool graph object.
            num_k (int): number of k shortest paths that should be found.
            precursors_v (gt.Vertex): graph-tool vertex object containing precursors.
            target_v (gt.Vertex): graph-tool vertex object containing target.
            edge_prop (str): name of edge property map which allows for filtering edges.
                Defaults to the word "bool".
            weight_prop (str): name of edge property map that stores edge weights/costs.
                Defaults to the word "weight".

        Returns:
            List of lists of graph vertices corresponding to each shortest path
                (sorted in increasing order by cost).
        """
        def path_cost(vertices):
            """Calculates path cost given a list of vertices.
            """
            cost = 0
            for j in range(len(vertices) - 1):
                cost += g.ep[weight_prop][g.edge(vertices[j], vertices[j + 1])]
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
                spur_path = gt.shortest_path(
                    gv, spur_v, target_v, weights=g.ep[weight_prop]
                )[0]

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
            all_entries ([ComputedEntry]): List of ComputedEntry-like objects to be
                filtered
            e_above_hull (float): Thermodynamic stability threshold (energy above hull)
                [eV/atom]
            include_polymorphs (bool): whether to include higher energy polymorphs of
                existing structures

        Returns:
            [ComputedEntry]: list of all entries with energies above hull equal to or
                less than the specified e_above_hull.
        """
        pd_dict = expand_pd(all_entries)
        energies_above_hull = dict()

        for entry in all_entries:
            for chemsys, phase_diag in pd_dict.items():
                if set(entry.composition.chemical_system.split("-")).issubset(
                    chemsys.split("-")
                ):
                    energies_above_hull[entry] = phase_diag.get_e_above_hull(entry)
                    break

        if e_above_hull == 0:
            filtered_entries = [e[0] for e in energies_above_hull.items() if e[1] == 0]
        else:
            filtered_entries = [
                e[0] for e in energies_above_hull.items() if e[1] <= e_above_hull
            ]

            if not include_polymorphs:
                filtered_entries_no_polymorphs = []
                all_comp = {
                    entry.composition.reduced_composition for entry in filtered_entries
                }
                for comp in all_comp:
                    polymorphs = [
                        entry
                        for entry in filtered_entries
                        if entry.composition.reduced_composition == comp
                    ]
                    min_entry = min(polymorphs, key=lambda x: x.energy_per_atom)
                    filtered_entries_no_polymorphs.append(min_entry)

                filtered_entries = filtered_entries_no_polymorphs

        return pd_dict, filtered_entries

    @staticmethod
    @njit(parallel=True)
    def _balance_all_paths(comp_matrices, net_coeffs, tol=1e-6):
        """
        Fast solution for reaction multiplicities via mass balance stochiometric
            constraints. Parallelized using Numba.

        Args:
            comp_matrices ([np.array]): list of numpy arrays containing stoichiometric
                coefficients of all compositions in all reactions, for each trial
                combination.
            net_coeffs ([np.array]): list of numpy arrays containing stoichiometric
                coefficients of net reaction.
            tol (float): numerical tolerance for determining if a multiplicity is zero
                (reaction was removed).

        Returns:
            ([bool],[np.array]): Tuple containing bool identifying which trial
                BalancedPathway objects were successfully balanced, and a list of all
                multiplicities arrays.
        """
        n = len(comp_matrices)
        comp_pseudo_inverse = List([0.0 * s for s in comp_matrices])
        multiplicities = List([0.0 * c for c in net_coeffs])
        is_balanced = List([False] * n)

        for i in prange(n):
            comp_pseudo_inverse[i] = np.linalg.pinv(comp_matrices[i]).T
            multiplicities[i] = comp_pseudo_inverse[i] @ net_coeffs[i]
            solved_coeffs = comp_matrices[i].T @ multiplicities[i]

            if (multiplicities[i] < tol).any():
                is_balanced[i] = False
            elif (
                np.abs(solved_coeffs - net_coeffs[i])
                <= (1e-08 + 1e-05 * np.abs(net_coeffs[i]))
            ).all():
                is_balanced[i] = True

        return is_balanced, multiplicities

    @staticmethod
    def _generate_all_combos(entries, max_num_combos):
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

    @staticmethod
    def _softplus(params, weights, t=273):
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

    @property
    def g(self):
        return self._g

    @property
    def pd(self):
        return self._pd

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

    @property
    def temp(self):
        return self._temp

    def __repr__(self):
        return (
            f"ReactionNetwork for chemical system: "
            f"{'-'.join(sorted([str(e) for e in self._pd.elements]))}, "
            f"with Graph: {str(self._g)}"
        )
