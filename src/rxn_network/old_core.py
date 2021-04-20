import inspect
import logging
import queue
from functools import partial
from itertools import chain, combinations, compress, groupby, product
from pprint import pprint
from time import time

import graph_tool.all as gt
import numpy as np
from dask import bag, compute, delayed
from numba import njit, prange
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.entry_tools import EntrySet
from scipy.special import comb
from tqdm import tqdm

from rxn_network.old_analysis import PathwayAnalysis
from rxn_network.entries import GibbsComputedStructureEntry, PDEntry, RxnEntries
from rxn_network.old_helpers import (
    BalancedPathway,
    RxnPathway,
    expand_pd,
    find_interdependent_rxns,
    find_rxn_edges,
    powerset,
    get_rxn_cost,
    grouper,
    react_interface,
)
from rxn_network.reaction import (
    BalancedReaction,
    ComputedReaction,
    Reaction,
    ReactionError,
)

__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.2"
__email__ = "mcdermott@lbl.gov"
__date__ = "December 20, 2020"


class ReactionNetwork:
    """
    This class creates and stores a weighted, directed graph in graph-tool
    that is a dense network of all possible chemical reactions (edges)
    between phase combinations (vertices) in a chemical system. Reaction
    pathway hypotheses are generated using pathfinding methods.
    """

    def __init__(
        self,
        entries,
        n=2,
        temp=300,
        interpolate_comps=None,
        extend_entries=None,
        include_metastable=False,
        include_polymorphs=False,
        include_chempot_restriction=False,
        filter_rxn_energies=0.5,
    ):
        """Initializes ReactionNetwork object with necessary preprocessing
        steps. This does not yet compute the graph.

        Args:
            entries ([ComputedStructureEntry]): list of ComputedStructureEntry-
                like objects to consider in network. These can be acquired
                from Materials Project (using MPRester) or created manually in
                pymatgen. Entries should have same compatability (e.g.
                MPCompability) for phase diagram generation.
            n (int): maximum number of phases allowed on each side of the
                reaction (default 2). Note that dim > 2 leads to significant (
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
            include_chempot_restriction (bool): Whether or not to consider reactions to
                those where the minimum possible μ of each element in the products
                can not be higher than the maximum possible μ of each element in the
                reactants. Seems to lead to more logical pathway predictions.
                Defaults to False.
        """
        self.logger = logging.getLogger("ReactionNetwork")
        self.logger.setLevel("INFO")

        # Chemical system / phase diagram variables
        self._all_entries = entries
        self._max_num_phases = n
        self._temp = temp
        self._e_above_hull = include_metastable
        self._include_polymorphs = include_polymorphs
        self._include_chempot_restriction = include_chempot_restriction
        self._elements = {
            elem for entry in self.all_entries for elem in entry.composition.elements
        }
        self._pd_dict, self._filtered_entries = self._filter_entries(
            entries, include_metastable, temp, include_polymorphs
        )
        self._entry_mu_ranges = {}
        self._pd = None
        self._rxn_e_filter = filter_rxn_energies

        if (
            len(self._elements) <= 10
        ):  # phase diagrams take considerable time to build with 10+ elems
            self._pd = PhaseDiagram(self._filtered_entries)
        elif len(self._elements) > 10 and include_chempot_restriction:
            raise ValueError(
                "Cannot include mu restriction for networks with greater than 10 elements!"
            )

        if interpolate_comps:
            interpolated_entries = []
            for comp in interpolate_comps:
                energy = self._pd.get_hull_energy(Composition(comp))
                interpolated_entries.append(
                    PDEntry(comp, energy, attribute={"interpolated": True})
                )
            print("Interpolated entries:", "\n")
            print(interpolated_entries)
            self._filtered_entries.extend(interpolated_entries)

        if extend_entries:
            self._filtered_entries.extend(extend_entries)

        for idx, e in enumerate(self._filtered_entries):
            e.entry_idx = idx

        self.num_entries = len(self._filtered_entries)

        if include_chempot_restriction:
            for e in self._filtered_entries:
                elems = e.composition.elements
                chempot_ranges = {}
                all_chempots = {e: [] for e in elems}
                for simplex, chempots in self._pd.get_all_chempots(
                    e.composition
                ).items():
                    for elem in elems:
                        all_chempots[elem].append(chempots[elem])
                for elem in elems:
                    chempot_ranges[elem] = (
                        min(all_chempots[elem]),
                        max(all_chempots[elem]),
                    )

                self._entry_mu_ranges[e] = chempot_ranges

        self._all_entry_combos = [
            set(combo)
            for combo in powerset(self._filtered_entries, self._max_num_phases)
        ]

        self.entry_set = EntrySet(self._filtered_entries)
        self.entry_indices = {e: idx for idx, e in enumerate(self._filtered_entries)}

        # Graph variables used during graph creation
        self._precursors = None
        self._all_targets = None
        self._current_target = None
        self._cost_function = None
        self._complex_loopback = None
        self._precursors_entries = None
        self._most_negative_rxn = None  # used in piecewise cost function
        self._g = None  # Graph object in graph-tool

        filtered_entries_str = ", ".join(
            [entry.composition.reduced_formula for entry in self._filtered_entries]
        )
        self.logger.info(
            f"Initializing network with {len(self._filtered_entries)} "
            f"entries: \n{filtered_entries_str}"
        )

    @staticmethod
    def get_target_edges(
        vertex,
        current_target,
        target_v,
        precursors,
        max_num_phases,
        entries_dict,
        complex_loopback,
    ):
        entry = vertex[0]
        v = vertex[1]

        edge_list = []
        phases = entry.entries
        if current_target.issubset(phases):
            edge_list.append([v, target_v, 0, None, True, False])

        if complex_loopback:
            combos = powerset(phases.union(precursors), max_num_phases)
        else:
            combos = powerset(phases, max_num_phases)

        if complex_loopback:
            for c in combos:
                combo_phases = set(c)
                if combo_phases.issubset(precursors):
                    continue
                combo_entry = RxnEntries(combo_phases, "R")
                loopback_v = entries_dict[combo_entry.chemsys]["R"][combo_entry]

                edge_list.append([v, loopback_v, 0, None, True, False])

        return edge_list

    def generate_rxn_network(
        self,
        precursors=None,
        targets=None,
        cost_function="softplus",
        complex_loopback=True,
    ):
        """
        Generates and stores the reaction network (weighted, directed graph)
        using graph-tool.

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
        else:
            precursors_entries = RxnEntries(precursors, "s")

        self._precursors_entries = precursors_entries

        g = gt.Graph()  # initialization of graph obj in graph-tool

        g.vp["entries"] = g.new_vertex_property("object")
        g.vp["type"] = g.new_vertex_property(
            "int"
        )  # Type 0: precursors, 1: reactants, 2: products, 3: target
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

        target_chemsys = set(
            list(self._current_target)[0].composition.chemical_system.split("-")
        )
        entries_dict = {}
        idx = 1
        for entries in self._all_entry_combos:
            reactants = RxnEntries(entries, "R")
            products = RxnEntries(entries, "P")
            chemsys = reactants.chemsys
            if (
                self._precursors_entries.description == "D"
                and not target_chemsys.issubset(chemsys.split("-"))
            ):
                continue
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

            if (
                self._precursors_entries.description == "D"
                and not self._all_targets.issubset(entries)
            ):
                idx = idx + 1
                continue

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
            idx = idx + 2

        g.add_vertex(idx)  # add ALL precursors, reactant, and product vertices
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

        self.logger.info("Generating reactions by chemical subsystem...")

        all_edges = []

        all_rxn_combos = []
        start_time = time()

        for chemsys, vertices in entries_dict.items():
            precursor_edges = [
                [precursors_v, v, 0, None, True, False]
                for entry, v in vertices["R"].items()
                if self._precursors_entries.description == "D"
                or entry.entries.issubset(self._precursors)
            ]

            target_edges = [
                edge
                for edge_list in map(
                    partial(
                        self.get_target_edges,
                        current_target=self._current_target,
                        target_v=int(target_v),
                        precursors=self._precursors,
                        max_num_phases=self._max_num_phases,
                        entries_dict=entries_dict,
                        complex_loopback=complex_loopback,
                    ),
                    vertices["P"].items(),
                )
                for edge in edge_list
                if edge
            ]

            rxn_combos = product(vertices["R"].items(), vertices["P"].items())
            all_rxn_combos.append(rxn_combos)

            all_edges.extend(precursor_edges)
            all_edges.extend(target_edges)

        db = bag.from_sequence(
            chain.from_iterable(all_rxn_combos), partition_size=100000
        )
        reaction_edges = db.map_partitions(
            find_rxn_edges,
            cost_function=cost_function,
            rxn_e_filter=self._rxn_e_filter,
            temp=self.temp,
            num_entries=self.num_entries,
        ).compute()

        all_edges.extend(reaction_edges)

        g.add_edge_list(
            all_edges, eprops=[g.ep["weight"], g.ep["rxn"], g.ep["bool"], g.ep["path"]]
        )

        end_time = time()
        self.logger.info(
            f"Graph creation took {round(end_time - start_time,1)} seconds."
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

        precursors_v = gt.find_vertex(g, g.vp["type"], 0)[0]
        target_v = gt.find_vertex(g, g.vp["type"], 3)[0]

        for num, path in enumerate(self._yens_ksp(g, k, precursors_v, target_v)):
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

    def find_intermediate_rxns(self, intermediates, targets, chempots=None):
        all_rxns = set()
        combos = list(powerset(intermediates, 2))
        for entries in tqdm(combos):
            n = len(entries)
            r1 = entries[0].composition.reduced_composition
            chemsys = {
                str(el) for entry in entries for el in entry.composition.elements
            }
            elem = None
            if chempots:
                elem = str(list(chempots.keys())[0])
                chemsys.update(elem)
                if chemsys == {elem}:
                    continue

            if n == 1:
                r2 = entries[0].composition.reduced_composition
            elif n == 2:
                r2 = entries[1].composition.reduced_composition
            else:
                raise ValueError("Can't have an interface that is not 1 to 2 entries!")

            if chempots:
                elem_comp = Composition(elem).reduced_composition
                if r1 == elem_comp or r2 == elem_comp:
                    continue

            entry_subset = self.entry_set.get_subset_in_chemsys(list(chemsys))
            pd = PhaseDiagram(entry_subset)
            grand_pd = None
            if chempots:
                grand_pd = GrandPotentialPhaseDiagram(entry_subset, chempots)

            rxns = react_interface(r1, r2, pd, self.num_entries, grand_pd)
            rxns_filtered = {r for r in rxns if set(r._product_entries) & targets}
            if rxns_filtered:
                most_favorable_rxn = min(
                    rxns_filtered,
                    key=lambda x: (
                        x.calculated_reaction_energy
                        / sum([x.get_el_amount(elem) for elem in x.elements])
                    ),
                )
                all_rxns.add(most_favorable_rxn)

        return all_rxns

    def find_all_rxn_pathways(
        self,
        k=15,
        precursors=None,
        targets=None,
        max_num_combos=4,
        chempots=None,
        consider_crossover_rxns=10,
        filter_interdependent=True,
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

        if not precursors or set(precursors) == self._precursors:
            precursors = self._precursors
        else:
            self.set_precursors(precursors, self._complex_loopback)

        try:
            net_rxn = ComputedReaction(
                list(precursors), list(targets), num_entries=self.num_entries
            )
        except ReactionError:
            raise ReactionError(
                "Net reaction must be balanceable to find all reaction pathways."
            )

        self.logger.info(f"NET RXN: {net_rxn} \n")

        for target in targets:
            print(f"PATHS to {target.composition.reduced_formula} \n")
            print("--------------------------------------- \n")
            self.set_target(target)
            paths = self.find_k_shortest_paths(k)
            paths = {
                rxn: cost
                for path in paths
                for (rxn, cost) in zip(path.rxns, path.costs)
            }
            paths_to_all_targets.update(paths)

        print("Finding crossover reactions paths...")

        if consider_crossover_rxns:
            intermediates = {
                entry for rxn in paths_to_all_targets for entry in rxn.all_entries
            } - targets
            intermediate_rxns = self.find_intermediate_rxns(
                intermediates, targets, chempots
            )
            paths = {
                rxn: get_rxn_cost(rxn, self._cost_function, self.temp)
                for rxn in intermediate_rxns
            }
            paths_to_all_targets.update(paths)
            paths_to_all_targets.update(
                self.find_crossover_rxns(intermediates, targets)
            )

        paths_to_all_targets.pop(net_rxn, None)

        paths_to_all_targets = {
            k: v
            for k, v in paths_to_all_targets.items()
            if not (
                self._precursors.intersection(k._product_entries)
                or self._all_targets.intersection(k._reactant_entries)
                or len(k._product_entries) > 3
            )
        }

        rxn_list = [r for r in paths_to_all_targets.keys()]
        pprint(rxn_list)
        normalized_rxns = [Reaction.from_string(r.normalized_repr) for r in rxn_list]

        num_rxns = len(rxn_list)

        self.logger.info(f"Considering {num_rxns} reactions...")
        batch_size = 500000
        total_paths = []
        for n in range(1, max_num_combos + 1):
            if n >= 4:
                self.logger.info(f"Generating and filtering size {n} pathways...")
            all_c_mats, all_m_mats = [], []
            for combos in tqdm(
                grouper(combinations(range(num_rxns), n), batch_size),
                total=int(comb(num_rxns, n) / batch_size),
            ):
                comp_matrices = np.stack(
                    [
                        np.vstack([rxn_list[r].vector for r in combo])
                        for combo in combos
                        if combo
                    ]
                )
                c_mats, m_mats = self._balance_path_arrays(
                    comp_matrices, net_rxn.vector
                )
                all_c_mats.extend(c_mats)
                all_m_mats.extend(m_mats)

            for c_mat, m_mat in zip(all_c_mats, all_m_mats):
                rxn_dict = {}
                for rxn_mat in c_mat:
                    reactant_entries = [
                        self._filtered_entries[i]
                        for i in range(len(rxn_mat))
                        if rxn_mat[i] < 0
                    ]
                    product_entries = [
                        self._filtered_entries[i]
                        for i in range(len(rxn_mat))
                        if rxn_mat[i] > 0
                    ]
                    rxn = ComputedReaction(
                        reactant_entries, product_entries, entries=self.num_entries
                    )
                    cost = paths_to_all_targets[
                        rxn_list[
                            normalized_rxns.index(
                                Reaction.from_string(rxn.normalized_repr)
                            )
                        ]
                    ]
                    rxn_dict[rxn] = cost
                p = BalancedPathway(rxn_dict, net_rxn, balance=False)
                p.set_multiplicities(m_mat.flatten())
                total_paths.append(p)

        if filter_interdependent:
            final_paths = set()
            for p in total_paths:
                interdependent, combined_rxn = find_interdependent_rxns(
                    p, [c.composition for c in precursors]
                )
                if interdependent:
                    continue
                final_paths.add(p)
        else:
            final_paths = total_paths

        return sorted(list(final_paths), key=lambda x: x.total_cost)

    def find_crossover_rxns(self, intermediates, targets):
        all_crossover_rxns = dict()
        for reactants_combo in powerset(intermediates, self._max_num_phases):
            for products_combo in powerset(targets, self._max_num_phases):
                try:
                    rxn = ComputedReaction(
                        list(reactants_combo),
                        list(products_combo),
                        num_entries=self.num_entries,
                    )
                except ReactionError:
                    continue
                if rxn._lowest_num_errors > 0:
                    continue
                path = {rxn: get_rxn_cost(rxn, self._cost_function, self._temp)}
                all_crossover_rxns.update(path)
        return all_crossover_rxns

    def set_precursors(self, precursors=None, complex_loopback=True):
        """
        Replaces network's previous precursor node with provided new precursors.
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
            if complex_loopback:
                raise ValueError(
                    "Complex loopback can't be enabled when using a dummy precursors "
                    "node!"
                )
        else:
            precursors_entries = RxnEntries(precursors, "s")

        g.remove_vertex(gt.find_vertex(g, g.vp["type"], 0))
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
                combos = powerset(phases.union(self._precursors), self._max_num_phases)
            else:
                combos = powerset(phases, self._max_num_phases)

            for c in combos:
                combo_phases = set(c)
                if complex_loopback and combo_phases.issubset(self._precursors):
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
        """
        Replaces network's current target phase with new target phase.

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

        g.remove_vertex(gt.find_vertex(g, g.vp["type"], 3))
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
        """
        Replaces network's current cost function with new function by recomputing
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
            g.ep["weight"][e] = get_rxn_cost(
                g.ep["rxn"][e],
            )

    def set_temp(self, temp):
        """
        Sets new temperature parameter of network by recomputing
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
            g.ep["weight"][e] = get_rxn_cost(rxn, self._cost_function, self.temp)

        self._precursors = {mapping[p] for p in self._precursors}
        self._all_targets = {mapping[t] for t in self._all_targets}
        self._current_target = {mapping[ct] for ct in self._current_target}

    def save(self, file_name=None):
        precursors_str = "_".join(
            sorted([e.composition.reduced_formula for e in self._precursors])
        )
        targets_str = "_".join(
            sorted([e.composition.reduced_formula for e in self._all_targets])
        )
        if not file_name:
            file_name = (
                f"network_{'-'.join(sorted([str(e) for e in self._elements]))}"
                f"_{precursors_str}_to_{targets_str}.gt"
            )
        for name, val in inspect.getmembers(self, lambda a: not (inspect.isroutine(a))):
            if name == "_g":
                continue
            if name[0] != "_" or name[1] == "_":
                continue

            print(name)
            data_type = type(val).__name__
            print(data_type)
            if data_type not in ["str", "int", "float", "bool"]:
                data_type = "object"
            if data_type == "str":
                data_type = "string"
            self._g.gp[name] = self._g.new_graph_property(data_type, val)

        self._g.save(file_name)

    @classmethod
    def load(cls, file_name):
        # g = gt.load_graph(file_name)
        pass

    @staticmethod
    def _update_vertex_properties(g, v, prop_dict):
        """
        Helper method for updating several vertex properties at once in a graph-tool
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
        """
        Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by
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
            """Calculates path cost given a list of vertices."""
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
    @njit(parallel=True)
    def _balance_path_arrays(
        comp_matrices,
        net_coeffs,
        tol=1e-6,
    ):
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
        shape = comp_matrices.shape
        net_coeff_filter = np.argwhere(net_coeffs != 0).flatten()
        len_net_coeff_filter = len(net_coeff_filter)
        all_multiplicities = np.zeros((shape[0], shape[1]), np.float64)
        indices = np.full(shape[0], False)

        for i in prange(shape[0]):
            correct = True
            for j in range(len_net_coeff_filter):
                idx = net_coeff_filter[j]
                if not comp_matrices[i][:, idx].any():
                    correct = False
                    break
            if not correct:
                continue

            comp_pinv = np.linalg.pinv(comp_matrices[i]).T
            multiplicities = comp_pinv @ net_coeffs
            solved_coeffs = comp_matrices[i].T @ multiplicities

            if (multiplicities < tol).any():
                continue
            elif not (
                np.abs(solved_coeffs - net_coeffs)
                <= (1e-08 + 1e-05 * np.abs(net_coeffs))
            ).all():
                continue
            all_multiplicities[i] = multiplicities
            indices[i] = True

        filtered_indices = np.argwhere(indices != 0).flatten()
        length = filtered_indices.shape[0]
        filtered_comp_matrices = np.empty((length, shape[1], shape[2]), np.float64)
        filtered_multiplicities = np.empty((length, shape[1]), np.float64)

        for i in range(length):
            idx = filtered_indices[i]
            filtered_comp_matrices[i] = comp_matrices[idx]
            filtered_multiplicities[i] = all_multiplicities[idx]

        return filtered_comp_matrices, filtered_multiplicities

    @staticmethod
    def _filter_entries(all_entries, e_above_hull, temp, include_polymorphs=False):
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
        pd_dict = {
            chemsys: PhaseDiagram(GibbsComputedStructureEntry.from_pd(pd, temp))
            for chemsys, pd in pd_dict.items()
        }

        filtered_entries = set()
        all_comps = dict()
        for chemsys, pd in pd_dict.items():
            for entry in pd.all_entries:
                if (
                    entry in filtered_entries
                    or pd.get_e_above_hull(entry) > e_above_hull
                ):
                    continue
                formula = entry.composition.reduced_formula
                if not include_polymorphs and (formula in all_comps):
                    if all_comps[formula].energy_per_atom < entry.energy_per_atom:
                        continue
                    filtered_entries.remove(all_comps[formula])
                all_comps[formula] = entry
                filtered_entries.add(entry)

        return pd_dict, list(filtered_entries)

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
