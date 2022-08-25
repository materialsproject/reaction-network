"""
Firetasks for running enumeration and network calculations
"""
import os
from copy import deepcopy
from typing import List

import numpy as np
import ray
from fireworks import FiretaskBase, FWAction, explicit_serialize
from monty.serialization import dumpfn, loadfn
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core.composition import Composition
from rxn_network.costs.calculators import (
    ChempotDistanceCalculator,
    PrimarySelectivityCalculator,
    SecondarySelectivityCalculator,
)
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.firetasks.utils import get_all_reactant_strs, load_entry_set, load_json
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger, grouper
from rxn_network.utils.ray import to_iterator

logger = get_logger(__name__)


@explicit_serialize
class RunEnumerators(FiretaskBase):
    """
    Run a list of enumerators on a provided set of computed entries and dump the
    calculated ComputedReaction objects to a file (rxns.json.gz).

    Required params:
        enumerators (List[Enumerator]): Enumerators to run
        entries (EntrySet): Computed entries to be fed into enumerate()
            methods

    Optional params:
        None

    """

    required_params: List[str] = ["enumerators", "entries"]
    optional_params: List[str] = ["task_label", "entries_fn", "open_elem", "chempot"]

    def run_task(self, fw_spec):
        enumerators = self["enumerators"]
        task_label = self.get("task_label", "")
        open_elem = self.get("open_elem")
        chempot = self.get("chempot")
        entries = load_entry_set(self, fw_spec)
        chemsys = "-".join(sorted(list(entries.chemsys)))

        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }

        added_elements = None
        added_chemsys = None

        if targets:
            added_elems = entries.chemsys - {
                str(e) for target in targets for e in Composition(target).elements
            }
            added_chemsys = "-".join(sorted(list(added_elems)))
            added_elements = [Element(e) for e in added_elems]

        metadata = {
            "task_label": task_label,
            "dir_name": os.getcwd(),
            "elements": [Element(e) for e in chemsys.split("-")],
            "chemsys": chemsys,
            "enumerators": [e.as_dict() for e in enumerators],
            "targets": list(sorted(targets)),
            "added_elements": added_elements,
            "added_chemsys": added_chemsys,
            "open_elem": open_elem,
            "chempot": chempot,
        }

        results = []
        for enumerator in enumerators:
            logger.info(f"Running {enumerator.__class__.__name__}")
            rxns = enumerator.enumerate(entries)
            results.extend(rxns)

        logger.info("Saving reactions...")

        results = ReactionSet.from_rxns(
            results,
            filter_duplicates=True,
            entries=entries,
            open_elem=open_elem,
            chempot=chempot,
        )

        dumpfn(results, "rxns.json.gz")
        dumpfn(metadata, "metadata.json.gz")

        return FWAction(
            update_spec={"rxns_fn": "rxns.json.gz", "metadata_fn": "metadata.json.gz"}
        )


@explicit_serialize
class CalculateSelectivitiesFromNetwork(FiretaskBase):
    """
    Calculates the competitiveness score using the CompetitivenessScoreCalculator for a
    set of reactions and stores that within the reaction object data.

    Required params:
        target_formula

    Optional params:
        rxns (ReactionSet): A set of reactions of interest. If not provided, will load
            from a rxns.json.gz file
        k (int): The number of reactions to be considered for the calculation of the
            competitiveness scores. Will take the k lowest-cost reactions.
        open_phases (Iterable[str]): An optional list of open phases to be used in
            calculating open reactions with the BasicOpenEnumerator.
        open_elem (Union[str, Element]): An optional open element to be used for
            calculating open reactions with MinimizeGrandPotentialEnumerator.
        chempot (float): The chemical potential of the open element.
        use_basic (bool): Whether or not to use the basic enumerator(s)
            (BasicEnumerator, BasicOpenEnumerator) in enumerating competing reactions. Defaults to True.
        use_minimize (bool): Whether or not to use the minimze enumerator(s)
            (MinimizeGibbsEnumerator, MinimizeGrandPotentialEnumerator) in enumerating
            competing reactions. Defaults to True.
        basic_enumerator_kwargs (dict): Optional kwargs to pass to BasicEnumerator (and BasicOpenEnumerator)
        minimize_enumerator_kwargs (dict): Optional kwargs to pass to
            MinimizeGibbsEnumerator (and MinimizeGrandPotentialEnumerator)
    """

    required_params: List[str] = ["target_formula"]
    optional_params: List[str] = ["rxns", "open_formula", "temp"]

    def run_task(self, fw_spec):
        all_rxns = load_json(self, "rxns", fw_spec)
        target_formula = self["target_formula"]
        open_formula = self.get("open_formula")
        temp = self.get("temp", 300)

        logger.info("Identifying target reactions...")

        target_rxns = []
        for rxn in all_rxns:
            product_formulas = [p.reduced_formula for p in rxn.products]
            if target_formula in product_formulas:
                target_rxns.append(rxn)

        logger.info(
            f"Identified {len(target_rxns)} target reactions out of"
            f" {len(all_rxns)} total reactions."
        )

        competing_rxns_dict = self._create_reactions_dict(all_rxns)
        competing_rxns_dict = ray.put(competing_rxns_dict)

        logger.info("Calculating selectivites...")

        batch_size = 10  # arbitrary

        processed_chunks = []
        for rxns_chunk in grouper(target_rxns, batch_size, fillvalue=None):
            processed_chunks.append(
                get_decorated_rxns_by_chunk.remote(
                    rxns_chunk, competing_rxns_dict, open_formula, temp
                )
            )

        decorated_rxns = []
        for r in tqdm(
            to_iterator(processed_chunks),
            total=len(processed_chunks),
            desc="Selectivity",
        ):
            decorated_rxns.extend(r)

        del processed_chunks

        logger.info("Saving decorated reactions.")

        results = ReactionSet.from_rxns(decorated_rxns)
        dumpfn(results, "rxns.json.gz")  # may overwrite existing rxns.json.gz

    @staticmethod
    def _create_reactions_dict(all_rxns):
        all_rxns_dict = {}

        for r in all_rxns:
            precursors_str = "-".join(sorted([p.reduced_formula for p in r.reactants]))
            if precursors_str in all_rxns_dict:
                all_rxns_dict[precursors_str].add(r)
            else:
                all_rxns_dict[precursors_str] = {r}

        return all_rxns_dict


@explicit_serialize
class CalculateChempotDistance(FiretaskBase):
    """
    Calculates the chemical potential distance using the ChempotDistanceCalculator for a
    set of reactions and stores that within the reaction object data.

    Required params:
        entries (GibbsEntrySet): An entry set to be used for the calculation of
            competitiveness scores.

    Optional params:
        rxns (ReactionSet): A set of reactions of interest. If not provided, will load
            from a rxns.json.gz file
    """

    required_params: List[str] = ["entries"]
    optional_params: List[str] = ["rxns", "cpd_kwargs", "metadata"]

    def run_task(self, fw_spec):
        entries = load_entry_set(self, fw_spec)
        try:
            rxns = load_json(self, "rxns", fw_spec)
        except KeyError:
            rxns = None

        cpd_kwargs = self.get("cpd_kwargs", {})
        metadata = load_json(self, "metadata", fw_spec)

        if rxns:
            logger.info("Decorating reactions with chemical potential distance...")
            results = self._get_decorated_rxns(rxns, entries, cpd_kwargs)

        metadata["cpd_kwargs"] = cpd_kwargs
        dumpfn(metadata, "metadata.json.gz")  # will overwrite existing metadata.json.gz

        if rxns:
            dumpfn(results, "rxns.json.gz")  # will overwrite existing rxns.json.gz

    @staticmethod
    def _get_decorated_rxns(rxns, entries, cpd_kwargs):
        cpd_calc_dict = {}
        new_rxns = []

        open_elem = rxns.open_elem
        if open_elem:
            open_elem_set = {open_elem}
        chempot = rxns.chempot

        for rxn in sorted(rxns, key=lambda rxn: len(rxn.elements), reverse=True):
            chemsys = rxn.chemical_system
            elems = chemsys.split("-")

            for c, cpd_calc in cpd_calc_dict.items():
                if set(c.split("-")).issuperset(elems):
                    break
            else:
                if open_elem:
                    filtered_entries = entries.get_subset_in_chemsys(
                        elems + [str(open_elem)]
                    )
                    filtered_entries = [
                        e.to_grand_entry({Element(open_elem): chempot})
                        for e in filtered_entries
                        if set(e.composition.elements) != open_elem_set
                    ]
                else:
                    filtered_entries = entries.get_subset_in_chemsys(elems)

                cpd_calc = ChempotDistanceCalculator.from_entries(
                    filtered_entries, **cpd_kwargs
                )
                cpd_calc_dict[chemsys] = cpd_calc

            new_rxns.append(cpd_calc.decorate(rxn))

        results = ReactionSet.from_rxns(new_rxns, entries=entries)
        return results


@explicit_serialize
class BuildNetwork(FiretaskBase):
    """
    Builds a reaction network from a set of computed entries, a list of enumerators, and
    a cost function. Optionally performs pathfinding if both the precursors and targets
    optional parameters are provided.

    Required params:
        entries (List[ComputedEntry]): Computed entries to be fed into enumerate()
            methods
        enumerators (List[Enumerator]): Enumerators to run
        cost_function (CostFunction): cost function to use for edge weights in the network

    Optional params:
        precursors (Iterable[Union[Entry, str]]): A list of precursor formulas to be used for pathfinding.
        targets (Iterable[Union[Entry, str]]): A list of target formulas to be used for pathfinding.
        k (int): The number of shortest paths to find to each target. Defaults to 10.
        open_elem (str): An optional open element to be used for modifying reaction energies
            and finding open reactions.
        chempot (float): The chemical potential of the specified open element.
    """

    required_params = ["entries", "enumerators", "cost_function"]
    optional_params = [
        "precursors",
        "targets",
        "k",
        "open_elem",
        "chempot",
        "entries_fn",
    ]

    def run_task(self, fw_spec):
        entries = load_entry_set(self, fw_spec)
        enumerators = self["enumerators"]
        cost_function = self["cost_function"]
        precursors = self.get("precursors")
        targets = self.get("targets")
        k = self.get("k", 10)
        open_elem = self.get("open_elem")
        chempot = self.get("chempot")

        chemsys = "-".join(sorted(list(entries.chemsys)))

        rn = ReactionNetwork(
            entries=entries,
            enumerators=enumerators,
            cost_function=cost_function,
            open_elem=open_elem,
            chempot=chempot,
        )
        rn.build()

        entries_fn = "entries.json.gz"
        graph_fn = "graph.gt.gz"
        network_fn = "network.json.gz"
        pathways_fn = None
        metadata = {}

        if precursors:
            rn.set_precursors(precursors)
        if precursors and targets:
            paths = rn.find_pathways(targets=targets, k=k)

            pathway_set = PathwaySet.from_paths(paths)
            pathways_fn = "pathways.json.gz"
            dumpfn(pathway_set, pathways_fn)

        rn.write_graph(graph_fn)
        dumpfn(rn, network_fn)
        dumpfn(entries, entries_fn)  # will overwrite existing entries.json.gz

        name = f"Reaction Network (Targets: {targets}): {chemsys}"

        return FWAction(
            update_spec={
                "name": name,
                "graph_fn": graph_fn,
                "network_fn": network_fn,
                "pathways_fn": pathways_fn,
                "metadata": metadata,
            }
        )


@explicit_serialize
class RunSolver(FiretaskBase):
    """
    Finds BalancedPathway objects from a set of graph-derived reaction Pathway objects
    by using the PathwaySolver class. Optional parameters are passed to the
    PathwaySolver.solve() function.

    Required params:
        pathways: (List[Pathway]): A list of pathway objects to be considered by the solver
        entries (GibbsEntrySet): GibbsEntrySet containing all entries in the network.
        cost_function (CostFunction): cost function to use for edge weights of solved intermediate reactions
        net_rxn (ComputedReaction): The net reaction representing the complete
            conversion of precursors --> targets.

    Optional params:
        max_num_combos (int): maximum number of combinations to enumerate. Defaults to 4
        find_intermediate_rxns (bool): whether to find intermediate reactions in the
            Solver. Defaults to True.
        intermediate_rxn_energy_cutoff (float): only consider intermediate reactions
            with energies below this cutoff value. Defaults to 0.0.
        use_basic_enumerator (bool): Whether to use the BasicEnumerator to find
            intermediate reactions. Defaults to True.
        use_minimize_enumerator (bool): Whether to use the Minimize enumerator(s) to find
            intermediate reactions. Defaults to False.
        filter_interdependent (bool): Whether to filter out BalancedPathway objects
            which contain interdependent reactions. Defaults to True.
    """

    required_params = ["pathways", "entries", "cost_function", "net_rxn"]
    optional_params = [
        "max_num_combos",
        "find_intermediate_rxns",
        "intermediate_rxn_energy_cutoff",
        "use_basic_enumerator",
        "use_minimize_enumerator",
        "filter_interdependent",
    ]

    def run_task(self, fw_spec):
        entries = load_entry_set(self, fw_spec)
        cost_function = self["cost_function"]
        if not self.get("pathways"):
            pathways = loadfn(fw_spec["pathways_fn"])

        net_rxn = get_computed_rxn(self["net_rxn"], entries)

        solver = PathwaySolver(
            entries=entries,
            pathways=pathways,
            cost_function=cost_function,
        )

        solver_params = {p: self.get(p) for p in self.optional_params if self.get(p)}

        paths = solver.solve(net_rxn, **solver_params)
        pathway_set = PathwaySet.from_paths(paths)

        balanced_pathways_fn = "balanced_pathways.json.gz"
        dumpfn(pathway_set, balanced_pathways_fn)

        return FWAction(update_spec={"balanced_pathways_fn": balanced_pathways_fn})


def get_decorated_rxn(rxn, competing_rxns, precursors_list, temp):
    if len(precursors_list) == 1:
        other_energies = np.array(
            [r.energy_per_atom for r in competing_rxns if r != rxn]
        )
        primary_selectivity = InterfaceReactionHull._primary_selectivity_from_energies(  # pylint: disable=protected-access
            rxn.energy_per_atom, other_energies, temp=temp
        )
        energy_diffs = rxn.energy_per_atom - other_energies
        max_diff = energy_diffs.max()
        secondary_selectivity = max_diff if max_diff > 0 else 0.0
        rxn.data["primary_selectivity"] = primary_selectivity
        rxn.data["secondary_selectivity"] = secondary_selectivity
        decorated_rxn = rxn
    else:
        competing_rxns = competing_rxns.get_rxns()

        if rxn not in competing_rxns:
            competing_rxns.append(rxn)

        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            competing_rxns,
        )

        calc_1 = PrimarySelectivityCalculator(irh=irh, temp=temp)
        calc_2 = SecondarySelectivityCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)

    return decorated_rxn


@ray.remote
def get_decorated_rxns_by_chunk(rxn_chunk, competing_rxns_dict, open_formula, temp):
    decorated_rxns = []

    for rxn in rxn_chunk:
        if not rxn:
            continue

        precursors = [r.reduced_formula for r in rxn.reactants]

        precursors_and_open_formulas = deepcopy(precursors)

        if open_formula:
            precursors_and_open_formulas.append(open_formula)

        competing_rxns = get_all_competing_rxns_from_dict(
            competing_rxns_dict, precursors_and_open_formulas
        )

        if len(precursors) >= 3:
            precursors = list(set(precursors_and_open_formulas) - {open_formula})

        decorated_rxns.append(get_decorated_rxn(rxn, competing_rxns, precursors, temp))

    return decorated_rxns


def get_all_competing_rxns_from_dict(competing_rxns_dict, precursors_and_open_formulas):
    all_possible_strs = get_all_reactant_strs(precursors_and_open_formulas)

    competing_rxns = ReactionSet.from_rxns(
        [
            r
            for rxn_set in [competing_rxns_dict.get(s, []) for s in all_possible_strs]
            for r in rxn_set
        ],
    )
    return competing_rxns
