"""
Firetasks for running enumeration and network calculations
"""
import warnings
from typing import List

from fireworks import FiretaskBase, FWAction, explicit_serialize
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition

from rxn_network.costs.competitiveness import CompetitivenessScoreCalculator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.firetasks.utils import get_logger, load_json
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.reaction_set import ReactionSet

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
    optional_params: List[str] = []

    def run_task(self, fw_spec):
        enumerators = self["enumerators"]
        entries = self["entries"] if self["entries"] else fw_spec["entries"]
        entries = GibbsEntrySet(entries)
        chemsys = "-".join(sorted(list(entries.chemsys)))

        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }
        added_elems = None

        if targets:
            added_elems = entries.chemsys - {
                str(e) for target in targets for e in Composition(target).elements
            }
            added_elems = "-".join(sorted(list(added_elems)))

        metadata = {
            "chemsys": chemsys,
            "enumerators": [e.as_dict() for e in enumerators],
            "targets": list(targets),
            "added_elems": added_elems,
        }

        results = []
        for enumerator in enumerators:
            rxns = enumerator.enumerate(entries)
            results.extend(rxns)

        results = ReactionSet.from_rxns(results)

        dumpfn(results, "rxns.json.gz")

        return FWAction(update_spec={"rxns_fn": "rxns.json.gz", "metadata": metadata})


@explicit_serialize
class CalculateCScores(FiretaskBase):
    """
    Calculates the competitiveness score using the CompetitivenessScoreCalculator for a
    set of reactions and stores that within the reaction object data.

    Required params:
        entries (GibbsEntrySet): An entry set to be used for the calculation of
            competitiveness scores.
        cost_function (CostFunction): The cost function to be used in determining the
            cost of competing reactions.

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

    required_params: List[str] = ["entries", "cost_function"]
    optional_params: List[str] = [
        "rxns",
        "k",
        "open_phases",
        "open_elem",
        "chempot",
        "use_basic",
        "use_minimize",
        "basic_enumerator_kwargs",
        "minimize_enumerator_kwargs",
    ]

    def run_task(self, fw_spec):
        entries = self["entries"] if self["entries"] else fw_spec["entries"]
        cost_function = self["cost_function"]
        rxns = ReactionSet.from_dict(load_json(self, "rxns", fw_spec))
        k = self.get("k", 15)
        open_phases = self.get("open_phases")
        open_elem = self.get("open_elem")
        chempot = self.get("chempot", 0.0)
        use_basic = self.get("use_basic", True)
        use_minimize = self.get("use_minimize", True)
        basic_enumerator_kwargs = self.get("basic_enumerator_kwargs", {})
        minimize_enumerator_kwargs = self.get("minimize_enumerator_kwargs", {})

        if use_minimize and open_phases and not open_elem:
            open_comp = Composition(open_phases[0])
            if open_comp.is_element:
                open_elem = open_comp.elements[0]
                warnings.warn(f"Using open phase element {open_elem}")

        calc = CompetitivenessScoreCalculator(
            entries=entries,
            cost_function=cost_function,
            open_phases=open_phases,
            open_elem=open_elem,
            chempot=chempot,
            use_basic=use_basic,
            use_minimize=use_minimize,
            basic_enumerator_kwargs=basic_enumerator_kwargs,
            minimize_enumerator_kwargs=minimize_enumerator_kwargs,
        )

        costs = [cost_function.evaluate(r) for r in rxns]
        sorted_rxns = [r for _, r in sorted(zip(costs, rxns), key=lambda x: x[0])]
        new_rxns = []

        for rxn in sorted_rxns[:k]:
            new_rxns.append(calc.decorate(rxn))
        for rxn in sorted_rxns[k:]:
            rxn.data.update({"c_score": None})
            new_rxns.append(rxn)

        results = ReactionSet.from_rxns(new_rxns)
        dumpfn(results, "rxns.json.gz")  # may overwrite existing rxns.json.gz

        return FWAction(
            mod_spec=[{"_set": {"metadata->cost_function": cost_function.as_dict()}}]
        )


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
    optional_params = ["precursors", "targets", "k", "open_elem", "chempot"]

    def run_task(self, fw_spec):
        entries = self["entries"] if self["entries"] else fw_spec["entries"]
        enumerators = self["enumerators"]
        cost_function = self["cost_function"]
        precursors = self.get("precursors")
        targets = self.get("targets")
        k = self.get("k", 10)
        open_elem = self.get("open_elem")
        chempot = self.get("chempot")

        entries = GibbsEntrySet(entries)
        chemsys = "-".join(sorted(list(entries.chemsys)))

        rn = ReactionNetwork(
            entries=entries,
            enumerators=enumerators,
            cost_function=cost_function,
            open_elem=open_elem,
            chempot=chempot,
        )
        rn.build()

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

        name = f"Reaction Network (Targets: " f"{targets}): {chemsys}"

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
        entries = self["entries"] if self["entries"] else fw_spec["entries"]
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
