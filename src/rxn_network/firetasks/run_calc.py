"""
Firetasks for running enumeration and network calculations
"""
import json

from fireworks import FiretaskBase, FWAction, explicit_serialize
from monty.json import MontyEncoder
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.firetasks.utils import env_chk, get_logger
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.reaction_set import ReactionSet

logger = get_logger(__name__)


@explicit_serialize
class RunEnumerators(FiretaskBase):
    """
    Run a list of enumerators on a provided set of computed entries and dump the
    calculated ComputedReaction objects to a file (rxns.json). Metadata is stored as metadata.json.

    Required params:
        enumerators (List[Enumerator]): Enumerators to run
        entries (EntrySet): Computed entries to be fed into enumerate()
            methods

    """

    required_params = ["enumerators", "entries"]

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
            "enumerators": enumerators,
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
class BuildNetwork(FiretaskBase):
    """
    Builds a reaction network from a set of computed entries, a list of enumerators , and
    a cost function.

    Required params:
        entries (List[ComputedEntry]): Computed entries to be fed into enumerate()
            methods
        enumerators (List[Enumerator]): Enumerators to run
        cost_function (CostFunction): cost function to use for edge weights
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
    Balance reaction pathways.

    Required params:
        entries (Solver): solver to use for balancing

    Optional params:
        max_num_combos (int): maximum number of combinations to enumerate
        find_intermediate_rxns (bool): whether to find intermediate reactions
    """

    required_params = ["pathways", "entries", "cost_function", "net_rxn"]
    optional_params = [
        "max_num_combos",
        "find_intermediate_rxns",
        "intermediate_rxn_energy_cutoff",
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
