"""
Firetasks for running enumeration and network calculations
"""
import json

from fireworks import FiretaskBase, FWAction, explicit_serialize
from monty.json import MontyEncoder
from monty.serialization import dumpfn
from pymatgen.core import Composition

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.firetasks.utils import env_chk, get_logger
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.network.network import ReactionNetwork

logger = get_logger(__name__)


@explicit_serialize
class RunEnumerators(FiretaskBase):
    """
    Run a list of enumerators on a provided set of computed entries and dump the
    calculated ComputedReaction objects to a file (rxns.json). Metadata is stored as metadata.json.

    Required params:
        enumerators (List[Enumerator]): Enumerators to run
        entries (List[ComputedEntry]): Computed entries to be fed into enumerate()
            methods

    """

    required_params = ["enumerators", "entries"]

    def run_task(self, fw_spec):
        enumerators = self["enumerators"]
        entries = self["entries"] if self["entries"] else fw_spec["entries"]
        print(entries.entries)
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

        dumpfn(results, "rxns.json")
        dumpfn(metadata, "metadata.json")


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
    optional_params = ["open_elem", "chempot"]

    def run_task(self, fw_spec):
        entries = (
            self.get(["entries"]) if "entries" in self else self.fw_spec["entries"]
        )
        enumerators = self.get(["enumerators"])
        cost_function = self["cost_function"]

        entries = GibbsEntrySet(entries)
        chemsys = "-".join(sorted(list(entries.chemsys)))

        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }

        metadata = {"chemsys": chemsys, "enumerators": enumerators, "targets": targets}

        results = []

        rn = ReactionNetwork(entries, enumerators, cost_function)
        rn.build()

        results = ReactionSet.from_rxns(results)

        dumpfn(results, "rxns.json")
        dumpfn(metadata, "metadata.json")


@explicit_serialize
class FindPathways(FiretaskBase):
    """
    Finds pathways in an existing reaction network.

    Required params:
        entries (List[ComputedEntry]): Computed entries to be fed into enumerate()
            methods
        enumerators (List[Enumerator]): Enumerators to run
        cost_function (CostFunction): cost function to use for edge weights
    """

    required_params = ["reaction_network", "graph_fn" "precursors", "targets"]
    optional_params = ["k"]

    def run_task(self, fw_spec):
        rn = self["reaction_network"]
        graph_fn = self["graph_fn"]
        precursors = self["precursors"]
        targets = self["targets"]

        k = self.get("k")

        rn.load_graph(graph_fn)

        rn.set_precursors(precursors)
        paths = rn.find_pathways(targets=targets, k=k)

        dumpfn(paths, "shortest_paths.json")


@explicit_serialize
class RunSolver(FiretaskBase):
    """
    Balance reaction pathways.

    Required params:
        solver (Solver): solver to use for balancing

    Optional params:
        max_num_combos (int): maximum number of combinations to enumerate
        find_intermediate_rxns (bool): whether to find intermediate reactions
    """

    required_params = ["solver", "net_rxn"]
    optional_params = [
        "max_num_combos",
        "find_intermediate_rxns",
        "intermediate_rxn_energy_cutoff",
        "fitler_interdependent",
    ]

    def run_task(self, fw_spec):
        solver = self["solver"]
        net_rxn = self["net_rxn"]

        max_num_combos = self.get("max_num_combos", None)
        find_intermediate_rxns = self.get("find_intermediate_rxns", None)
        intermediate_rxn_energy_cutoff = self.get(
            "intermediate_rxn_energy_cutoff", None
        )
        filter_interdependent = self.get("filter_interdependent", None)

        paths = solver.solve(
            net_rxn,
            max_num_combos=max_num_combos,
            find_intermediate_rxns=find_intermediate_rxns,
            intermediate_rxn_energy_cutoff=intermediate_rxn_energy_cutoff,
            filter_interdependent=filter_interdependent,
        )

        dumpfn(paths, "balanced_paths.json")
