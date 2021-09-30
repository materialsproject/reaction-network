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
        entries = self.get("entries", None)

        if not entries:
            entries = fw_spec["entries"]
        else:
            entries = entries["entries"]

        entries = GibbsEntrySet(entries)
        chemsys = "-".join(sorted(list(entries.chemsys)))
        target = enumerators[0].target
        added_elems = None

        if target:
            added_elems = entries.chemsys - {
                str(e) for e in Composition(target).elements
            }
            added_elems = "-".join(sorted(list(added_elems)))

        metadata = {
            "chemsys": chemsys,
            "enumerators": enumerators,
            "target": target,
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
class RunNetwork(FiretaskBase):
    pass
