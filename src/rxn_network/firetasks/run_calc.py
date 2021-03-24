import json
from monty.json import MontyEncoder
from monty.serialization import dumpfn

from fireworks import explicit_serialize, FiretaskBase, FWAction
from rxn_network.firetasks.utils import get_logger, env_chk
from rxn_network.entries.entry_set import GibbsEntrySet


logger = get_logger(__name__)


@explicit_serialize
class RunEnumerators(FiretaskBase):
    required_params = ["enumerators", "entries"]

    def run_task(self, fw_spec):
        enumerators = self["enumerators"]
        entries = self.get("entries", None)

        if not entries:
            entries = fw_spec['entries']
        else:
            entries = entries["entries"]

        entries = GibbsEntrySet(entries)
        chemsys = "-".join(sorted(list(entries.chemsys)))

        metadata = {}
        metadata["chemsys"] = chemsys
        metadata["enumerators"] = enumerators
        metadata["targets"] = [enumerator.target for enumerator in enumerators]
        metadata["entries"] = entries

        results = []
        for enumerator in enumerators:
            rxns = enumerator.enumerate(entries)
            results.extend(rxns)

        dumpfn(results, "rxns.json")
        dumpfn(metadata, "metadata.json")


@explicit_serialize
class RunNetwork(FiretaskBase):
    pass