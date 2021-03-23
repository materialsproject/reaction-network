import json
from monty.json import MontyEncoder

from fireworks import explicit_serialize, FiretaskBase, FWAction
from rxn_network.firetasks.utils import get_logger, env_chk

logger = get_logger(__name__)

@explicit_serialize
class RunEnumerators(FiretaskBase):
    required_params = ["enumerators", "entries"]

    def run_task(self, fw_spec):
        enumerators = self.get("enumerators")
        entries = self.get("entries")

        results = []
        for enumerator in enumerators:
            rxns = enumerator.enumerate(entries)
            results.extend(rxns)

        with open("rxns.json", "w") as fp:
            json.dump(task_doc, fp, cls=MontyEncoder)

