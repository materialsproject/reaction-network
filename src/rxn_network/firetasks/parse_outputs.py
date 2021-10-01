"""
Firetasks for storing enumerated reaction or network data into a MongoDB.
"""
import datetime
import json
import os

from fireworks import FiretaskBase, FWAction, explicit_serialize
from maggma.stores import MongoStore
from monty.json import MontyDecoder, jsanitize
from monty.serialization import loadfn

from rxn_network.firetasks.utils import env_chk, get_logger

logger = get_logger(__name__)


@explicit_serialize
class ReactionsToDb(FiretaskBase):
    """
    Stores calculated reactions (rxns.json) and their metadata (metadata.json) in a
    MongoDB.
    """

    def run_task(self, fw_spec):
        db_file = env_chk(self.get("db_file"), fw_spec)

        d = {}
        rxns = loadfn("rxns.json")
        metadata = loadfn("metadata.json")

        d["name"] = (
            f"Reaction Enumeration (Target: "
            f"{metadata['target']}): {metadata['chemsys']}"
        )
        d["rxns"] = jsanitize(rxns, strict=True)
        d["metadata"] = jsanitize(metadata, strict=True)

        with MongoStore.from_db_file(db_file) as db:
            task_ids = sorted(db.distinct("task_id"))
            if task_ids:
                d["task_id"] = task_ids[-1] + 1
            else:
                d["task_id"] = 1

            d["last_updated"] = datetime.datetime.utcnow()
            db.update(d)


@explicit_serialize
class NetworkToDb(FiretaskBase):
    """
    Stores calculated reaction network in a MongoDB.
    """

    def run_task(self, fw_spec):
        pass
