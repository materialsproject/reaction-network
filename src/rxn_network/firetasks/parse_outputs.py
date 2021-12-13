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
        rxns_fn = get("rxns_fn", "rxns.json")
        metadata_fn = get("metadata_fn", "metadata.json")

        rxns = loadfn(rxns_fn)
        metadata = loadfn(metadata_fn)

        d = {}
        d["name"] = (
            f"Reaction Enumeration (Targets: "
            f"{metadata['targets']}): {metadata['chemsys']}"
        )
        d["rxns"] = jsanitize(rxns, strict=True)
        d["metadata"] = jsanitize(metadata, strict=True)

        store_in_mongo_db(d, db_file)


@explicit_serialize
class NetworkToDb(FiretaskBase):
    """
    Stores calculated reaction network and paths in a MongoDB.
    """

    def run_task(self, fw_spec):
        db_file = env_chk(self.get("db_file"), fw_spec)
        network_fn = self.get("network_fn", "network.json")
        graph_fn = self.get("graph_fn", "graph.gt.gz")
        paths_fn = self.get("paths_fn", "paths.json")

        network = loadfn("network_fn.json")
        metadata = loadfn("metadata.json")

        d["name"] = (
            f"Reaction Network (Targets: "
            f"{metadata['targets']}): {metadata['chemsys']}"
        )
        d["network"] = jsanitize(network, strict=True)
        d["paths"] = jsanitize(paths, strict=True)
        d["graph_fn"] = graph_fn
        d["metadata"] = jsanitize(metadata, strict=True)

        store_in_mongo_db(d, db_file)


def store_in_mongo_db(d, db_file):
    with MongoStore.from_db_file(db_file) as db:
        task_ids = sorted(db.distinct("task_id"))

        if task_ids:
            d["task_id"] = task_ids[-1] + 1
        else:
            d["task_id"] = 1

        d["last_updated"] = datetime.datetime.utcnow()
        db.update(d)
