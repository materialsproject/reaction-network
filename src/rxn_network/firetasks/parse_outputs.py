"""
Firetasks for storing enumerated reaction or network data into a MongoDB.
"""
import datetime
import json
import os

import json
from fireworks import FiretaskBase, FWAction, explicit_serialize
from maggma.stores import MongoStore
from monty.json import MontyDecoder, jsanitize
from monty.serialization import loadfn

from rxn_network.firetasks.utils import env_chk, get_logger, load_json
from rxn_network.utils.database import CalcDb


logger = get_logger(__name__)


@explicit_serialize
class ReactionsToDb(FiretaskBase):
    """
    Stores calculated reactions (rxns.json.gz) and their metadata (metadata.json) in a
    CalcDb (i.e., MongoDB).
    """

    required_params = ["rxns", "metadata"]
    optional_params = ["db_file", "rxns_fn", "metadata_fn"]

    def run_task(self, fw_spec):
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = CalcDb(db_file)

        rxns = load_json(self, "rxns", fw_spec)
        metadata = load_json(self, "metadata", fw_spec)

        d = {}
        d["name"] = (
            f"Reaction Enumeration (Targets: "
            f"{metadata['targets']}): {metadata['chemsys']}"
        )
        d["rxns"] = rxns
        d["metadata"] = metadata

        db.insert(d)


@explicit_serialize
class NetworkToDb(FiretaskBase):
    """
    Stores calculated reaction network and paths in a CalcDb (i.e., MongoDB).
    """

    required_params = []
    optional_params = [
        "network",
        "paths",
        "balanced_paths",
        "open_elem",
        "chempot",
        "db_file",
    ]

    def run_task(self, fw_spec):
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = CalcDb(db_file)

        graph_fn = self.get("graph_fn") if self.get("graph_fn") else fw_spec["graph_fn"]

        network = load_json(self, "network", fw_spec)
        paths = load_json(self, "paths", fw_spec)
        balanced_paths = load_json(self, "balanced_paths", fw_spec)

        d = {}
        d["network"] = network
        d["paths"] = paths
        d["balanced_paths"] = balanced_paths
        d["graph_fn"] = graph_fn

        db.insert(d)
