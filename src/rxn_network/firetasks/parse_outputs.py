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
        dir_name = os.path.abspath(os.getcwd())
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
        "pathways",
        "balanced_pathways",
        "open_elem",
        "chempot",
        "db_file",
    ]

    def run_task(self, fw_spec):
        dir_name = os.path.abspath(os.getcwd())
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = CalcDb(db_file)

        graph_fn = self.get("graph_fn") if self.get("graph_fn") else fw_spec["graph_fn"]

        network = load_json(self, "network", fw_spec)
        pathways = load_json(self, "pathways", fw_spec)
        balanced_pathways = load_json(self, "balanced_pathways", fw_spec)

        d = {}
        d["dir_name"] = dir_name
        d["network"] = network
        d["pathways"] = pathways
        d["balanced_pathways"] = balanced_pathways
        d["graph_fn"] = graph_fn

        db.insert(d)
