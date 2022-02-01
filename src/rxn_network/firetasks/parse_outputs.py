"""
Firetasks for storing enumerated reaction or network data into a MongoDB.
"""
import os
from typing import List

from fireworks import FiretaskBase, explicit_serialize

from rxn_network.firetasks.utils import env_chk, get_logger, load_json
from rxn_network.utils.database import CalcDb

logger = get_logger(__name__)


@explicit_serialize
class ReactionsToDb(FiretaskBase):
    """
    Stores calculated reactions (rxns.json.gz) in a CalcDb (i.e., MongoDB).

    Required params:
        None

    Optional params:
        rxns (Iterable[Reaction]): Optional list of reactions to store. Otherwise, looks
            for rxns.json.gz file in current folder.
        metadata (dict): Optional metadata to store with the ReactionSet.
        db_file (str): Optional path to CalcDb file.

    """

    required_params: List[str] = []
    optional_params: List[str] = ["rxns", "metadata", "db_file"]

    def run_task(self, fw_spec):
        dir_name = os.path.abspath(os.getcwd())
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = CalcDb(db_file)

        rxns = load_json(self, "rxns", fw_spec)
        metadata = self.get("metadata", fw_spec["metadata"])

        d = {}
        d["dir_name"] = dir_name
        d["name"] = (
            f"Reaction Enumeration (Targets: "
            f"{metadata.get('targets')}): {metadata.get('chemsys')}"
        )
        d["rxns"] = rxns

        d.update(metadata)
        db.insert(d)


@explicit_serialize
class NetworkToDb(FiretaskBase):
    """
    Stores calculated reaction network and paths in a CalcDb (i.e., MongoDB).

    Required params:
        None

    Optional params:
        network (ReactionNetwork): Optional ReactioNetwork object. Otherwise, looks
            for rxns.json.gz file in current folder.
        pathways (List[Pathway]): Optional list of graph pathways to store.
        balanced_patwhays (List[BalancedPathway]): Optional list of balanced pathways to store.
        metadata (dict): Optional metadata to store.
        db_file (str): Optional path to CalcDb file. Otherwise, will acquire from fw_env.
        graph_fn (str): Optional filename of graph-tool Graph object (e.g., graph.gt.gz)

    """

    required_params: List[str] = []
    optional_params: List[str] = [
        "network",
        "pathways",
        "balanced_pathways",
        "metadata",
        "db_file",
        "graph_fn",
    ]

    def run_task(self, fw_spec):
        dir_name = os.path.abspath(os.getcwd())

        network = load_json(self, "network", fw_spec)
        pathways = load_json(self, "pathways", fw_spec)
        balanced_pathways = load_json(self, "balanced_pathways", fw_spec)
        metadata = self.get("metadata", {})

        db_file = env_chk(self.get("db_file"), fw_spec)
        db = CalcDb(db_file)

        graph_fn = self.get("graph_fn") if self.get("graph_fn") else fw_spec["graph_fn"]

        d = {}
        d["name"] = fw_spec.get("name", "Reaction Network")
        d["dir_name"] = dir_name
        d["network"] = network
        d["pathways"] = pathways
        d["balanced_pathways"] = balanced_pathways
        d["graph_fn"] = graph_fn

        d.update(metadata)
        db.insert(d)
