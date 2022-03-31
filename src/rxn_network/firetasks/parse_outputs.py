"""
Firetasks for storing enumerated reaction or network data into a MongoDB.
"""
import os
from typing import List

from fireworks import FiretaskBase, explicit_serialize
from monty.json import jsanitize

from rxn_network.firetasks.utils import env_chk, get_logger, load_json
from rxn_network.utils.database import CalcDb
from rxn_network.utils.models import EnumeratorTask

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
        use_gridfs (bool): Whether or not to store the reactions in GridFS. Defaults to
            True if not supplied.

    """

    required_params: List[str] = []
    optional_params: List[str] = [
        "rxns",
        "metadata",
        "db_file",
        "use_gridfs",
    ]

    def run_task(self, fw_spec):
        rxns = load_json(self, "rxns", fw_spec)
        metadata = load_json(self, "metadata", fw_spec)
        db_file = env_chk(self.get("db_file"), fw_spec)
        use_gridfs = self.get("use_gridfs", True)

        db = CalcDb(db_file)

        task = EnumeratorTask.from_rxns_and_metadata(rxns, metadata)
        d = jsanitize(task.dict(), strict=True, allow_bson=True)

        if use_gridfs:
            del d["rxns"]  # remove rxns from doc to store later in GridFS

        task_id = db.insert(d)

        if use_gridfs:
            d_fs = {"task_id": task_id, "rxns": rxns}
            db.insert_gridfs(d_fs, metadata_keys=["task_id"])


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


@explicit_serialize
class ChangeDir(FiretaskBase):
    """
    FireTask to create new folder with the option of changing directory to the new folder.

    Required params:
        folder_name (str): folder name.

    Optional params:
        change_dir(bool): change working dir to new folder after creation.
            Defaults to False.
        relative_path (bool): whether folder name is relative or absolute.
            Defaults to True.
    """

    required_params = ["folder_name"]

    def run_task(self, fw_spec):
        os.chdir(self["folder_name"])
