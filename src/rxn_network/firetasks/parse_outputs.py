import json
from monty.json import MontyEncoder

from fireworks import FiretaskBase, FWAction, explicit_serialize
from rxn_network.firetasks.utils import get_logger, env_chk
from maggma.stores import MongoStore


logger = get_logger(__name__)


@explicit_serialize
class ReactionsToDb(FiretaskBase):
    def run_task(self, fw_spec):
        calc_dir = self.get("calc_dir", os.getcwd())
        db_file = env_chk(self.get("db_file"), fw_spec)

        with open(os.path.join(calc_dir, "rxns.json"), "r") as fp:
            task_doc = json.load(fp)

        db = MongoStore.from_db_file(db_file, admin=True)
        db.insert(task_doc)

@explicit_serialize
class NetworkToDb(FiretaskBase):
    def run_task(self, fw_spec):
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = MongoStore.from_db_file(db_file, admin=True)