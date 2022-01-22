"""
Database utilities for the rxn_network package to facilitate connection to databases
(e.g., MongoDB)
"""

from datetime import datetime

from maggma.core import StoreError
from maggma.stores import MongoStore
from monty.json import MSONable, jsanitize

from rxn_network.firetasks.utils import get_logger

logger = get_logger(__name__)


class CalcDb(MSONable):
    """
    A lightweight class for connecting and interacting with documents in a MongoDB database.
    """

    def __init__(self, db_file: str):
        """
        Args:
            db_file: Path to the database file.
        """
        self.db_file = db_file

        try:
            db = MongoStore.from_db_file(db_file)
            db.connect()
        except StoreError as e:
            logger.error(e)
            logger.error(f"Could not connect to database {db_file}")

        self.db = db

    def insert(self, d: dict, update_duplicates: bool = True):
        """
        Insert the task document in the database collection.

        Args:
            d: task document
            update_duplicates: whether to update the duplicates
        """
        result = self.db.query_one({"dir_name": d["dir_name"]}, ["dir_name", "task_id"])
        if result is None or update_duplicates:
            d["last_updated"] = datetime.utcnow()
            if result is None:
                if ("task_id" not in d) or (not d["task_id"]):
                    d["task_id"] = self.db.count() + 1
                logger.info(f"Inserting {d['dir_name']} with taskid = {d['task_id']}")
            elif update_duplicates:
                d["task_id"] = result["task_id"]
                logger.info(f"Updating {d['dir_name']} with taskid = {d['task_id']}")

            d = jsanitize(d, allow_bson=True)
            self.db.update(
                d,
                "dir_name",
            )
            task_id = d["task_id"]
        else:
            logger.info(f"Skipping duplicate {d['dir_name']}")
            task_id = None

        return task_id
