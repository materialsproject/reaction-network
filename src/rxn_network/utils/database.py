import datetime
import logging
import os
import sys
from typing import Optional

from maggma.core import StoreError
from maggma.stores import MongoStore
from monty.json import jsanitize
from pymongo import ReturnDocument

from rxn_network.firetasks.utils import get_logger

logger = get_logger(__name__)


class CalcDb:
    def __init__(self, db_file):
        self.db_file = db_file
        try:
            db = MongoStore.from_db_file(db_file)
            db.connect()
        except StoreError as e:
            logger.error(e)
            logger.error("Could not connect to database {}".format(db_file))

        self.db = db

    def insert(self, d, update_duplicates=True):
        """
        Insert the task document in the database collection.

        Args:
            d (dict): task document
            update_duplicates (bool): whether to update the duplicates
        """
        result = self.db.query_one({"dir_name": d["dir_name"]}, ["dir_name", "task_id"])
        if result is None or update_duplicates:
            d["last_updated"] = datetime.datetime.utcnow()
            if result is None:
                if ("task_id" not in d) or (not d["task_id"]):
                    d["task_id"] = self.db.count() + 1
                logger.info(
                    "Inserting {} with taskid = {}".format(d["dir_name"], d["task_id"])
                )
            elif update_duplicates:
                d["task_id"] = result["task_id"]
                logger.info(
                    "Updating {} with taskid = {}".format(d["dir_name"], d["task_id"])
                )
            d = jsanitize(d, allow_bson=True)
            self.db.update(
                d,
                "dir_name",
            )
            return d["task_id"]
        else:
            logger.info("Skipping duplicate {}".format(d["dir_name"]))
            return None
