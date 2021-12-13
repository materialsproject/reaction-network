import logging
import os
import sys
from typing import Optional

from maggma.stores import MongoStore
from monty.json import MontyDecoder, jsanitize


class CalcDb:
    def __init__(self, db_file):
        self.db_file = db_file
        self.db = MongoStore.from_db_file(db_file)

    def insert(self, d, update_duplicates=True):
        """
        Insert the task document ot the database collection.

        Args:
            d (dict): task document
            update_duplicates (bool): whether to update the duplicates
        """
        result = self.collection.find_one(
            {"dir_name": d["dir_name"]}, ["dir_name", "task_id"]
        )
        if result is None or update_duplicates:
            d["last_updated"] = datetime.datetime.utcnow()
            if result is None:
                if ("task_id" not in d) or (not d["task_id"]):
                    d["task_id"] = self.db.counter.find_one_and_update(
                        {"_id": "taskid"},
                        {"$inc": {"c": 1}},
                        return_document=ReturnDocument.AFTER,
                    )["c"]
                logger.info(
                    "Inserting {} with taskid = {}".format(d["dir_name"], d["task_id"])
                )
            elif update_duplicates:
                d["task_id"] = result["task_id"]
                logger.info(
                    "Updating {} with taskid = {}".format(d["dir_name"], d["task_id"])
                )
            d = jsanitize(d, allow_bson=True)
            self.collection.update_one(
                {"dir_name": d["dir_name"]}, {"$set": d}, upsert=True
            )
            return d["task_id"]
        else:
            logger.info("Skipping duplicate {}".format(d["dir_name"]))
            return None
