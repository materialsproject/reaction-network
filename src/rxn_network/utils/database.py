"""Utility functions for interfacing with a reaction database."""

from __future__ import annotations

from jobflow import SETTINGS
from monty.json import MontyDecoder

from rxn_network.core import Composition
from rxn_network.reactions.reaction_set import ReactionSet


def get_rxns_from_db(
    target: str,
    open_elem: str | None = None,
    chempot: float = 0.0,
    fw_id: int | None = None,
):
    """Get a reaction set from the database. This is useful for retrieving synthesis
    planning results.

    Args:
        target: The target formula
        open_elem: The open element
        chempot: The chemical potential
        temp: The temperature (in Kelvin)
        fw_id: The firework id (if present in the metadata)

    """
    store = SETTINGS.JOB_STORE
    store.connect()

    query = {
        "output.target_formula": Composition(target).reduced_formula,
        "output.open_elem": open_elem,
        "output.chempot": chempot,
    }

    if fw_id is not None:
        query["metadata.fw_id"] = fw_id

    data = store.query_one(query, sort={"completed_at": -1}, load=True)

    if not data:
        raise ValueError("Could not find matching firework in database!!")

    if fw_id is not None:
        print("Firework:", data["metadata"]["fw_id"])

    data = MontyDecoder().process_decoded(data["output"])
    print(data.last_updated)

    rxns = data.rxns
    if open_elem is not None:
        rxns = ReactionSet.from_rxns(rxns, open_elem=open_elem, chempot=chempot)

    return rxns
