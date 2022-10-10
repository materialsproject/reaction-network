"""Tests for reaction-network jobs"""

import pytest
from jobflow.managers.local import run_locally

from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.jobs.core import (
    CalculateSelectivitiesMaker,
    GetEntrySetMaker,
    NetworkMaker,
    ReactionEnumerationMaker,
)
from rxn_network.reactions.reaction_set import ReactionSet


@pytest.fixture
def entry_set_maker():
    entry_set_maker = GetEntrySetMaker(entry_db_name=None)
    return entry_set_maker


@pytest.fixture
def entry_job(entry_set_maker):
    job = entry_set_maker.make("Mn-O-Y")
    return job


@pytest.fixture
def reaction_enumeration_maker():
    reaction_enumeration_maker = ReactionEnumerationMaker()
    return reaction_enumeration_maker


@pytest.fixture
def enumeration_job(reaction_enumeration_maker, filtered_entries):
    enumerators = [BasicEnumerator(precursors=["Y2O3", "MnO2"])]
    job = reaction_enumeration_maker.make(enumerators, filtered_entries)
    return job


@pytest.fixture
def calculate_selectivities_maker():
    calculate_selectivities_maker = CalculateSelectivitiesMaker()
    return calculate_selectivities_maker


@pytest.fixture
def selectivities_job(calculate_selectivities_maker, ymno3_rxns, filtered_entries):
    target_formula = "YMnO3"
    job = calculate_selectivities_maker.make(
        [ReactionSet.from_rxns(ymno3_rxns)], filtered_entries, target_formula
    )
    return job


@pytest.fixture
def network_job(network_maker, filtered_entries):
    job = network_maker.make(filtered_entries)
    return job


@pytest.fixture
def network_maker():
    network_maker = NetworkMaker()
    return network_maker


def test_entry_job(entry_job, job_store):
    output = run_locally(entry_job, store=job_store, ensure_success=True)

    doc = output[entry_job.uuid][1].output
    entries = doc.entries

    assert doc.task_label == "get_and_process_entries"
    assert doc.__class__.__name__ == "EntrySetDocument"
    assert entries.__class__.__name__ == "GibbsEntrySet"

    assert entries.chemsys == {"Mn", "O", "Y"}


def test_enumeration_job(enumeration_job, job_store):
    output = run_locally(enumeration_job, store=job_store, ensure_success=True)


def test_calculate_selectivities_job(selectivities_job, job_store):
    output = run_locally(selectivities_job, store=job_store, ensure_success=True)


def test_network_job(network_maker, job_store):
    output = run_locally(job, store=job_store, ensure_success=True)
