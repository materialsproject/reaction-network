"""Tests for reaction-network jobs"""

import pytest
from jobflow.managers.local import run_locally

from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.jobs.core import (
    GetEntrySetMaker,
    NetworkMaker,
    PathwaySolverMaker,
    ReactionEnumerationMaker,
)


@pytest.fixture
def entry_set_maker():
    return GetEntrySetMaker(entry_db_name=None)


@pytest.fixture
def entry_job(entry_set_maker):
    return entry_set_maker.make("Mn-O-Y")


@pytest.fixture
def reaction_enumeration_maker():
    return ReactionEnumerationMaker()


@pytest.fixture
def enumeration_job(reaction_enumeration_maker, filtered_entries):
    enumerators = [BasicEnumerator(precursors=["Y2O3", "MnO2"])]
    return reaction_enumeration_maker.make(enumerators, filtered_entries)


@pytest.fixture
def network_maker():
    return NetworkMaker(precursors=["Y2O3", "Mn2O3"], targets=["YMn2O5", "Mn3O4"], calculate_pathways=10)


@pytest.fixture
def network_job(network_maker, all_ymno_rxns):
    return network_maker.make([all_ymno_rxns])


@pytest.fixture
def pathway_solver_maker():
    return PathwaySolverMaker(precursors=["Y2O3", "Mn2O3"], targets=["YMn2O5", "Mn3O4"], max_num_combos=2)


@pytest.fixture
def pathway_solver_job(pathway_solver_maker, ymn2o5_mn3o4_paths, mn_o_y_network_entries):
    return pathway_solver_maker.make(ymn2o5_mn3o4_paths, mn_o_y_network_entries)


def test_entry_job(entry_job, job_store):
    """Note: this test will fail if there is no internet connection."""
    output = run_locally(entry_job, store=job_store, ensure_success=True)

    doc = output[entry_job.uuid][1].output
    entries = doc.entries

    assert doc.task_label == "get_and_process_entries"
    assert doc.__class__.__name__ == "EntrySetDocument"
    assert entries.__class__.__name__ == "GibbsEntrySet"

    assert entries.chemsys == {"Mn", "O", "Y"}


def test_enumeration_job(enumeration_job, job_store):
    output = run_locally(enumeration_job, store=job_store, ensure_success=True)
    doc = output[enumeration_job.uuid][1].output
    assert doc.__class__.__name__ == "EnumeratorTaskDocument"


def test_network_job(network_job, job_store):
    output = run_locally(network_job, store=job_store, ensure_success=True)
    doc = output[network_job.uuid][1].output
    assert doc.__class__.__name__ == "NetworkTaskDocument"
    assert len(doc.paths) == 20


def test_pathway_solver_job(pathway_solver_job, job_store):
    output = run_locally(pathway_solver_job, store=job_store, ensure_success=True)
    doc = output[pathway_solver_job.uuid][1].output
    assert doc.__class__.__name__ == "PathwaySolverTaskDocument"
    assert len(doc.balanced_paths) >= 1
