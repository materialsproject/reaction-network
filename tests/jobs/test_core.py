"""Tests for reaction-network jobs"""

import pytest
from jobflow.managers.local import run_locally

from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.jobs.core import (
    CalculateCompetitionMaker,
    GetEntrySetMaker,
    NetworkMaker,
    PathwaySolverMaker,
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
def calculate_competition_maker():
    calculate_competition_maker = CalculateCompetitionMaker()
    return calculate_competition_maker


@pytest.fixture
def competition_job(calculate_competition_maker, all_ymno_rxns, filtered_entries):
    target_formula = "YMnO3"
    job = calculate_competition_maker.make(
        [ReactionSet.from_rxns(all_ymno_rxns)], filtered_entries, target_formula
    )
    return job


@pytest.fixture
def network_maker():
    network_maker = NetworkMaker(
        precursors=["Y2O3", "Mn2O3"], targets=["YMnO3"], calculate_pathways=10
    )
    return network_maker


@pytest.fixture
def network_job(network_maker, all_ymno_rxns):
    job = network_maker.make([all_ymno_rxns])
    return job


@pytest.fixture
def pathway_solver_maker():
    pathway_solver_maker = PathwaySolverMaker(
        precursors=["Y2O3", "Mn2O3"], targets=["YMn2O5", "Mn3O4"], max_num_combos=2
    )
    return pathway_solver_maker


@pytest.fixture
def pathway_solver_job(
    pathway_solver_maker, ymn2o5_mn3o4_paths, mn_o_y_network_entries
):
    job = pathway_solver_maker.make(ymn2o5_mn3o4_paths, mn_o_y_network_entries)
    return job


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
    doc = output[enumeration_job.uuid][1].output
    assert doc.__class__.__name__ == "EnumeratorTaskDocument"


def test_calculate_competition_job(competition_job, job_store):
    output = run_locally(competition_job, store=job_store, ensure_success=True)
    doc = output[competition_job.uuid][1].output
    assert doc.__class__.__name__ == "CompetitionTaskDocument"
    for r in doc.rxns:
        assert r.data["primary_competition"] is not None
        assert r.data["secondary_competition"] is not None
        assert r.data["chempot_distance"] is not None


def test_network_job(network_job, job_store):
    output = run_locally(network_job, store=job_store, ensure_success=True)
    doc = output[network_job.uuid][1].output
    assert doc.__class__.__name__ == "NetworkTaskDocument"
    assert len(doc.paths) == 10


def test_pathway_solver_job(pathway_solver_job, job_store):
    output = run_locally(pathway_solver_job, store=job_store, ensure_success=True)
    doc = output[pathway_solver_job.uuid][1].output
    assert doc.__class__.__name__ == "PathwaySolverTaskDocument"
    assert len(doc.balanced_paths) >= 1
