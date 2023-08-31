"""Tests for jobflow-based workflows"""

import pytest
from jobflow.managers.local import run_locally
from pymatgen.core.periodic_table import Element

from rxn_network.flows.core import NetworkFlowMaker, SynthesisPlanningFlowMaker


@pytest.fixture
def network_flow_job(filtered_entries):
    """Create network flow job"""
    return NetworkFlowMaker().make(
        ["Y2O3", "Mn2O3"], ["YMnO3", "Mn3O4"], entries=filtered_entries
    )


@pytest.fixture
def synthesis_planning_flow_job(filtered_entries):
    """Create retrosynthesis flow job"""
    return SynthesisPlanningFlowMaker().make("YMnO3", entries=filtered_entries)


@pytest.fixture
def synthesis_planning_flow_additional_elems_job(gibbs_entries_with_na_cl):
    """Create retrosynthesis flow job with Na and Cl as additional elems"""
    return SynthesisPlanningFlowMaker().make(
        "Y2Mn2O7", entries=gibbs_entries_with_na_cl, added_elems=["Na", "Cl"]
    )


@pytest.fixture
def synthesis_planning_flow_open_job(filtered_entries):
    """Create retrosynthesis flow job with open oxygen"""
    return SynthesisPlanningFlowMaker(open_elem=Element("O"), chempots=[0.0]).make(
        "YMnO3", entries=filtered_entries
    )


def test_network_flow_job_no_paths(network_flow_job, job_store):
    assert len(network_flow_job.jobs) == 2
    output = run_locally(network_flow_job, store=job_store, ensure_success=True)

    doc = output[network_flow_job.job_uuids[-1]][1].output
    assert doc is not None


def test_synthesis_planning_flow_job(synthesis_planning_flow_job, job_store):
    assert len(synthesis_planning_flow_job.jobs) == 2
    output = run_locally(
        synthesis_planning_flow_job, store=job_store, ensure_success=True
    )

    competition_doc = output[synthesis_planning_flow_job.job_uuids[-1]][1].output
    assert competition_doc is not None


def test_synthesis_planning_flow_additional_elems_job(
    synthesis_planning_flow_additional_elems_job, job_store
):
    assert len(synthesis_planning_flow_additional_elems_job.jobs) == 2
    output = run_locally(
        synthesis_planning_flow_additional_elems_job,
        store=job_store,
        ensure_success=True,
    )

    competition_doc = output[
        synthesis_planning_flow_additional_elems_job.job_uuids[-1]
    ][1].output
    assert competition_doc is not None


@pytest.mark.skip(reason="Chemical potential distance difficulty in open systems")
def test_synthesis_planning_flow_open_job(synthesis_planning_flow_open_job, job_store):
    assert len(synthesis_planning_flow_open_job.jobs) == 4
    output = run_locally(
        synthesis_planning_flow_open_job, store=job_store, ensure_success=True
    )

    competition_doc = output[synthesis_planning_flow_open_job.job_uuids[-1]][1].output
    assert competition_doc is not None
