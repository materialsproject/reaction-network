"""Tests for jobflow-based workflows"""

import pytest
from jobflow.managers.local import run_locally

from rxn_network.flows.core import NetworkFlowMaker, SynthesisPlanningFlowMaker


@pytest.fixture
def network_flow_job(filtered_entries):
    """Create network flow job"""
    return NetworkFlowMaker().make(
        ["Y2O3", "Mn2O3"], ["YMnO3", "Mn3O4"], entries=filtered_entries
    )


@pytest.fixture
def retrosynthesis_flow_job(filtered_entries):
    """Create retrosynthesis flow job"""
    return SynthesisPlanningFlowMaker().make("YMnO3", entries=filtered_entries)


def test_network_flow_job_no_paths(network_flow_job, job_store):
    assert len(network_flow_job.jobs) == 2
    output = run_locally(network_flow_job, store=job_store, ensure_success=True)

    doc = output[network_flow_job.job_uuids[-1]][1].output


def test_retrosynthesis_flow_job(retrosynthesis_flow_job, job_store):
    assert len(retrosynthesis_flow_job.jobs) == 2
    output = run_locally(retrosynthesis_flow_job, store=job_store, ensure_success=True)

    competition_doc = output[retrosynthesis_flow_job.job_uuids[-1]][1].output
