"""Tests for ray utils"""
import logging
import os

import pytest
import ray
from rxn_network.utils.ray import initialize_ray

LOGGER = logging.getLogger(__name__)


def test_initialize_ray(caplog):
    """Test initialize_ray"""
    if ray.is_initialized():
        ray.shutdown()

    with caplog.at_level(logging.INFO):
        initialize_ray(quiet=False)

    assert "Ray is not initialized. Checking for existing cluster..." in caplog.text
    assert ray.is_initialized()

    ray.shutdown()


def test_initialize_ray_quiet(caplog):
    """Test initialize_ray"""

    if ray.is_initialized():
        ray.shutdown()

    with caplog.at_level(logging.WARNING):
        initialize_ray(quiet=True)

    assert caplog.text == ""
    assert ray.is_initialized()

    ray.shutdown()


def test_initialize_ray_with_slurm_cluster(capsys):
    """Test initialize_ray"""

    original_ip_head = os.environ.get("IP_HEAD")
    os.environ["IP_HEAD"] = "test_ip_head"

    if ray.is_initialized():
        ray.shutdown()

    try:
        initialize_ray()
    except Exception as error:  # if it does fail, make sure it's the right error
        assert "ConnectionError" in str(error.type)  # noqa

    if "Connected" in capsys.readouterr().err:
        pytest.skip("Skipping test_initialize_ray_with_slurm_cluster due to existing cluster")

    if original_ip_head is not None:
        os.environ["IP_HEAD"] = original_ip_head  # tear down


def test_initialize_ray_with_pbs_cluster(capsys):
    original_pbs_nnodes = os.environ.get("PBS_NNODES")
    os.environ["PBS_NNODES"] = "2"

    try:
        initialize_ray()
    except Exception as error:  # if it does fail, make sure it's the right error
        assert "ConnectionError" in str(error.type)  # noqa

    if "Connected" in capsys.readouterr().err:
        pytest.skip("Skipping test_initialize_ray_with_slurm_cluster due to existing cluster")

    if original_pbs_nnodes is not None:
        os.environ["PBS_NNODES"] = original_pbs_nnodes
