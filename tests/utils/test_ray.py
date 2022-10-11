"""Tests for ray utils"""
import logging
import os
import sys
from pathlib import Path

import pytest
import ray

from rxn_network.utils.ray import initialize_ray, to_iterator

LOGGER = logging.getLogger(__name__)


def test_initialize_ray(caplog):
    """Test initialize_ray"""
    with caplog.at_level(logging.INFO):
        initialize_ray(quiet=False)

    assert "Ray is not initialized. Checking for existing cluster..." in caplog.text
    assert ray.is_initialized()

    ray.shutdown()


def test_initialize_ray_quiet(caplog):
    """Test initialize_ray"""
    with caplog.at_level(logging.WARNING):
        initialize_ray(quiet=True)

    assert caplog.text == ""
    assert ray.is_initialized()

    ray.shutdown()


def test_initialize_ray_with_slurm_cluster():
    """Test initialize_ray"""

    original_ip_head = os.environ.get("ip_head")
    os.environ["ip_head"] = "test_ip_head"

    with pytest.raises(Exception) as error:
        initialize_ray(quiet=False)
    assert "ConnectionError" in str(error.type)

    if original_ip_head is not None:
        os.environ["ip_head"] = original_ip_head  # tear down


def test_initialize_ray_with_pbs_cluster():
    original_pbs_nnodes = os.environ.get("PBS_NNODES")
    os.environ["PBS_NNODES"] = "2"

    with pytest.raises(Exception) as error:
        initialize_ray(quiet=False)
    assert "ConnectionError" in str(error.type)

    if original_pbs_nnodes is not None:
        os.environ["PBS_NNODES"] = original_pbs_nnodes
