"""Functions for working with Ray (parallelization library)."""
import os

import ray

from rxn_network.utils.funcs import get_logger

logger = get_logger(__name__)


def initialize_ray(quiet: bool = False):
    """
    Simple function to initialize ray. Basic support for running ray on multiple nodes.
    Currently supports SLURM and PBS job schedulers.

    SLURM:
        Checks enviornment for existence of "ip_head" for situations where the user is
        running on multiple nodes. Automatically creats a new ray cluster if it has not
        been initialized. See https://github.com/NERSC/slurm-ray-cluster/
    PBS:
        Checks environment for PBS_NNODES > 1.

    """
    if not quiet:
        logger.setLevel("INFO")
    if not ray.is_initialized():
        logger.info("Ray is not initialized. Checking for existing cluster...")
        if os.environ.get("ip_head") or int(os.environ.get("PBS_NNODES", 0)) > 1:
            ray.init(
                address="auto",
            )
        else:
            logger.info(
                "Could not identify existing Ray instance. Creating a new one..."
            )
            ray.init()

        logger.info(
            f"HOST: {ray.nodes()[0]['NodeManagerHostname']}, "
            f"Num CPUs: {ray.cluster_resources()['CPU']}, "
            f"Total Memory: {ray.cluster_resources()['memory']}"
        )
    else:
        logger.info("Ray is already initialized.")


def to_iterator(obj_ids, get_obj_ids: bool = False):
    """
    Method to convert a list of ray object ids to an iterator that can be used in a for
    loop.
    """
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        if get_obj_ids:
            yield done[0], ray.get(done[0])
        else:
            yield ray.get(done[0])
