"""
Utility Fireworks functions borrowed from the atomate package
"""
import logging
import os
import sys
from typing import Optional


def env_chk(
    val: str,
    fw_spec: dict,
    strict: Optional[bool] = True,
    default: Optional[str] = None,
):
    """
    Code borrowed from the atomate package.

    env_chk() is a way to set different values for a property depending
    on the worker machine. For example, you might have slightly different
    executable names or scratch directories on different machines.

    Args:
        val: any value, with ">><<" notation reserved for special env lookup values
        fw_spec: fw_spec where one can find the _fw_env keys
        strict: if True, errors if env format (>><<) specified but cannot be found in fw_spec
        default: if val is None or env cannot be found in non-strict mode,
                 return default
    """
    if val is None:
        return default

    if isinstance(val, str) and val.startswith(">>") and val.endswith("<<"):
        if strict:
            return fw_spec["_fw_env"][val[2:-2]]
        return fw_spec.get("_fw_env", {}).get(val[2:-2], default)
    return val


def get_logger(
    name: str,
    level=logging.DEBUG,
    log_format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
):
    """
    Code borrowed from the atomate package.

    Helper method for acquiring logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(log_format)

    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger
