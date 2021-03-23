import logging
import os
import sys

def env_chk(val, fw_spec, strict=True, default=None):
    """
    Code borrowed from the atomate package.

    env_chk() is a way to set different values for a property depending
    on the worker machine. For example, you might have slightly different
    executable names or scratch directories on different machines.

    env_chk() works using the principles of the FWorker env in FireWorks.

    This helper method translates string "val" that looks like this:
    ">>ENV_KEY<<"
    to the contents of:
    fw_spec["_fw_env"][ENV_KEY]

    Otherwise, the string "val" is interpreted literally and passed-through as is.

    The fw_spec["_fw_env"] is in turn set by the FWorker. For more details,
    see: https://materialsproject.github.io/fireworks/worker_tutorial.html

    Since the fw_env can be set differently for each FireWorker, one can
    use this method to translate a single "val" into multiple possibilities,
    thus achieving different behavior on different machines.

    Args:
        val: any value, with ">><<" notation reserved for special env lookup values
        fw_spec: (dict) fw_spec where one can find the _fw_env keys
        strict (bool): if True, errors if env format (>><<) specified but cannot be found in fw_spec
        default: if val is None or env cannot be found in non-strict mode,
                 return default
    """
    if val is None:
        return default

    if isinstance(val, str) and val.startswith(">>") and val.endswith("<<"):
        if strict:
            return fw_spec['_fw_env'][val[2:-2]]
        return fw_spec.get('_fw_env', {}).get(val[2:-2], default)
    return val


def get_logger(name, level=logging.DEBUG,
               log_format='%(asctime)s %(levelname)s %(name)s %(message)s',
               stream=sys.stdout):
    """Helper method for acquiring logger; code borrowed from the atomate package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
