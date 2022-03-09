"""
Reaction enumerator classes and associated utilities.
"""

import ray

from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)

if not ray.is_initialized():
    ray.init(
        _redis_password="test_password",
        ignore_reinit_error=True,
    )
