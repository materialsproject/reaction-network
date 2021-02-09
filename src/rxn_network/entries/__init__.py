" Specialized Entry types for the RXN Network "
import hashlib

from pymatgen.analysis.phase_diagram import PDEntry


def _new_pdentry_hash(self):
    data_md5 = hashlib.md5(
        f"{self.composition.formula}_" f"{self.energy}".encode("utf-8")
    ).hexdigest()
    return int(data_md5, 16)


# necessary fix, will be updated in pymatgen in future
PDEntry.__hash__ = _new_pdentry_hash

from rxn_network.entries.gibbs import (  # pylint: disable=C0413
    GibbsComputedEntry,  # noqa: E402
)
from rxn_network.entries.nist import (  # pylint: disable=C0413
    NISTReferenceEntry,  # noqa: E402
)
