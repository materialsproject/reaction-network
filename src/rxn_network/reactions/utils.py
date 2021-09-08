"""
Utility functions used in the reaction classes.
"""

import numpy as np
from pymatgen.core.composition import Element


def is_separable_rxn(rxn, target_comp, added_elems):
    """

    Args:
        rxn:
        target_comp:
        added_elems:

    Returns:

    """
    added_elems = [Element(e) for e in added_elems.split("-")]
    products = rxn.products.copy()
    products.remove(target_comp)
    separable = np.array(
        [set(comp.elements).issubset(added_elems) for comp in products]
    )
    found = False
    if separable.all():
        found = True
    return found
