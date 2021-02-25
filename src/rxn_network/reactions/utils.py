import numpy as np
from pymatgen import Element

def check_metathesis_like(rxn, target_comp, metathesis_elems):
    metathesis_elems = [Element(e) for e in metathesis_elems.split("-")]
    products = rxn.products.copy()
    products.remove(target_comp)
    is_metathesis = np.array([set(comp.elements).issubset(metathesis_elems)
                              for comp in products])
    found = False
    if is_metathesis.all():
        found = True
    return found