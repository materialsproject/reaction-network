from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from pymatgen.analysis.interface_reactions import InterfacialReactivity

from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import get_entry_combinations, get_total_chemsys, group_by_chemsys


class MinimizePotentialEnumerator(Enumerator):
    def __init__(self):
        pass
    def enumerate(self):
        pass


class MinimizeGrandPotentialEnumerator(Enumerator):
    def __init__(self):
        pass
    def enumerate(self):
        pass

