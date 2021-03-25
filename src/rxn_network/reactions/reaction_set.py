from typing import List, Dict, Optional
from functools import cached_property
import numpy as np
from monty.json import MSONable

from rxn_network.reactions.computed import ComputedReaction


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    """
    def __init__(self, entries, all_indices, all_coeffs):
        self.entries = entries
        self.all_indices = all_indices
        self.all_coeffs = all_coeffs

    @cached_property
    def rxns(self):
        rxns = []
        all_entries = np.array(self.entries)
        for vec in self.array:
            indices = vec.nonzero()
            coefficients = vec[indices]
            entries = all_entries[indices]
            rxns.append(ComputedReaction(entries=entries, coefficients=coefficients))

        return rxns

    @classmethod
    def from_rxns(cls, rxns, entries=None):
        if not entries:
            entries = cls._get_unique_entries(rxns)

        entries = sorted(list(set(entries)), key=lambda r: r.composition)
        n = len(entries)
        arr = []
        for rxn in rxns:
            entry_indices = [entries.index(e) for e in rxn.entries]
            arr.append(rxn.get_vector(entry_indices, n))
        arr = np.array(arr)

        return cls(array=arr, entries=entries)

    @staticmethod
    def _get_unique_entries(rxns):
        entries = set()
        for r in rxns:
            entries.update(r.entries)
        return entries
