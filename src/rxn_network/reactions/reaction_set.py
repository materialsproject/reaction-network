from typing import List, Dict, Optional
from functools import cached_property
import numpy as np
from monty.json import MSONable

from rxn_network.reactions.computed import ComputedReaction


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    """
    def __init__(self, entries, all_indices, all_coeffs, all_data=None):
        self.entries = entries
        self.all_indices = all_indices
        self.all_coeffs = all_coeffs
        if not all_data:
            all_data = []
        self.all_data = all_data

    @cached_property
    def rxns(self):
        rxns = []
        for indices, coeffs in zip(self.all_indices, self.all_coeffs):
            entries = [self.entries[i] for i in indices]
            rxns.append(ComputedReaction(entries=entries, coefficients=coeffs))
        return rxns

    def calculate_costs(self, cf):
        return [cf.evaluate(rxn) for rxn in self.rxns]

    @classmethod
    def from_rxns(cls, rxns, entries=None):
        if not entries:
            entries = cls._get_unique_entries(rxns)

        entries = sorted(list(set(entries)), key=lambda r: r.composition)
        n = len(entries)
        all_indices, all_coeffs = [], []
        for rxn in rxns:
            all_indices.append([entries.index(e) for e in rxn.entries])
            all_coeffs.append(list(rxn.coefficients))
            all_data.append(rxn.data)

        return cls(entries=entries, all_indices=all_indices, all_coeffs=all_coeffs,
                   all_data=all_data)

    @staticmethod
    def _get_unique_entries(rxns):
        entries = set()
        for r in rxns:
            entries.update(r.entries)
        return entries
