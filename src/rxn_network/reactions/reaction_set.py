from typing import List, Dict, Optional
from functools import lru_cache
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element

from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    """

    def __init__(self, entries, all_indices, all_coeffs, all_data=None):
        """

        Args:
            entries:
            all_indices:
            all_coeffs:
            all_data:
        """
        self.entries = entries
        self.all_indices = all_indices
        self.all_coeffs = all_coeffs
        if not all_data:
            all_data = []
        self.all_data = all_data

    @lru_cache(1)
    def get_rxns(self, open_elem=None, chempot=0):
        """

        Args:
            open_elem:
            chempot:

        Returns:

        """
        rxns = []
        chempots=None
        if open_elem:
            chempots = {Element(open_elem): chempot}
        for indices, coeffs, data in zip(
            self.all_indices, self.all_coeffs, self.all_data
        ):
            entries = [self.entries[i] for i in indices]
            if chempots:
                rxns.append(
                    OpenComputedReaction(
                        entries=entries,
                        coefficients=coeffs,
                        data=data,
                        chempots=chempots,
                    )
                )
            else:
                rxns.append(
                    ComputedReaction(entries=entries, coefficients=coeffs, data=data)
                )
        return rxns

    def calculate_costs(self, cf):
        """

        Args:
            cf:

        Returns:

        """
        return [cf.evaluate(rxn) for rxn in self.rxns]

    @classmethod
    def from_rxns(cls, rxns, entries=None):
        """

        Args:
            rxns:
            entries:

        Returns:

        """
        if not entries:
            entries = cls._get_unique_entries(rxns)

        entries = sorted(list(set(entries)), key=lambda r: r.composition)
        n = len(entries)
        all_indices, all_coeffs, all_data = [], [], []
        for rxn in rxns:
            all_indices.append([entries.index(e) for e in rxn.entries])
            all_coeffs.append(list(rxn.coefficients))
            all_data.append(rxn.data)

        return cls(
            entries=entries,
            all_indices=all_indices,
            all_coeffs=all_coeffs,
            all_data=all_data,
        )

    @staticmethod
    def _get_unique_entries(rxns):
        " Return only unique entries from reactions"
        entries = set()
        for r in rxns:
            entries.update(r.entries)
        return entries
