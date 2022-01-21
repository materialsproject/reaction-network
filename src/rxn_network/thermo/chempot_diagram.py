"""
This module implements added features to the ChemicalPotentialDiagram class from
pymatgen.
"""
import warnings
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram as ChempotDiagram
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.composition import Element
from scipy.spatial import KDTree


class ChemicalPotentialDiagram(ChempotDiagram):
    """This class is an extension of the ChemicalPotentialDiagram class from pymatgen.
    Several features have been added to the original class for the purpose of
    calculating the shortest distance between two chemical potential domains.
    """

    def __init__(
        self,
        entries: List[PDEntry],
        limits: Optional[Dict[Element, float]] = None,
        default_min_limit: Optional[float] = -20.0,
    ):
        """
        Initialize a ChemicalPotentialDiagram object.

        Args:
            entries: List of PDEntry-like objects containing a composition and
                energy. Must contain elemental references and be suitable for typical
                phase diagram construction. Entries must be within a chemical system
                of with 2+ elements
            limits: Bounds of elemental chemical potentials (min, max), which are
                used to construct the border hyperplanes used in the
                HalfSpaceIntersection algorithm; these constrain the space over which the
                domains are calculated and also determine the size of the plotted
                diagram. Any elemental limits not specified are covered in the
                default_min_limit argument
            default_min_limit (float): Default minimum chemical potential limit for
                unspecified elements within the "limits" argument. This results in
                default limits of (default_min_limit, 0)
        """
        super().__init__(
            entries=entries, limits=limits, default_min_limit=default_min_limit
        )

    @cached_property
    def domains(self) -> Dict[str, np.ndarray]:
        """
        Mapping of formulas to array of domain boundary points. Cached for speed.
        """
        return self._get_domains()

    def shortest_domain_distance(self, f1: str, f2: str) -> float:
        """
        Args:
            f1: chemical formula (1)
            f2: chemical formula (2)

        Returns:
            Shortest distance between domain boundaries in the full
            (hyper)dimensional space, calculated using KDTree.
        """
        pts1 = self.domains[f1]
        pts2 = self.domains[f2]

        tree = KDTree(pts1)

        return min(tree.query(pts2)[0])

    def shortest_elemental_domain_distances(self, f1: str, f2: str) -> float:
        """
        Args:
            f1: chemical formula (1)
            f2: chemical formula (2)

        Returns:
            Shortest distance between domain boundaries along one elemental axis.
        """
        warnings.warn(
            "Use with caution; this function may not result in anything meaningful!"
        )

        pts1 = self.domains[f1]
        pts2 = self.domains[f2]
        pts1 = pts1[~np.isclose(pts1, self.default_min_limit).any(axis=1)]
        pts2 = pts2[~np.isclose(pts2, self.default_min_limit).any(axis=1)]
        num_elems = pts1.shape[1]

        mesh = np.meshgrid(pts1, pts2)
        diff = abs(mesh[0] - mesh[1])
        diff = diff.reshape(-1, num_elems)

        return diff.min(axis=0)
