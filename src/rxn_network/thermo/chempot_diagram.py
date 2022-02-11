"""
This module implements added features to the ChemicalPotentialDiagram class from
pymatgen.
"""

from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram as ChempotDiagram
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Element
from scipy.spatial import HalfspaceIntersection, KDTree

from rxn_network.entries.entry_set import GibbsEntrySet


class ChemicalPotentialDiagram(ChempotDiagram):
    """
    This class is an extension of the ChemicalPotentialDiagram class from pymatgen.
    Several features have been added to the original class for the purpose of efficiently
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
                HalfspaceIntersection algorithm; these constrain the space over which the
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
        self._hs_int = self._get_halfspace_intersection()

        num_hyperplanes = len(self._hyperplanes)
        num_border_hyperplanes = len(self._border_hyperplanes)
        self._border_hyperplane_indices = list(
            range(num_hyperplanes, num_hyperplanes + num_border_hyperplanes)
        )

    def shortest_domain_distance(self, f1: str, f2: str) -> float:
        """
        Args:
            f1: chemical formula (1)
            f2: chemical formula (2)

        Returns:
            Shortest distance between domain boundaries in the full
            (hyper)dimensional space, calculated using KDTree.
        """

        if f1 in self.domains:
            pts1 = self.domains[f1]
        elif f1 in self.metastable_domains:
            pts1 = self.metastable_domains[f1]
        else:
            raise ValueError(f"Formula {f1} not in domains!")

        if f2 in self.domains:
            pts2 = self.domains[f2]
        elif f2 in self.metastable_domains:
            pts2 = self.metastable_domains[f2]
        else:
            raise ValueError(f"Formula {f2} not in domains!")

        tree = KDTree(pts1)

        return min(tree.query(pts2)[0])

    def add_entry(self, entry):
        """
        Adds an entry to existing chemical potential diagram. Uses scipy's
        add_halfspaces() method to save some time/memory.

        Args:
            entry:

        Returns:
            None

        """
        if entry in self.entries:
            return
        if not set(entry.composition.elements).issubset(self.elements):
            raise ValueError("New entry is not within the same chemical system!")

        hyperplane = self._get_hyperplane(entry)
        self._hyperplanes = np.vstack([self._hyperplanes, hyperplane])
        self._hyperplane_entries.append(entry)
        self._hs_int.add_halfspaces([hyperplane], restart=True)
        self._entry_dict[entry.composition.reduced_formula] = entry
        self.entries.append(entry)

        try:
            del self.domains  # clear cache
        except AttributeError:
            pass

    def _get_halfspace_intersection(self):
        hs_hyperplanes = np.vstack([self._hyperplanes, self._border_hyperplanes])
        interior_point = np.min(self.lims, axis=1) + 1e-1
        return HalfspaceIntersection(hs_hyperplanes, interior_point, incremental=True)

    def _get_domains(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of domains as {formula: np.ndarray}"""
        domains = {entry.composition.reduced_formula: [] for entry in self._hyperplane_entries}  # type: ignore
        entries = self._hyperplane_entries

        for intersection, facet in zip(
            self.hs_int.intersections, self.hs_int.dual_facets
        ):
            for v in facet:
                if v not in self._border_hyperplane_indices:
                    if v > max(self._border_hyperplane_indices):
                        v = v - len(self._border_hyperplane_indices)
                    this_entry = entries[v]
                    formula = this_entry.composition.reduced_formula
                    domains[formula].append(intersection)

        return {k: np.array(v) for k, v in domains.items() if v}

    def _get_hyperplanes_and_entries(self) -> Tuple[np.ndarray, List[PDEntry]]:
        """Returns both the array of hyperplanes, as well as a list of the minimum
        entries"""
        data = np.array([self._get_hyperplane(e) for e in self._min_entries])
        vec = [self.el_refs[el].energy_per_atom for el in self.elements] + [1]
        form_e = -np.dot(data, vec)

        inds = np.where(form_e < -PhaseDiagram.formation_energy_tol)[0].tolist()

        inds.extend([self._min_entries.index(el) for el in self.el_refs.values()])

        hyperplanes = data[inds]
        hyperplane_entries = [self._min_entries[i] for i in inds]

        return hyperplanes, hyperplane_entries

    def _get_hyperplane(self, entry):
        data = np.array(
            [entry.composition.get_atomic_fraction(el) for el in self.elements]
            + [-entry.energy_per_atom]
        )
        return data

    @property
    def hs_int(self):
        """Returns the scipy HalfSpaceIntersection object"""
        return self._hs_int

    @cached_property
    def domains(self) -> Dict[str, np.ndarray]:
        """
        Mapping of formulas to array of domain boundary points. Cached for speed.
        """
        return self._get_domains()

    @cached_property
    def metastable_domains(self) -> Dict[str, np.ndarray]:
        """Gets a dictionary of the chemical potential domains for metastable chemical
        formulas. This corresponds to the domains of the relevant phases if they were
        just barely stable"""
        return self._get_metastable_domains()

    def _get_metastable_domains(self):
        e_set = GibbsEntrySet(self.entries)
        e_dict = e_set.min_entries_by_formula
        stable_formulas = list(self.domains.keys())

        metastable_entries = [e for f, e in e_dict.items() if f not in stable_formulas]
        metastable_domains = {}

        for e in metastable_entries:
            formula = e.composition.reduced_formula
            new_entry = e_set.get_stabilized_entry(e)
            cpd = ChemicalPotentialDiagram(list(e_dict.values()) + [new_entry])
            metastable_domains[formula] = cpd.domains[formula]

        return metastable_domains
