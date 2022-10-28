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
    Several features have been added to the original class for the purpose of
    efficiently calculating the shortest distance between two chemical potential
    domains.

    For more information on this specific implementation of the algorithm,
    please cite/reference the paper below:

        Todd, P. K., McDermott, M. J., Rom, C. L., Corrao, A. A., Denney, J. J.,
        Dwaraknath, S. S.,  Khalifah, P. G., Persson, K. A., & Neilson, J. R. (2021).
        Selectivity in Yttrium Manganese Oxide Synthesis via Local Chemical Potentials
        in Hyperdimensional Phase Space. Journal of the American Chemical Society,
        143(37), 15185-15194. https://doi.org/10.1021/jacs.1c06229
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        entries: List[PDEntry],
        limits: Optional[Dict[Element, float]] = None,
        default_min_limit: Optional[float] = -100.0,
    ):
        """
        Initialize a ChemicalPotentialDiagram object.

        Args:
            entries: List of PDEntry-like objects containing a composition and
                energy. Must contain elemental references and be suitable for typical
                phase diagram construction. Entries must be within a chemical system of
                with 2+ elements
            limits: Bounds of elemental chemical potentials (min, max), which are
                used to construct the border hyperplanes used in the
                HalfspaceIntersection algorithm; these constrain the space over which
                the domains are calculated and also determine the size of the plotted
                diagram. Any elemental limits not specified are covered in the
                default_min_limit argument
            default_min_limit (float): Default minimum chemical potential limit for
                unspecified elements within the "limits" argument. This results in
                default limits of (-100, 0)
        """
        self.entries = list(
            sorted(entries, key=lambda e: e.composition.reduced_formula)
        )
        self._entry_set = GibbsEntrySet(self.entries)
        self.limits = limits
        self.default_min_limit = default_min_limit
        self.elements = list(
            sorted({els for e in self.entries for els in e.composition.elements})
        )
        self.dim = len(self.elements)
        self._min_entries, self._el_refs = self._get_min_entries_and_el_refs(
            self.entries
        )
        self._entry_dict = {e.composition.reduced_formula: e for e in self._min_entries}
        self._border_hyperplanes = self._get_border_hyperplanes()
        (
            self._hyperplanes,
            self._hyperplane_entries,
        ) = self._get_hyperplanes_and_entries()

        if self.dim < 2:
            raise ValueError(
                "ChemicalPotentialDiagram currently requires phase "
                "diagrams with 2 or more elements!"
            )

        if len(self.el_refs) != self.dim:
            missing = set(self.elements).difference(self.el_refs.keys())
            raise ValueError(
                f"There are no entries for the terminal elements: {missing}"
            )
        self._hs_int = self._get_halfspace_intersection()

        num_hyperplanes = len(self._hyperplanes)
        num_border_hyperplanes = len(self._border_hyperplanes)

        self._border_hyperplane_indices = list(
            range(num_hyperplanes, num_hyperplanes + num_border_hyperplanes)
        )
        self._metastable_domains: Dict[str, List] = {}  # for caching

    def shortest_domain_distance(self, f1: str, f2: str, offset=0.0) -> float:
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
        else:
            pts1 = self._get_metastable_domain(f1)

        if f2 in self.domains:
            pts2 = self.domains[f2]
        else:
            pts2 = self._get_metastable_domain(f2)

        tree = KDTree(pts1)

        return min(tree.query(pts2)[0]) + offset

    def _get_halfspace_intersection(self):
        hs_hyperplanes = np.vstack([self._hyperplanes, self._border_hyperplanes])
        interior_point = np.min(self.lims, axis=1) + 1e-1
        return HalfspaceIntersection(hs_hyperplanes, interior_point)

    def _get_domains(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of domains as {formula: np.ndarray}"""
        domains: Dict[str, List] = {
            entry.composition.reduced_formula: [] for entry in self._hyperplane_entries
        }
        entries = self._hyperplane_entries

        for intersection, facet in zip(
            self.hs_int.intersections, self.hs_int.dual_facets
        ):
            for v in facet:
                if v not in self._border_hyperplane_indices:
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

    def get_offset(self, entry):
        """ """
        if (
            entry in self._min_entries
            and entry.composition.reduced_formula in self.domains
        ):
            offset = 0.0
        else:
            e_above_hull = self._entry_set.get_e_above_hull(entry)
            hyperplane = self._get_hyperplane(entry)
            offset = self._get_distance_between_parallel_hyperplanes(
                hyperplane[:-1], e_above_hull
            )

        return offset

    @staticmethod
    def _get_distance_between_parallel_hyperplanes(a, delta_b):
        """Returns the distance between two parallel hyperplanes"""
        return np.abs(delta_b) / np.linalg.norm(a)

    def _get_metastable_domain(self, formula):
        """Returns the metastable domain for a given formula"""
        if formula in self._metastable_domains:
            return self._metastable_domains[formula]

        orig_entry = self._entry_set.get_min_entry_by_formula(formula)
        new_entry = self._entry_set.get_stabilized_entry(orig_entry, tol=1e-1)
        self._entry_set.add(new_entry)
        cpd = ChemicalPotentialDiagram(self._entry_set, default_min_limit=-500)

        try:
            metastable_domain = cpd.domains[formula]
        except KeyError as exc:
            raise ValueError(
                f"Metastable domain for {formula} not found! Please investigate."
            ) from exc

        self._metastable_domains[formula] = metastable_domain
        self._entry_set.remove(new_entry)

        return metastable_domain

    @property
    def hs_int(self):
        """Returns the scipy HalfSpaceIntersection object"""
        return self._hs_int

    @cached_property
    def domains(self) -> Dict[str, np.ndarray]:
        """Mapping of formulas to array of domain boundary points"""
        return self._get_domains()

    @property
    def metastable_domains(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of the chemical potential domains for metastable chemical
        formulas. This corresponds to the domains of the relevant phases if they were
        just barely stable
        """
        return {
            e.composition.reduced_formula: self._get_metastable_domain(
                e.composition.reduced_formula
            )
            for e in self._min_entries
            if e.composition.reduced_formula not in self.domains
        }
