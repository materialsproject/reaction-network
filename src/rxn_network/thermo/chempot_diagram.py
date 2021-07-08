" A chemical potential diagram class"

from functools import cached_property
from typing import List, Dict, Optional
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull, KDTree
from scipy.spatial.qhull import QhullError
from monty.json import MSONable

from rxn_network.thermo.chempot_layouts import (
    default_chempot_layout_3d,
    default_chempot_annotation_layout,
)
from rxn_network.thermo.utils import simple_pca, get_centroid_2d

import plotly.express as px
import plotly.graph_objects as go

from pymatgen.core.composition import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter


class ChempotDiagram(MSONable):
    """
    The chemical potential diagram is the mathematical dual (intensive analog) to the
    traditional compositional phase diagram. To create the diagram, convex
    minimization is performed in E vs. μ space by taking the lower convex envelope of
    hyperplanes. Accordingly, "points" on the standard phase diagram become
    "domains" in chemical potential space.
    """

    def __init__(
        self,
        pd: PhaseDiagram,
        limits: Optional[Dict[Element, float]] = None,
        default_limit: Optional[float] = -15.0,
    ):
        """
        Args:
            pd: Phase diagram object (pymatgen).
            limits: Chemical potential value chosen for bordering elemental
                hyperplanes; constrains the space over which the domains are calculated
            default_limit (float): Default minimum limit for unspecified elements.
        """
        self.pd = pd
        self.limits = limits
        self.default_limit = default_limit

        self.dim = pd.dim

    @cached_property
    def domains(self) -> Dict[str, np.array]:
        """
        Mapping of formulas to array of domain boundary points
        """
        lims = np.array([[self.default_limit, 0]] * self.dim)
        for idx, elem in enumerate(self.pd.elements):
            if self.limits and elem in self.limits:
                lims[idx, :] = self.limits[elem]

        data = self.pd.qhull_data
        hyperplanes = np.insert(
            data, [0], (1 - np.sum(data[:, :-1], axis=1)).reshape(-1, 1), axis=1
        )
        hyperplanes[:, -1] = hyperplanes[:, -1] * -1  # flip to all positive energies
        entries = self.pd.qhull_entries

        border_hyperplanes = np.array(([[0] * (self.dim + 1)] * (2 * self.dim)))

        for idx, limit in enumerate(lims):
            border_hyperplanes[2 * idx, idx] = -1
            border_hyperplanes[2 * idx, -1] = limit[0]
            border_hyperplanes[(2 * idx) + 1, idx] = 1
            border_hyperplanes[(2 * idx) + 1, -1] = limit[1]

        hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])

        interior_point = np.average(lims, axis=1).tolist()
        hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

        domains = {entry.composition.reduced_formula: [] for entry in self.entries}

        for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
            for v in facet:
                if v < len(entries):
                    this_entry = entries[v]
                    formula = this_entry.composition.reduced_formula
                    domains[formula].append(intersection)

        return {k: np.array(v) for k, v in domains.items() if v}

    def get_plot(
        self,
        elements: list = None,
        formulas: list = [],
        formula_mode="mesh",
        formula_colors=None,
        label_stable=True,
        shade_energy=True,
    ) -> go.Scatter3d:
        """

        Args:
            elements:
            formulas:
            formula_mode:
            formula_colors:
            label_stable:
            shade_energy:

        Returns:

        """

        if not formula_colors:
            formula_colors = px.colors.qualitative.Dark2

        if not elements:
            elements = self.pd.elements[:3]
        else:
            elements = [Element(e) for e in elements]

        elem_indices = [self.pd.elements.index(e) for e in elements]

        domain_vertices = {}
        annotations = []
        font_dict = {"color": "black", "size": 16.0}
        opacity = 0.7

        extra_domains = {}
        domains = self.domains.copy()
        comps_reduced = [
            Composition(formula).reduced_composition for formula in formulas
        ]

        for formula, points in domains.items():
            entry = self.entry_dict[formula]
            points_3d = np.array(points[:, elem_indices])
            contains_target_elems = set(entry.composition.elements).issubset(elements)

            if formulas:
                if entry.composition.reduced_composition in comps_reduced:
                    domains[formula] = None
                    extra_domains[formula] = points_3d

                    if contains_target_elems:
                        domains[formula] = points_3d
                    else:
                        continue

            if not contains_target_elems:
                domains[formula] = None
                continue

            # Try to get all convex polyhedra (3d), default to polygons (2d)
            try:
                domain = ConvexHull(points_3d)
                ann_loc = np.mean(points_3d.T, axis=1)
            except QhullError as e:
                points_2d, v, w = simple_pca(points_3d, k=2)
                domain = ConvexHull(points_2d)
                centroid_2d = get_centroid_2d(points_2d[domain.vertices])
                ann_loc = centroid_2d @ w.T + np.mean(
                    points_3d.T, axis=1
                )  # recover orig 3D coords from eigenvectors

            simplices = [Simplex(points_3d[indices]) for indices in domain.simplices]

            formula_disp = formula
            if hasattr(entry, "original_entry"):
                formula_disp = entry.original_entry.composition.reduced_formula

            clean_formula = PDPlotter._htmlize_formula(formula_disp)
            annotation = default_chempot_annotation_layout.copy()

            annotation.update(
                {
                    "x": ann_loc[0],
                    "y": ann_loc[1],
                    "z": ann_loc[2],
                    "font": font_dict,
                    "text": clean_formula,
                    "opacity": opacity,
                }
            )
            annotations.append(annotation)
            domains[formula] = simplices
            domain_vertices[formula] = points_3d

        x, y, z = [], [], []
        meshes = []
        cmax = 0.0001
        cmin = (
            min([self.pd.get_form_energy_per_atom(e) for e in self.pd.stable_entries])
            - 0.0001
        )

        for phase, simplexes in domains.items():
            if simplexes:
                for s in simplexes:
                    x.extend(s.coords[:, 0].tolist() + [None])
                    y.extend(s.coords[:, 1].tolist() + [None])
                    z.extend(s.coords[:, 2].tolist() + [None])

        layout = default_chempot_layout_3d.copy()
        layout["scene"].update(
            {
                "xaxis": self._get_chempot_axis_layout(elements[0]),
                "yaxis": self._get_chempot_axis_layout(elements[1]),
                "zaxis": self._get_chempot_axis_layout(elements[2]),
            }
        )
        layout["scene"]["annotations"] = None
        if label_stable:
            layout["scene"].update({"annotations": annotations})

        lines = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="black", width=4.5),
                showlegend=False,
            )
        ]

        extra_phases = []

        for idx, formula in enumerate(formulas):
            try:
                coords = extra_domains[formula]
            except KeyError:
                continue

            points_3d = coords[:, :3]
            entry = self.entry_dict[formula]
            if "mesh" in formula_mode:
                extra_phases.append(
                    go.Mesh3d(
                        x=points_3d[:, 0],
                        y=points_3d[:, 1],
                        z=points_3d[:, 2],
                        alphahull=0,
                        showlegend=True,
                        lighting=dict(fresnel=1.0),
                        color=formula_colors[idx],
                        name=f"{entry.composition.reduced_formula} (mesh)",
                        opacity=0.13,
                    )
                )
            if "lines" in formula_mode:
                points_2d = points_3d[:, 0:2]
                domain = ConvexHull(points_2d)
                simplexes = [
                    Simplex(points_3d[indices]) for indices in domain.simplices
                ]
                x, y, z = [], [], []
                for s in simplexes:
                    x.extend(s.coords[:, 0].tolist() + [None])
                    y.extend(s.coords[:, 1].tolist() + [None])
                    z.extend(s.coords[:, 2].tolist() + [None])

                extra_phases.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        line={"width": 8, "color": formula_colors[idx]},
                        opacity=1.0,
                        name=f"{formula} (lines)",
                    )
                )

        layout["scene_camera"] = dict(
            eye=dict(x=0, y=0, z=2.0), projection=dict(type="orthographic")
        )
        fig = go.Figure(meshes + lines + extra_phases, layout)
        fig.update_layout(
            coloraxis={
                "colorscale": "blugrn",
                "cmin": cmin,
                "cmax": cmax,
                "showscale": True,
            }
        )

        return fig

    @property
    def entries(self) -> List[ComputedEntry]:
        " Returns mu diagram entries (i.e., stable entries of phase diagram) "
        return self.pd.stable_entries

    @cached_property
    def entry_dict(self) -> Dict[str, ComputedEntry]:
        """ Mapping between reduced formula and ComputedEntry """
        return {e.composition.reduced_formula: e for e in self.entries}

    def shortest_domain_distance(self, f1: str, f2: str) -> float:
        """
        Args:
            f1: chemical formula (1)
            f2: chemical formula (2)

        Returns:
            Shortest distance between domain boundaries in the full
            (hyper)dimensional space, calculated using KDTree.
        """
        pts1 = self.domains[Composition(f1).reduced_formula]
        pts2 = self.domains[Composition(f2).reduced_formula]

        tree = KDTree(pts1)

        return min(tree.query(pts2)[0])

    def shortest_elemental_domain_distances(self, f1, f2) -> float:
        """
        TODO: Use with caution; this function may not yet make sense geometrically!
        """
        pts1 = self.domains[f1]
        pts2 = self.domains[f2]
        pts1 = pts1[~np.isclose(pts1, self.default_limit).any(axis=1)]
        pts2 = pts2[~np.isclose(pts2, self.default_limit).any(axis=1)]
        num_elems = pts1.shape[1]

        mesh = np.meshgrid(pts1, pts2)
        diff = abs(mesh[0] - mesh[1])
        diff = diff.reshape(-1, num_elems)

        return diff.min(axis=0)

    @property
    def chemical_system(self) -> str:
        " Returns the chemical system (A-B-C-...) of diagram object "
        return "-".join(sorted([e.symbol for e in self.pd.elements]))

    @staticmethod
    def _get_chempot_axis_layout(element):
        return dict(
            title=f"μ<sub>{str(element)}</sub> - μ<sub>"
            f"{str(element)}</sub><sup>o</sup> (eV)",
            titlefont={"size": 30},
            gridcolor="#dbdbdb",
            gridwidth=5.0,
            tickfont={"size": 16},
            ticks="inside",
            ticklen=14,
            showline=True,
            backgroundcolor="rgba(0,0,0,0)",
        )

    def __repr__(self):
        return f"ChempotMap for {self.chemical_system} with {len(self.entries)} entries"
