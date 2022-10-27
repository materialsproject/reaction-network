"""Code for analyzing sets of reactions between two phases."""

from functools import cached_property
from typing import List

import numpy as np
import plotly.express as px
from monty.json import MSONable
from plotly.graph_objs import Figure
from scipy.spatial import ConvexHull

from rxn_network.core.composition import Composition
from rxn_network.reactions.computed import ComputedReaction


class InterfaceReactionHull(MSONable):
    """
    A class for storing and analyzing a set of reactions at an interface between two
    reactants. This class is more generalized than the InterfacialReactivity class and
    can encompass any set of reactions between two reactants, regardless of whether
    the reaction products are "stable" (i.e. together on the convex hull)
    """

    def __init__(
        self,
        c1: Composition,
        c2: Composition,
        reactions: List[ComputedReaction],
        max_num_constraints=1,
    ):
        """
        Args:
            c1: Composition of reactant 1
            c2: Composition of reactant 2
            reactions: List of reactions containing all enumerated reactions between the
                two reactants. Note that this list should
        """
        self.c1 = Composition(c1).reduced_composition
        self.c2 = Composition(c2).reduced_composition
        self.e1 = None
        self.e2 = None

        for rxn in reactions:
            for e in rxn.reactant_entries:
                if e.composition.reduced_composition == self.c1:
                    self.e1 = e
                elif e.composition.reduced_composition == self.c2:
                    self.e2 = e
            if self.e1 is not None and self.e2 is not None:
                break
        else:
            print(c1, c2)
            print(reactions)
            raise ValueError(
                "Provided reactions do not correspond to reactant compositons!"
            )

        reactions = [
            r
            for r in reactions
            if r.data.get("num_constraints", 1) <= max_num_constraints
        ]

        endpoint_reactions = [
            ComputedReaction.balance([self.e1], [self.e1]),
            ComputedReaction.balance([self.e2], [self.e2]),
        ]

        reactions_with_endpoints = reactions + endpoint_reactions

        coords = np.array(
            [(self.get_coordinate(r), r.energy_per_atom) for r in reactions]
        )
        coords = np.append(coords, [[0, 0], [1, 0]], axis=0)

        idx_sort = coords[:, 0].argsort()

        self.coords = coords[idx_sort]
        self.reactions = [reactions_with_endpoints[i] for i in idx_sort]
        self.hull = ConvexHull(self.coords)
        self.endpoint_reactions = endpoint_reactions

    def plot(self, y_max=0.2):
        """
        Plot the reaction hull.
        """
        pts = self._get_scatter()
        lines = self._get_lines()

        fig = Figure(data=lines + [pts])

        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br> <br><b>Atomic fraction</b>:"
                " %{x:.3f}<br><b>Energy</b>: %{y:.3f} (eV/atom)"
            )
        )
        fig.update_layout(yaxis_range=[min(self.coords[:, 1]) - 0.01, y_max])
        fig.update_layout(xaxis_title="Mixing ratio")
        fig.update_layout(yaxis_title="Energy (eV/atom)")

        return fig

    def get_energy_above_hull(self, reaction):
        """
        Get the energy of a reaction above the reaction hull.
        """
        idx = self.reactions.index(reaction)
        x, y = self.coords[idx]
        e_above_hull = y - self.get_hull_energy(x)

        return e_above_hull

    def get_coordinate(self, reaction):
        """Get coordinate of reaction in reaction hull. This is expressed as the atomic
        mixing ratio of component 2 in the reaction."""
        amt_c1 = reaction.reactant_atomic_fractions.get(self.c1, 0)
        amt_c2 = reaction.reactant_atomic_fractions.get(self.c2, 0)
        total = amt_c1 + amt_c2  # will add to 1.0 with two-component reactions
        try:
            coordinate = amt_c2 / total
        except ZeroDivisionError as e:
            raise ValueError(
                f"Can't compute coordinate for {reaction} with {self.c1}, {self.c2}"
            ) from e

        return round(coordinate, 12)  # avoids numerical issues

    def get_hull_energy(self, coordinate):
        """
        Get the energy of the reaction at a given coordinate.

        Args:
            coordinate: Coordinate of reaction in reaction hull.

        Returns:
            Energy of reaction at given coordinate.
        """
        reactions = self.get_reactions_by_coordinate(coordinate)
        return sum(weight * r.energy_per_atom for r, weight in reactions.items())

    def get_reactions_by_coordinate(self, coordinate):
        """Get the reaction(s) at a given coordinate."""
        sorted_vertices = np.sort(self.hull_vertices)
        for i in range(len(sorted_vertices) - 1):
            v1 = sorted_vertices[i]
            v2 = sorted_vertices[i + 1]

            x1 = self.coords[v1, 0]
            x2 = self.coords[v2, 0]

            if np.isclose(coordinate, x1):
                return {self.reactions[v1]: 1.0}
            if np.isclose(coordinate, x2):
                return {self.reactions[v2]: 1.0}
            if x1 < coordinate < x2:
                return {
                    self.reactions[v1]: (x2 - coordinate) / (x2 - x1),
                    self.reactions[v2]: (coordinate - x1) / (x2 - x1),
                }

            continue

        raise ValueError("No reactions found!")

    def get_coords_in_range(self, x1, x2):
        """
        Get the coordinates in the range [x1, x2].

        Args:
            x1: Start of range.
            x2: End of range.

        Returns:
            Array of coordinates in the range.

        """
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = self.get_hull_energy(x_min), self.get_hull_energy(x_max)

        coords = []

        if np.isclose(x_min, 0.0) and not np.isclose(y_min, 0.0):
            coords.append([0.0, 0.0])

        coords.append([x_min, y_min])

        coords.extend(
            [
                self.coords[i]
                for i in self.hull_vertices
                if self.coords[i, 0] < x_max
                and self.coords[i, 0] > x_min
                and self.coords[i, 1] <= 0
            ]
        )

        if x_max != x_min:
            coords.append([x_max, y_max])
        if np.isclose(x_max, 1.0) and not np.isclose(y_max, 0.0):
            coords.append([1.0, 0.0])

        coords = np.array(coords)

        return coords[coords[:, 0].argsort()]

    def _get_scatter(self):
        marker_size = 10

        pts = px.scatter(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            hover_name=[str(r) for r in self.reactions],
            labels={
                "x": "Mixing Ratio",
                "y": (
                    r"$\Delta G_{\mathrm{rxn}} ~"
                    r" \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
                ),
            },
        )
        pts.update_traces(marker={"size": marker_size})

        return pts.data[0]

    def _get_lines(self):
        coords = self.coords[self.hull.simplices]

        coords = coords[(coords[:, :, 1] <= 0).all(axis=1)]
        coords = coords[~(coords[:, :, 1] == 0).all(axis=1)]

        lines = [
            px.line(x=c[:, 0], y=c[:, 1]) for c in coords if not (c[:, 1] == 0).all()
        ]

        line_data = []
        for line in lines:
            line.update_traces(line={"color": "black"})
            line_data.append(line.data[0])

        return line_data

    @cached_property
    def hull_vertices(self):
        hull_vertices = [
            i
            for i in self.hull.vertices
            if self.coords[i, 1] <= 0
            and np.isclose(
                self.coords[self.coords[:, 0] == self.coords[i, 0]][:, 1].min(),
                self.coords[i, 1],  # make sure point is lower than others on same x
            )
        ]

        return np.array(hull_vertices)

    @cached_property
    def stable_reactions(self):
        """
        Returns the reactions that are stable (on the convex hull) of the interface
        reaction hull.
        """
        return [r for i, r in enumerate(self.reactions) if i in self.hull_vertices]

    @cached_property
    def unstable_reactions(self):
        """
        Returns the reactions that are unstable (NOT on the convex hull) of the
        interface reaction hull.
        """
        return [r for i, r in enumerate(self.reactions) if i not in self.hull_vertices]
