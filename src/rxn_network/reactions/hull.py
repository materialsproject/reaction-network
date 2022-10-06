"""Code for analyzing sets of reactions between two phases."""

from functools import cached_property, lru_cache
from itertools import combinations
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from plotly.express import line, scatter
from plotly.graph_objs import Figure
from pymatgen.core.units import kb
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
            if coordinate > x1 and coordinate < x2:
                return {
                    self.reactions[v1]: (x2 - coordinate) / (x2 - x1),
                    self.reactions[v2]: (coordinate - x1) / (x2 - x1),
                }

            continue

        raise ValueError("No reactions found!")

    def get_primary_selectivity(self, reaction: ComputedReaction, temp=300):
        """
        Calculates the competition score (c-score) for a given reaction. This formula is
        based on a methodology presented in the following paper: (TBD)

        Args:
            reaction: Reaction to calculate the competition score for.

        Returns:
            The c-score for the reaction
        """
        energy = reaction.energy_per_atom
        idx = self.reactions.index(reaction)
        competing_rxns = self.reactions[:idx] + self.reactions[idx + 1 :]
        primary = self._primary_selectivity_from_energies(
            energy, [r.energy_per_atom for r in competing_rxns], temp
        )

        return primary

    def get_secondary_selectivity(
        self, reaction: ComputedReaction, normalize=True, recursive=False
    ):
        """
        Calculates the score for a given reaction. This formula is based on a
        methodology presented in the following paper: (TBD)

        Args:
            reaction: Reaction to calculate the selectivity score for.

        Returns:
            The selectivity score for the reaction
        """
        x = self.get_coordinate(reaction)
        if recursive:
            (
                left_energy,
                left_num_paths,
            ) = self.get_decomposition_energy_and_num_paths_recursive(0, x)
            (
                right_energy,
                right_num_paths,
            ) = self.get_decomposition_energy_and_num_paths_recursive(x, 1)
        else:
            left_energy = self.get_decomposition_energy(0, x)
            left_num_paths = self.count(len(self.get_coords_in_range(0, x)) - 2)
            right_energy = self.get_decomposition_energy(x, 1)
            right_num_paths = self.count(len(self.get_coords_in_range(x, 1)) - 2)

        if left_num_paths == 0:
            left_num_paths = 1
        if right_num_paths == 0:
            right_num_paths = 1

        energy = left_energy * right_num_paths + right_energy * left_num_paths
        total = left_num_paths * right_num_paths

        if normalize:
            energy = energy / total

        e_above_hull = self.get_energy_above_hull(reaction)
        energy -= e_above_hull

        return -1 * energy

    def get_decomposition_energy(self, x1: float, x2: float):
        """
        Calculates the energy of the reaction decomposition between two points.

        Args:
            x1: Coordinate of first point.
            x2: Coordinate of second point.

        Returns:
            The energy of the reaction decomposition between the two points.
        """
        coords = self.get_coords_in_range(x1, x2)
        n = len(coords) - 2

        energy = 0
        for c in combinations(range(len(coords)), 3):
            i_left, i_mid, i_right = sorted(c)

            c_left, c_mid, c_right = coords[[i_left, i_mid, i_right]]

            n_left = (i_mid - i_left) - 1
            n_right = (i_right - i_mid) - 1

            count = self._altitude_multiplicity(n_left, n_right, n)
            energy += count * self._calculate_altitude(c_left, c_mid, c_right)

        return energy

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

    def count(self, num):
        """
        Reurns the number of decomposition pathways for the interface reaction hull
        based on the number of **product** vertices (i.e., total # of vertices
        considered - 2 reactant vertices).

        Precomputed for hulls up to size 15. Otherwise, calls recursive implementation
        of counting function.

        Args:
            num: Number of product vertices.
        """
        counts = [
            1,
            1,
            2,
            5,
            14,
            42,
            132,
            429,
            1430,
            4862,
            16796,
            58786,
            208012,
            742900,
            2674440,
        ]
        if num < 15:
            count = counts[num]
        else:
            count = self._count_recursive(num)[0]

        return count

    @lru_cache(maxsize=128)
    def get_decomposition_energy_and_num_paths_recursive(
        self, x1: float, x2: float, use_x_min_ref=True, use_x_max_ref=True
    ) -> Tuple[float, int]:
        """
        This is a recursive implementation of the get_decomposition_energy function. It
        significantly slower than the non-recursive implementation but is more
        straightforward to understand. Both should return the same answer, however the
        refcursive implementation also includes "free" computation of the total number
        of paths. The function has been cached for speed.

        Args:
            x1: Coordinate of first point.
            x2: Coordinate of second point.
            use_x_min_ref: Useful for recursive calls. If true, uses the reactant at x=0
                as the reference (sometimes there is a decomposition reaction of the reactant that is
                lower in energy than the reactant).
            use_x_max_ref: Useful for recursive calls. If true, uses the reactant at x=1.0
                as the reference (sometimes there is a decomposition reaction of the reactant that is
                lower in energy than the reactant).

        Returns:
            Tuple of decomposition energy and the number of decomposition pathways.
        """
        all_coords = self.get_coords_in_range(x1, x2)

        if not use_x_min_ref and all_coords[1, 0] == 0.0:
            all_coords = all_coords[1:]
        if not use_x_max_ref and all_coords[-1, 0] == 1.0:
            all_coords = all_coords[:-1]

        x_min, y_min = all_coords[0]
        x_max, y_max = all_coords[-1]

        coords = all_coords[1:-1, :]

        if len(coords) == 0:
            val, total = 0, 1
            return val, total
        elif len(coords) == 1:
            val = self._calculate_altitude([x_min, y_min], coords[0], [x_max, y_max])
            total = 1
            return val, total
        else:
            val = 0
            total = 0
            for c in coords:
                if c[0] == 0.0 and not np.isclose(c[1], 0.0):
                    use_x_min_ref = False
                elif c[0] == 1.0 and not np.isclose(c[1], 0.0):
                    use_x_max_ref = False

                height = self._calculate_altitude([x_min, y_min], c, [x_max, y_max])
                (
                    left_decomp,
                    left_total,
                ) = self.get_decomposition_energy_and_num_paths_recursive(
                    x_min, c[0], use_x_min_ref, use_x_max_ref
                )
                (
                    right_decomp,
                    right_total,
                ) = self.get_decomposition_energy_and_num_paths_recursive(
                    c[0], x_max, use_x_min_ref, use_x_max_ref
                )

                val += (
                    height * (left_total * right_total)
                    + left_decomp * right_total
                    + right_decomp * left_total
                )
                total += left_total * right_total

        return val, total

    @lru_cache(maxsize=128)
    def _altitude_multiplicity(self, n_left, n_right, n):
        """
        This function is used in the non-recursive implementation of the
        get_decomposition_energy function. It allows for rapid computation of the number
        times (multiplicitly) that a particular altitude appears in the decomposition.

        Args:
            n_left: number of vertices occurring in between the leftmost and middle vertices.
            n_right: number of vertices occurring in between the middle and rightmost
            vertices.
            n: total number of product vertices in the full interface reaction hull
                (i.e., not including the 2 reactant reference vertices )
        """
        remainder = n - n_left - n_right - 1
        if remainder < 0:
            return 0

        return self.count(n_left) * self.count(n_right) * self.count(remainder)

    def _count_recursive(self, n, cache=None):
        """
        A recursive implementation of the counting function.

        This implementation is courtesy of @mcgalcode.
        """
        if cache is None:
            cache = []
        if n == 0:
            return 1, [1]
        elif n == 1:
            return 1, [1, 1]
        elif len(cache) >= n:
            return cache[n], cache
        else:
            total = 0
            biggest_cache = []
            for i in range(n):
                left = i
                right = n - i - 1
                left_divs, c1 = self._count_recursive(left, biggest_cache)

                right_divs, c2 = self._count_recursive(right, biggest_cache)

                if len(c1) > len(biggest_cache):
                    biggest_cache = c1

                if len(c2) > len(biggest_cache):
                    biggest_cache = c2

                total += left_divs * right_divs
            return total, biggest_cache + [total]

    def _get_scatter(self):
        marker_size = 10

        pts = scatter(
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

        lines = [line(x=c[:, 0], y=c[:, 1]) for c in coords if not (c[:, 1] == 0).all()]

        line_data = []
        for l in lines:
            l.update_traces(line={"color": "black"})
            line_data.append(l.data[0])

        return line_data

    @cached_property
    def hull_vertices(self):
        hull_vertices = [
            i
            for i in self.hull.vertices  # pylint: disable=not-an-iterable
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
        Returns the reactions that are stable (on the convex hull) of the interface reaction hull.
        """
        return [r for i, r in enumerate(self.reactions) if i in self.hull_vertices]

    @cached_property
    def unstable_reactions(self):
        """
        Returns the reactions that are unstable (NOT on the convex hull) of the interface reaction hull.
        """
        return [r for i, r in enumerate(self.reactions) if i not in self.hull_vertices]

    @staticmethod
    def _calculate_altitude(c_left, c_mid, c_right):
        """
        Helper geometry method: calculates the altitude of a point on a line defined by three points.

        Args:
            x1: point 1
            x2: point 2
            x3: point 3

        Returns:
            The altitude of the point
        """
        x1, y1 = c_left
        x2, y2 = c_mid
        x3, y3 = c_right

        xd = (x2 - x1) / (x3 - x1)
        yd = y1 + xd * (y3 - y1)

        return y2 - yd

    @staticmethod
    def _primary_selectivity_from_energies(rxn_energy, other_rxn_energies, temp):
        """
        Calculates the primary selectivity given a list of reaction energy differences
        """
        all_rxn_energies = np.append(other_rxn_energies, rxn_energy)
        Q = np.sum(np.exp(-all_rxn_energies / (kb * temp)))

        probability = np.exp(-rxn_energy / (kb * temp)) / Q

        return -np.log(probability)
