"""Code for analyzing sets of reactions between two phases."""

from typing import List
from functools import cached_property, lru_cache
from itertools import combinations

import numpy as np
from monty.json import MSONable
from plotly.express import scatter, line
from plotly.graph_objs import Figure
from pymatgen.core.composition import Composition
from scipy.spatial import ConvexHull

from rxn_network.reactions.computed import ComputedReaction


class InterfaceReactionHull(MSONable):
    """
    A class for storing and analyzing a set of reactions at an interface between two
    reactants. This class is more generalized than the InterfacialReactivity class and
    can encompass any set of reactions between two reactants, regardless of whether they
    are on the convex hull.
    """

    def __init__(
        self, c1: Composition, c2: Composition, reactions: List[ComputedReaction]
    ):
        """
        Args:
            c1: Composition of reactant 1
            c2: Composition of reactant 2
            reactions: Reaction set containing known reactions between the two reactants
        """
        self.c1 = c1.reduced_composition
        self.c2 = c2.reduced_composition
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
            raise ValueError(
                "Provided reactions do not correspond to reactant compositons!"
            )

        endpoint_reactions = [
            ComputedReaction.balance([self.e1], [self.e1]),
            ComputedReaction.balance([self.e2], [self.e2]),
        ]

        reactions_with_endpoints = reactions + endpoint_reactions

        coords = np.array(
            [(self.get_x_coordinate(r), r.energy_per_atom) for r in reactions]
        )
        coords = np.append(coords, [[0, 0], [1, 0]], axis=0)

        idx_sort = coords[:, 0].argsort()

        self.coords = coords[idx_sort]
        self.reactions = [reactions_with_endpoints[i] for i in idx_sort]
        self.hull = ConvexHull(self.coords)

    def plot(self, y_max=0.2):
        """
        Plot the reaction hull.
        """
        pts = self._get_scatter()
        lines = self._get_lines()

        fig = Figure(data=lines + [pts])

        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br> <br><b>Mixing ratio</b>:"
                " %{x:.3f}<br><b>Energy</b>: %{y:.3f} (eV/atom)"
            )
        )
        fig.update_layout(yaxis_range=[min(self.coords[:, 1]) - 0.01, y_max])
        fig.update_layout()

        return fig

    def get_energy_above_hull(self, reaction):
        """
        Get the energy of a reaction above the reaction hull.
        """
        idx = self.reactions.index(reaction)
        x, y = self.coords[idx]
        return y - self.get_hull_energy(x)

    def get_x_coordinate(self, reaction):
        """Get coordinate of reaction in reaction hull."""
        x1 = reaction.reactant_fractions.get(self.c1, 0)
        x2 = reaction.reactant_fractions.get(self.c2, 0)
        return x2 / (x1 + x2)

    def get_hull_energy(self, coordinate):
        """ """
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
            elif np.isclose(coordinate, x2):
                return {self.reactions[v2]: 1.0}
            elif coordinate > x1 and coordinate < x2:
                return {
                    self.reactions[v1]: (x2 - coordinate) / (x2 - x1),
                    self.reactions[v2]: (coordinate - x1) / (x2 - x1),
                }
            else:
                continue

        raise ValueError("No reactions found!")

    def get_primary_selectivity(self, reaction: ComputedReaction, scale=100):
        """
        Calculates the competition score (c-score) for a given reaction. This formula is
        based on a methodology presented in the following paper: (TBD)

        Args:
            reaction: Reaction to calculate the competition score for.

        Returns:
            The c-score for the reaction
        """
        energy = reaction.energy_per_atom
        competing_rxns = [
            r for r in self.reactions if r != reaction and not r.is_identity
        ]
        energies = np.array([energy - r.energy_per_atom for r in competing_rxns])
        c_score = np.sum(np.log(1 + np.exp(scale * energies)))

        return c_score

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
        x = self.get_x_coordinate(reaction)
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
        coords = self.get_coords_in_range(x1, x2)
        n = len(coords) - 2

        energy = 0
        for c in combinations(range(len(coords)), 3):
            i_left, i_mid, i_right = sorted(c)

            c_left, c_mid, c_right = coords[[i_left, i_mid, i_right]]

            n_left = (i_mid - i_left) - 1
            n_right = (i_right - i_mid) - 1

            count = self.altitude_multiplicity(n_left, n_right, n)
            energy += count * self.calculate_altitude(c_left, c_mid, c_right)

        return energy

    def get_coords_in_range(self, x1, x2):
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = self.get_hull_energy(x_min), self.get_hull_energy(x_max)

        coords = [[x_min, y_min]]

        coords.extend(
            [
                self.coords[i]
                for i in self.hull_vertices
                if self.coords[i, 0] < x_max
                and self.coords[i, 0] > x_min
                and self.coords[i, 1] <= 0
            ]
        )
        coords.append([x_max, y_max])

        return np.array(coords)

    def count(self, num):
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
            count = self.count_recursive(num)[0]

        return count

    @lru_cache(maxsize=None)
    def get_decomposition_energy_and_num_paths_recursive(self, x1: float, x2: float):
        """ """
        all_coords = self.get_coords_in_range(x1, x2)

        x_min = all_coords[0, 0]
        x_max = all_coords[-1, 0]
        y_min = self.get_hull_energy(x_min)
        y_max = self.get_hull_energy(x_max)

        coords = all_coords[1:-1, :]

        if len(coords) == 0:
            val, total = 0, 1
            return val, total
        elif len(coords) == 1:
            val = self.calculate_altitude([x_min, y_min], coords[0], [x_max, y_max])
            total = 1
            return val, total
        else:
            val = 0
            total = 0
            for c in coords:
                height = self.calculate_altitude([x_min, y_min], c, [x_max, y_max])
                (
                    left_decomp,
                    left_total,
                ) = self.get_decomposition_energy_and_num_paths_recursive(x_min, c[0])
                (
                    right_decomp,
                    right_total,
                ) = self.get_decomposition_energy_and_num_paths_recursive(c[0], x_max)

                # # for debug counting purposes
                # for _ in range(right_total - 1):
                #     self.get_decomposition_energy_and_num_paths(x_min, c[0])
                # for _ in range(left_total - 1):
                #     self.get_decomposition_energy_and_num_paths(c[0], x_max)

                val += (
                    height * (left_total * right_total)
                    + left_decomp * right_total
                    + right_decomp * left_total
                )
                total += left_total * right_total

                # for j in range(left_total * right_total - 1):
                #     self.calculate_altitude([x_min, y_min], c, [x_max, y_max])

        return val, total

    def count_recursive(self, n, cache=[]):
        """
        Courtesy Max G.
        """
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
                left_divs, c1 = self.count_recursive(left, biggest_cache)

                right_divs, c2 = self.count_recursive(right, biggest_cache)

                if len(c1) > len(biggest_cache):
                    biggest_cache = c1

                if len(c2) > len(biggest_cache):
                    biggest_cache = c2

                total += left_divs * right_divs
            return total, biggest_cache + [total]

    @lru_cache(maxsize=None)
    def altitude_multiplicity(self, n_left, n_right, n):
        remainder = n - n_left - n_right - 1
        if remainder < 0:
            return 0

        return self.count(n_left) * self.count(n_right) * self.count(remainder)

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
        return np.array(
            [
                i
                for i in self.hull.vertices  # pylint: disable=not-an-iterable
                if self.coords[i, 1] <= 0
            ]
        )

    @cached_property
    def stable_reactions(self):
        """ """
        return [r for i, r in enumerate(self.reactions) if i in self.hull_vertices]

    @cached_property
    def unstable_reactions(self):
        """ """
        return [r for i, r in enumerate(self.reactions) if i not in self.hull_vertices]

    @staticmethod
    def _get_c_score(cost, competing_costs, scale=1000):
        """
        Calculates the c-score for a given reaction.

        This formula is based on a methodology presented in the following paper:
        (TBD)

        Args:
            cost: the cost of the selected reaction
            competing_costs: the costs of all other competing reactions
            scale: the (abritrary) scale factor used to scale the c-score. Defaults to 10.

        Returns:
            The c-score for the reaction
        """
        return np.sum([np.log(1 + np.exp(scale * (cost - c))) for c in competing_costs])

    @staticmethod
    def calculate_altitude(c_left, c_mid, c_right):
        """
        Calculates the altitude of a point on a line defined by three points.

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

        # l = round(c_left[0], 3)
        # m = round(c_mid[0], 3)
        # r = round(c_right[0], 3)

        # print(l, m, r)

        return y2 - yd
