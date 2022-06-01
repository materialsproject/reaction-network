"""Code for analyzing sets of reactions between two phases."""

from typing import List
from functools import cached_property

import numpy as np
from monty.json import MSONable
from plotly.express import scatter
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

    def plot(self):
        """
        Plot the reaction hull.
        """
        return scatter(x=self.coords[:, 0], y=self.coords[:, 1])

    def get_energy_above_hull(self, reaction):
        """
        Get the energy of a reaction above the reaction hull.
        """
        idx = self.reactions.index(reaction)
        x, y = self.coords[idx]
        return y - self.get_hull_energy(x)

    def get_x_coordinate(self, reaction):
        """Get coordinate of reaction in reaction hull."""
        x1 = reaction.reactant_fractions[self.c1]
        x2 = reaction.reactant_fractions[self.c2]
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

    def get_competition_score(self, reaction: ComputedReaction, scale=1000):
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
        c_score = np.sum(
            [
                np.log(1 + np.exp(scale * (energy - r.energy_per_atom)))
                for r in competing_rxns
            ]
        )
        return c_score

    def get_selectivity_score(self, reaction: ComputedReaction):
        """ """
        x = self.get_x_coordinate(reaction)
        return (
            reaction.energy_per_atom
            + self.get_decomposition_energy(0, x)
            + self.get_decomposition_energy(x, 1)
        )

    def get_decomposition_energy(self, x1: float, x2: float):
        """ """
        pts = sorted([x1, x2])
        x_min = pts[0]
        x_max = pts[1]

        y_min = self.get_hull_energy(x_min)
        y_max = self.get_hull_energy(x_max)

        coords = [
            self.coords[i]
            for i in self.hull_vertices
            if self.coords[i, 0] < x_max
            and self.coords[i, 0] > x_min
            and self.coords[i, 1] <= 0
        ]

        if len(coords) == 0:
            return 0
        elif len(coords) == 1:
            return self.calculate_altitude([x_min, y_min], coords[0], [x_max, y_max])
        else:
            val = 0
            for c in coords:
                height = self.calculate_altitude([x_min, y_min], c, [x_max, y_max])
                left_decomp = self.get_decomposition_energy(x_min, c[0])
                right_decomp = self.get_decomposition_energy(c[0], x_max)
                val += height + left_decomp + right_decomp
            return val

    @cached_property
    def hull_vertices(self):
        return np.array([i for i in self.hull.vertices if self.coords[i, 1] <= 0])

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

        return y2 - yd
