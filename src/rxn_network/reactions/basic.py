"""This module for defining chemical reaction objects was originally sourced from
pymatgen and streamlined for the reaction-network code.
"""
from __future__ import annotations

import re
from copy import deepcopy
from functools import cached_property
from itertools import chain, combinations
from typing import TYPE_CHECKING

import numpy as np
from monty.fractions import gcd_float

from rxn_network.core import Composition
from rxn_network.data import COMMON_GASES
from rxn_network.reactions.base import Reaction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymatgen.core.periodic_table import Element

TOLERANCE = 1e-6  # Tolerance for determining if a particular component fraction is > 0.


class BasicReaction(Reaction):
    """An object representing a basic chemical reaction: compositions and their
    coefficients.
    """

    def __init__(
        self,
        compositions: Iterable[Composition],
        coefficients: Iterable[float] | np.ndarray,
        balanced: bool | None = None,
        data: dict | None = None,
        lowest_num_errors: float = 0,
    ):
        """A BasicReaction object is defined by a list of compositions and their
        corresponding coefficients, where a negative coefficient refers to a
        reactant, and a positive coefficient refers to a product.

        Args:
            compositions: List of composition objects (pymatgen).
            coefficients: List of coefficients, where negative coeff distinguishes a
                reactant.
            balanced: Whether the reaction is stoichiometricaly balanced or not
                (see construction via balance() method).
            data: Optional corresponding data in dictionary format; often used to store
                various calculated parameters.
            lowest_num_errors: the minimum number of errors reported by the reaction
                balancing algorithm (see the balance() method). A number of errors
                >= 1 means that the reaction may be different than intended (some
                phases may be shuffled or removed entirely).
        """
        self._compositions = [Composition(c) for c in compositions]
        self._coefficients = np.array(coefficients)

        self.reactant_coeffs = {comp: coeff for comp, coeff in zip(self._compositions, self._coefficients) if coeff < 0}
        self.product_coeffs = {comp: coeff for comp, coeff in zip(self._compositions, self._coefficients) if coeff > 0}

        if balanced is not None:
            self.balanced = balanced
        else:
            sum_reactants = sum([k * abs(v) for k, v in self.reactant_coeffs.items()], Composition({}))
            sum_products = sum([k * abs(v) for k, v in self.product_coeffs.items()], Composition({}))

            if not sum_reactants.almost_equals(sum_products, rtol=0, atol=TOLERANCE):
                self.balanced = False
            else:
                self.balanced = True

        self.data = data if data else {}
        self.lowest_num_errors = lowest_num_errors

    @classmethod
    def balance(
        cls,
        reactants: list[Composition],
        products: list[Composition],
        data: dict | None = None,
    ) -> BasicReaction:
        """Reactants and products to be specified as list of
        pymatgen.core.Composition. e.g., [comp1, comp2].

        Args:
            reactants: List of reactants.
            products: List of products.
            data: Optional dictionary containing extra data about the reaction.
        """
        compositions = list(reactants + products)
        coeffs, lowest_num_errors, num_constraints = cls._balance_coeffs(reactants, products)
        if not data:
            data = {}
        data["num_constraints"] = num_constraints

        balanced = True
        if coeffs is None or lowest_num_errors == np.inf:
            balanced = False
            coeffs = np.zeros(len(compositions))

        return cls(
            compositions=compositions,
            coefficients=coeffs,
            balanced=balanced,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )

    def normalize_to(self, comp: Composition, factor: float = 1) -> BasicReaction:
        """Normalizes the reaction to one of the compositions via the provided factor.

        By default, normalizes such that the composition given has a coefficient of
        1.

        Args:
            comp: Composition object to normalize to
            factor: factor to normalize to. Defaults to 1.
        """
        all_comp = self.compositions
        coeffs = self.coefficients.copy()
        scale_factor = abs(1 / coeffs[self.compositions.index(comp)] * factor)
        coeffs *= scale_factor
        return BasicReaction(all_comp, coeffs)

    def normalize_to_element(self, element: Element, factor: float = 1) -> BasicReaction:
        """Normalizes the reaction to one of the elements.
        By default, normalizes such that the amount of the element is 1.
        Another factor can be specified.

        Args:
            element (Element/Species): Element to normalize to.
            factor (float): Factor to normalize to. Defaults to 1.
        """
        all_comp = self.compositions
        coeffs = self.coefficients.copy()
        current_el_amount = sum(all_comp[i][element] * abs(coeffs[i]) for i in range(len(all_comp))) / 2
        scale_factor = factor / current_el_amount
        coeffs *= scale_factor
        return BasicReaction(all_comp, coeffs)

    def get_el_amount(self, element: Element) -> float:
        """Returns the amount of the element in the reaction.

        Args:
            element (Element/Species): Element in the reaction

        Returns:
            Amount of that element in the reaction.
        """
        return sum(self.compositions[i][element] * abs(self.coefficients[i]) for i in range(len(self.compositions))) / 2

    def get_coeff(self, comp: Composition):
        """Returns coefficient for a particular composition."""
        return self.coefficients[self.compositions.index(comp)]

    def normalized_repr_and_factor(self):
        """Normalized representation for a reaction
        For example, ``4 Li + 2 O -> 2Li2O`` becomes ``2 Li + O -> Li2O``.
        """
        return self._str_from_comp(self.coefficients, self.compositions, reduce=True)

    def copy(self) -> BasicReaction:
        """Returns a copy of the BasicReaction object."""
        return BasicReaction(
            compositions=self.compositions,
            coefficients=self.coefficients.copy(),
            balanced=self.balanced,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    def reverse(self) -> BasicReaction:
        """Returns a copy of the original BasicReaction object where original reactants are
        new products, and vice versa.
        """
        return BasicReaction(
            compositions=self.compositions,
            coefficients=-1 * self.coefficients.copy(),
            balanced=self.balanced,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    def is_separable(self, target: str | Composition) -> bool:
        """Checks if the reaction forms byproducts which are separable from the target
        composition.

        Separable byproducts are those that are common gases (e.g., CO2), or other phases that do not contain any of the
        elements in the target phase.

        Args:
            target: Composition of target; elements in this phase will be used to
                determine whether byproducts only contain added elements.

        Returns:
            True if reaction is separable from target, False otherwise.
        """
        target = Composition(target)
        identified_targets = [c for c in self.compositions if c.reduced_composition == target.reduced_composition]

        if len(identified_targets) == 0:
            raise ValueError(f"Target composition {target} not in reaction {self}")

        added_elems = set(self.elements) - set(target.elements)
        products = set(deepcopy(self.products))

        for t in identified_targets:
            products.remove(t)

        are_separable = []
        for comp in products:
            if comp in COMMON_GASES or added_elems.issuperset(comp.elements):
                are_separable.append(True)
            else:
                are_separable.append(False)

        return all(are_separable)

    @cached_property
    def reactant_atomic_fractions(self) -> dict:
        """Returns the atomic mixing ratio of reactants in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        return {
            c.reduced_composition: -coeff * c.num_atoms / self.num_atoms for c, coeff in self.reactant_coeffs.items()
        }

    @cached_property
    def product_atomic_fractions(self) -> dict:
        """Returns the atomic mixing ratio of reactants in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        return {c.reduced_composition: coeff * c.num_atoms / self.num_atoms for c, coeff in self.product_coeffs.items()}

    @cached_property
    def reactant_molar_fractions(self) -> dict:
        """Returns the molar mixing ratio of reactants in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        total = sum(self.reactant_coeffs.values())

        return {c: coeff / total for c, coeff in self.reactant_coeffs.items()}

    @cached_property
    def product_molar_fractions(self) -> dict:
        """Returns the molar mixing ratio of products in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        total = sum(self.product_coeffs.values())

        return {c: coeff / total for c, coeff in self.product_coeffs.items()}

    @classmethod
    def from_string(cls, rxn_string: str) -> BasicReaction:
        """Generates a balanced reaction from a string. The reaction must
        already be balanced.

        Args:
            rxn_string:
                The reaction string. For example, "4 Li + O2-> 2 Li2O"

        Returns:
            BalancedReaction
        """
        rct_str, prod_str = rxn_string.split("->")

        def get_comp_amt(comp_str):
            return {
                Composition(m.group(2)): float(m.group(1) or 1)
                for m in re.finditer(r"([\d\.]*(?:[eE]-?[\d\.]+)?)\s*([A-Z][\w\.\(\)]*)", comp_str)
            }

        reactant_coeffs = get_comp_amt(rct_str)
        product_coeffs = get_comp_amt(prod_str)

        return cls._from_coeff_dicts(reactant_coeffs, product_coeffs)

    @classmethod
    def from_formulas(cls, reactants: list[str], products: list[str]) -> BasicReaction:
        """Initialize a reaction from a list of 1) reactant formulas and 2) product
        formulas.

        Args:
            reactants: List of reactant formulas
            products: List of product formulas

        Returns:
            A BasicReaction object
        """
        reactant_comps = [Composition(r) for r in reactants]
        product_comps = [Composition(p) for p in products]
        return cls.balance(reactants=reactant_comps, products=product_comps)

    @property
    def reactants(self) -> list[Composition]:
        """List of reactants for this reaction."""
        return list(self.reactant_coeffs.keys())

    @property
    def products(self) -> list[Composition]:
        """List of products for this reaction."""
        return list(self.product_coeffs.keys())

    @property
    def compositions(self) -> list[Composition]:
        """List of composition objects for this reaction."""
        return self._compositions

    @property
    def coefficients(self) -> np.ndarray:
        """Array of reaction coefficients."""
        return self._coefficients

    @cached_property
    def num_atoms(self) -> float:
        """Total number of atoms in this reaction."""
        return sum(coeff * sum(c[el] for el in self.elements) for c, coeff in self.product_coeffs.items())

    @cached_property
    def energy(self) -> float:
        """The energy of this reaction."""
        raise ValueError("No energy for a basic reaction!")

    @cached_property
    def energy_per_atom(self) -> float:
        """The energy per atom of this reaction."""
        raise ValueError("No energy per atom for a basic reaction!")

    @cached_property
    def is_identity(self) -> bool:
        """Returns True if the reaction has identical reactants and products."""
        return self._get_is_identity()

    def _get_is_identity(self):
        """Returns True if the reaction has identical reactants and products."""
        if set(self.reactants) != set(self.products):
            return False
        if self.balanced is False:  # if not balanced, can not check coefficients
            return True
        return all(np.isclose(self.reactant_coeffs[c] * -1, self.product_coeffs[c]) for c in self.reactant_coeffs)

    @cached_property
    def chemical_system(self) -> str:
        """Returns the chemical system as string in the form of A-B-C-..."""
        return "-".join(sorted([str(el) for el in self.elements]))

    @property
    def normalized_repr(self) -> str:
        """A normalized representation of the reaction. All factors are converted
        to lowest common factors.
        """
        return self.normalized_repr_and_factor()[0]

    @classmethod
    def _balance_coeffs(
        cls, reactants: list[Composition], products: list[Composition]
    ) -> tuple[np.ndarray, int | float, int]:
        """Balances the reaction and returns the new coefficient matrix."""
        compositions = reactants + products
        num_comp = len(compositions)

        all_elems = sorted({elem for c in compositions for elem in c.elements})
        num_elems = len(all_elems)

        comp_matrix = np.array([[c[el] for el in all_elems] for c in compositions]).T

        rank = np.linalg.matrix_rank(comp_matrix)
        diff = num_comp - rank
        num_constraints = diff if diff >= 2 else 1

        # an error = a component changing sides or disappearing
        lowest_num_errors = np.inf

        first_product_idx = len(reactants)

        # start with simplest product constraints, work to more complex constraints
        product_constraints = chain.from_iterable(
            [combinations(range(first_product_idx, num_comp), n_constr) for n_constr in range(num_constraints, 0, -1)]
        )
        reactant_constraints = chain.from_iterable(
            [combinations(range(first_product_idx), n_constr) for n_constr in range(num_constraints, 0, -1)]
        )
        best_soln = np.zeros(num_comp)

        for constraints in chain(product_constraints, reactant_constraints):
            n_constr = len(constraints)

            comp_and_constraints = np.append(comp_matrix, np.zeros((n_constr, num_comp)), axis=0)
            b = np.zeros((num_elems + n_constr, 1))
            b[-n_constr:] = 1 if min(constraints) >= first_product_idx else -1

            for num, idx in enumerate(constraints):
                comp_and_constraints[num_elems + num, idx] = 1
                # arbitrarily fix coeff to 1

            coeffs = np.matmul(np.linalg.pinv(comp_and_constraints), b)

            num_errors = 0
            if np.allclose(np.matmul(comp_matrix, coeffs), np.zeros((num_elems, 1))):
                expected_signs = np.array([-1] * len(reactants) + [+1] * len(products))
                num_errors = np.sum(np.multiply(expected_signs, coeffs.T) < TOLERANCE)
                if num_errors == 0:
                    lowest_num_errors = 0
                    best_soln = coeffs
                    break
                if num_errors < lowest_num_errors:
                    lowest_num_errors = num_errors
                    best_soln = coeffs

        return np.squeeze(best_soln), lowest_num_errors, num_constraints

    @staticmethod
    def _from_coeff_dicts(reactant_coeffs, product_coeffs) -> BasicReaction:
        reactant_comps, r_coefs = zip(*[(comp, -1 * coeff) for comp, coeff in reactant_coeffs.items()])
        product_comps, p_coefs = zip(*list(product_coeffs.items()))
        return BasicReaction(reactant_comps + product_comps, r_coefs + p_coefs)

    @staticmethod
    def _str_from_formulas(coeffs, formulas, tol=TOLERANCE) -> str:
        reactant_str = []
        product_str = []
        for amt, formula in zip(coeffs, formulas):
            if abs(amt + 1) < tol:
                reactant_str.append(formula)
            elif abs(amt - 1) < tol:
                product_str.append(formula)
            elif amt < -tol:
                reactant_str.append(f"{-amt:.4g} {formula}")
            elif amt > tol:
                product_str.append(f"{amt:.4g} {formula}")

        return " + ".join(reactant_str) + " -> " + " + ".join(product_str)

    @classmethod
    def _str_from_comp(cls, coeffs, compositions, reduce=False) -> tuple[str, float]:
        r_coeffs = np.zeros(len(coeffs))
        r_formulas = []
        for i, (amt, comp) in enumerate(zip(coeffs, compositions)):
            formula, factor = comp.get_reduced_formula_and_factor()
            r_coeffs[i] = amt * factor
            r_formulas.append(formula)
        if reduce:
            factor = 1 / gcd_float(np.abs(r_coeffs))
            r_coeffs *= factor
        else:
            factor = 1
        return cls._str_from_formulas(r_coeffs, r_formulas), factor

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        if not self.chemical_system == other.chemical_system:
            return False

        if not len(self.products) == len(other.products):
            return False

        if not len(self.reactants) == len(other.reactants):
            return False

        if not np.allclose(sorted(self.coefficients), sorted(other.coefficients)):
            return False

        if not set(self.reactants) == set(other.reactants):
            return False

        if not set(self.products) == set(other.products):
            return False

        return True

    def __hash__(self):
        return hash(
            (self.chemical_system, tuple(sorted(self.coefficients)))
        )  # not checking here for reactions that are multiples (too expensive)

    def __str__(self) -> str:
        return self._str_from_comp(self.coefficients, self.compositions)[0]

    def __repr__(self) -> str:
        return self.__str__()
