# coding: utf-8
"""
This module for defining chemical reaction objects was originally sourced from
pymatgen and streamlined for the reaction-network code.
"""

import re
from functools import cached_property
from itertools import chain, combinations
from typing import Dict, List, Optional

import numpy as np
from monty.fractions import gcd_float
from pymatgen.core.composition import Composition, Element

from rxn_network.core import Reaction


class BasicReaction(Reaction):
    """
    An object representing a basic chemical reaction.
    """

    # Tolerance for determining if a particular component fraction is > 0.
    TOLERANCE = 1e-6

    def __init__(
        self,
        reactant_coeffs: Dict[Composition, float],
        product_coeffs: Dict[Composition, float],
        balanced: Optional[bool] = None,
    ):
        """
        Reactants and products to be specified as dict of {Composition: coeff}.

        Args:
            reactant_coeffs ({Composition: float}): Reactants as dict of
                {Composition: amt}.
            product_coeffs ({Composition: float}): Products as dict of
                {Composition: amt}.
        """
        self.reactant_coeffs = reactant_coeffs
        self.product_coeffs = product_coeffs

        if balanced is not None:
            self.balanced = balanced
        else:
            # sum reactants and products
            all_reactants = sum(
                [k * v for k, v in reactant_coeffs.items()], Composition({})
            )
            all_products = sum(
                [k * v for k, v in product_coeffs.items()], Composition({})
            )

            if not all_reactants.almost_equals(
                all_products, rtol=0, atol=self.TOLERANCE
            ):
                self.balanced = False
            else:
                self.balanced = True

    @property
    def reactants(self) -> List[Composition]:
        " List of reactants for this reaction "
        return list(self.reactant_coeffs.keys())

    @property
    def products(self) -> List[Composition]:
        " List of products for this reaction "
        return list(self.product_coeffs.keys())

    @cached_property
    def coefficients(self) -> np.array:  # pylint: disable = W0236
        """
        Coefficients of the reaction
        """
        return np.concatenate(
            np.array(self.reactant_coeffs.values()) * -1
            + np.array(self.product_coeffs.values)
        )

    @property
    def energy(self) -> float:
        raise ValueError("No energy for a basic reaction")

    def copy(self) -> "BasicReaction":
        """
        Returns a copy of the Reaction object.
        """
        return BasicReaction(self.reactant_coeffs, self.product_coeffs, self.balanced)

    def normalize_to(self, comp: Composition, factor: float = 1) -> "BasicReaction":
        """
        Normalizes the reaction to one of the compositions.
        By default, normalizes such that the composition given has a
        coefficient of 1. Another factor can be specified.

        Args:
            comp (Composition): Composition to normalize to
            factor (float): Factor to normalize to. Defaults to 1.
        """
        scale_factor = abs(
            1 / self.coefficients[self.compositions.index(comp)] * factor
        )
        self.coefficients *= scale_factor
        return self

    def normalize_to_element(
        self, element: Element, factor: float = 1
    ) -> "BasicReaction":
        """
        Normalizes the reaction to one of the elements.
        By default, normalizes such that the amount of the element is 1.
        Another factor can be specified.

        Args:
            element (Element/Species): Element to normalize to.
            factor (float): Factor to normalize to. Defaults to 1.
        """
        all_comp = self.compositions
        coeffs = self.coefficients
        current_el_amount = (
            sum([all_comp[i][element] * abs(coeffs[i]) for i in range(len(all_comp))])
            / 2
        )
        scale_factor = factor / current_el_amount
        self.coefficients *= scale_factor
        return self

    def get_el_amount(self, element: Element) -> float:
        """
        Returns the amount of the element in the reaction.

        Args:
            element (Element/Species): Element in the reaction

        Returns:
            Amount of that element in the reaction.
        """
        return (
            sum(
                [
                    self.compositions[i][element] * abs(self.coefficients[i])
                    for i in range(len(self.compositions))
                ]
            )
            / 2
        )

    def get_coeff(self, comp):
        """
        Returns coefficient for a particular composition
        """
        return self.coefficients[self.compositions.index(comp)]

    def normalized_repr_and_factor(self):
        """
        Normalized representation for a reaction
        For example, ``4 Li + 2 O -> 2Li2O`` becomes ``2 Li + O -> Li2O``
        """
        return self._str_from_comp(self.coefficients, self.compositions, True)

    @property
    def normalized_repr(self):
        """
        A normalized representation of the reaction. All factors are converted
        to lowest common factors.
        """
        return self.normalized_repr_and_factor()[0]

    @classmethod
    def _str_from_formulas(cls, coeffs, formulas) -> str:
        reactant_str = []
        product_str = []
        for amt, formula in zip(coeffs, formulas):
            if abs(amt + 1) < cls.TOLERANCE:
                reactant_str.append(formula)
            elif abs(amt - 1) < cls.TOLERANCE:
                product_str.append(formula)
            elif amt < -cls.TOLERANCE:
                reactant_str.append("{:.4g} {}".format(-amt, formula))
            elif amt > cls.TOLERANCE:
                product_str.append("{:.4g} {}".format(amt, formula))

        return " + ".join(reactant_str) + " -> " + " + ".join(product_str)

    @classmethod
    def _str_from_comp(cls, coeffs, compositions, reduce=False):
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

    def __str__(self):
        return self._str_from_comp(self.coefficients, self.compositions)[0]

    __repr__ = __str__

    @staticmethod
    def from_string(rxn_string) -> "BasicReaction":
        """
        Generates a balanced reaction from a string. The reaction must
        already be balanced.

        Args:
            rxn_string:
                The reaction string. For example, "4 Li + O2-> 2Li2O"

        Returns:
            BalancedReaction
        """
        rct_str, prod_str = rxn_string.split("->")

        def get_comp_amt(comp_str):
            return {
                Composition(m.group(2)): float(m.group(1) or 1)
                for m in re.finditer(
                    r"([\d\.]*(?:[eE]-?[\d\.]+)?)\s*([A-Z][\w\.\(\)]*)", comp_str
                )
            }

        return BasicReaction(get_comp_amt(rct_str), get_comp_amt(prod_str))

    @classmethod
    def _balance_coeffs(
        cls, reactants: List[Composition], products: List[Composition]
    ) -> np.array:
        """
        Balances the reaction and returns the new coefficient matrix
        """
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

        # start with simplest product constraints, work towards most complex reactant constraints
        product_constraints = chain.from_iterable(
            [
                combinations(range(first_product_idx, num_comp), n_constr)
                for n_constr in range(num_constraints, 0, -1)
            ]
        )
        reactant_constraints = chain.from_iterable(
            [
                combinations(range(0, first_product_idx), n_constr)
                for n_constr in range(num_constraints, 0, -1)
            ]
        )
        best_soln = None

        for constraints in chain(product_constraints, reactant_constraints):
            n_constr = len(constraints)

            comp_and_constraints = np.append(
                comp_matrix, np.zeros((n_constr, num_comp)), axis=0
            )
            b = np.zeros((num_elems + n_constr, 1))
            b[-n_constr:] = 1 if min(constraints) >= first_product_idx else -1

            for num, idx in enumerate(constraints):
                comp_and_constraints[num_elems + num, idx] = 1
                # arbitrarily fix coeff to 1

            coeffs = np.matmul(np.linalg.pinv(comp_and_constraints), b)

            if np.allclose(np.matmul(comp_matrix, coeffs), np.zeros((num_elems, 1))):
                expected_signs = np.array([-1] * len(reactants) + [+1] * len(products))
                num_errors = np.sum(
                    np.multiply(expected_signs, coeffs.T) < cls.TOLERANCE
                )
                if num_errors == 0:
                    lowest_num_errors = 0
                    best_soln = coeffs
                    break
                if num_errors < lowest_num_errors:
                    lowest_num_errors = num_errors
                    best_soln = coeffs

        return np.squeeze(best_soln)

    @classmethod
    def balance(
        cls, reactants: List[Composition], products: List[Composition]
    ) -> "BasicReaction":
        """
        Reactants and products to be specified as list of
        pymatgen.core.structure.Composition.  e.g., [comp1, comp2]

        Args:
            reactants ([Composition]): List of reactants.
            products ([Composition]): List of products.
        """
        compositions = reactants + products

        coeffs = cls._balance_coeffs(reactants, products)

        new_reactants = {
            comp: abs(num) for comp, num in zip(compositions, coeffs) if num < 0
        }
        new_products = {
            comp: abs(num) for comp, num in zip(compositions, coeffs) if num > 0
        }

        return cls(reactant_coeffs=new_reactants, product_coeffs=new_products)
