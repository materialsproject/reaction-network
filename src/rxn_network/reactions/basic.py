"""
This module for defining chemical reaction objects was originally sourced from
pymatgen and streamlined for the reaction-network code.
"""

import re
from itertools import chain, combinations
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from monty.fractions import gcd_float
from pymatgen.core.composition import Composition, Element

from rxn_network.core import Reaction


class BasicReaction(Reaction):
    """
    An object representing a basic chemical reaction: compositions and their
    coefficients.
    """

    # Tolerance for determining if a particular component fraction is > 0.
    TOLERANCE = 1e-6

    def __init__(
        self,
        compositions: List[Composition],
        coefficients: Union[List[float], np.ndarray],
        balanced: Optional[bool] = None,
        data: Optional[Dict] = None,
        lowest_num_errors: Union[int, float] = 0,
    ):
        """
        A BasicReaction object is defined by a list of compositions and their
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

        self.reactant_coeffs = {
            comp: coeff
            for comp, coeff in zip(self._compositions, self._coefficients)
            if coeff < 0
        }
        self.product_coeffs = {
            comp: coeff
            for comp, coeff in zip(self._compositions, self._coefficients)
            if coeff > 0
        }

        if balanced is not None:
            self.balanced = balanced
        else:
            sum_reactants = sum(
                [k * abs(v) for k, v in self.reactant_coeffs.items()], Composition({})
            )
            sum_products = sum(
                [k * abs(v) for k, v in self.product_coeffs.items()], Composition({})
            )

            if not sum_reactants.almost_equals(
                sum_products, rtol=0, atol=self.TOLERANCE
            ):
                self.balanced = False
            else:
                self.balanced = True

        self.data = data
        self.lowest_num_errors = lowest_num_errors

    @classmethod
    def balance(
        cls,
        reactants: List[Composition],
        products: List[Composition],
        data: Optional[Dict] = None,
    ) -> "BasicReaction":
        """
        Reactants and products to be specified as list of
        pymatgen.core.Composition. e.g., [comp1, comp2]

        Args:
            reactants: List of reactants.
            products: List of products.
            data: Optional dictionary containing extra data about the reaction.
        """
        compositions = reactants + products
        coeffs, lowest_num_errors = cls._balance_coeffs(reactants, products)

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

    def normalize_to(self, comp: Composition, factor: float = 1) -> "BasicReaction":
        """
        Normalizes the reaction to one of the compositions via the provided factor.

        By default, normalizes such that the composition given has a coefficient of
        1.

        Args:
            comp: Composition object to normalize to
            factor: factor to normalize to. Defaults to 1.
        """
        all_comp = self.compositions
        coeffs = self.coefficients
        scale_factor = abs(1 / coeffs[self.compositions.index(comp)] * factor)
        coeffs *= scale_factor
        return BasicReaction(all_comp, coeffs)

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
        coeffs *= scale_factor
        return BasicReaction(all_comp, coeffs)

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

    def get_coeff(self, comp: Composition):
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

    def copy(self) -> "BasicReaction":
        """Returns a copy of the BasicReaction object"""
        return BasicReaction(
            compositions=self.compositions,
            coefficients=self.coefficients,
            balanced=self.balanced,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    def reverse(self) -> "BasicReaction":
        """
        Returns a copy of the original BasicReaction object where original reactants are
        new products, and vice versa.
        """
        return BasicReaction(
            compositions=self.compositions,
            coefficients=-1 * self.coefficients,
            balanced=self.balanced,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    @classmethod
    def from_string(cls, rxn_string) -> "BasicReaction":
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

        reactant_coeffs = get_comp_amt(rct_str)
        product_coeffs = get_comp_amt(prod_str)

        return cls._from_coeff_dicts(reactant_coeffs, product_coeffs)

    @classmethod
    def from_formulas(
        cls, reactants: List[str], products: List[str]
    ) -> "BasicReaction":
        """

        Args:
            reactants:
            products:

        Returns:

        """

        reactant_comps = [Composition(r) for r in reactants]
        product_comps = [Composition(p) for p in products]
        rxn = cls.balance(reactants=reactant_comps, products=product_comps)

        return rxn

    @property
    def reactants(self) -> List[Composition]:
        """List of reactants for this reaction"""
        return list(self.reactant_coeffs.keys())

    @property
    def products(self) -> List[Composition]:
        """List of products for this reaction"""
        return list(self.product_coeffs.keys())

    @property
    def compositions(self) -> List[Composition]:
        """List of composition objects for this reaction"""
        return self._compositions

    @property
    def coefficients(self) -> np.ndarray:  # pylint: disable = W0236
        """Array of reaction coefficients"""
        return self._coefficients

    @property
    def energy(self) -> float:
        """The energy of this reaction"""
        raise ValueError("No energy for a basic reaction!")

    @property
    def energy_per_atom(self) -> float:
        """The energy per atom of this reaction"""
        raise ValueError("No energy per atom for a basic reaction!")

    @property
    def is_identity(self):
        """Returns True if the reaction has identical reactants and products"""
        if set(self.reactants) != set(self.products):
            return False
        if self.balanced is False:  # if not balanced, can not check coefficients
            return True
        return all(
            [
                np.isclose(self.reactant_coeffs[c] * -1, self.product_coeffs[c])
                for c in self.reactant_coeffs
            ]
        )

    @property
    def chemical_system(self):
        """Returns the chemical system as string in the form of A-B-C-..."""
        return "-".join(sorted([str(el) for el in self.elements]))

    @property
    def normalized_repr(self):
        """
        A normalized representation of the reaction. All factors are converted
        to lowest common factors.
        """
        return self.normalized_repr_and_factor()[0]

    @classmethod
    def _balance_coeffs(
        cls, reactants: List[Composition], products: List[Composition]
    ) -> Tuple[np.ndarray, Union[int, float]]:
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

        # start with simplest product constraints, work to more complex constraints
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
        best_soln = np.zeros(num_comp)

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

        return np.squeeze(best_soln), lowest_num_errors

    @staticmethod
    def _from_coeff_dicts(reactant_coeffs, product_coeffs) -> "BasicReaction":
        reactant_comps, r_coefs = zip(
            *[(comp, -1 * coeff) for comp, coeff in reactant_coeffs.items()]
        )
        product_comps, p_coefs = zip(
            *[(comp, coeff) for comp, coeff in product_coeffs.items()]
        )
        return BasicReaction(reactant_comps + product_comps, r_coefs + p_coefs)

    @classmethod
    def _str_from_formulas(cls, coeffs, formulas) -> str:
        reactant_str = []
        product_str = []
        tol = cls.TOLERANCE
        for amt, formula in zip(coeffs, formulas):
            if abs(amt + 1) < tol:
                reactant_str.append(formula)
            elif abs(amt - 1) < tol:
                product_str.append(formula)
            elif amt < -tol:
                reactant_str.append("{:.4g} {}".format(-amt, formula))
            elif amt > tol:
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

    def __eq__(self, other):
        if self is other:
            return True
        elif str(self) == str(other):
            return True
        else:
            return (set(self.reactants) == set(other.reactants)) & (
                set(self.products) == set(other.products)
            )

    def __hash__(self):
        return hash(
            "-".join(
                [e.reduced_formula for e in sorted(self.reactants)]
                + [e.reduced_formula for e in sorted(self.products)]
            )
        )

    def __str__(self):
        return self._str_from_comp(self.coefficients, self.compositions)[0]

    def __repr__(self):
        return self.__str__()
