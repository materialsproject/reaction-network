""" Test for BasicReaction. Some tests adapted from pymatgen. """
import unittest

from pymatgen.core.composition import Composition
from rxn_network.reactions.basic import BasicReaction


class BasicReactionTest(unittest.TestCase):
    def test_init(self):
        reactants = [Composition("Fe"), Composition("O2")]
        products = [Composition("Fe2O3")]
        coefficients = [-2, -1.5, 1]

        rxn = BasicReaction(reactants + products, coefficients, balanced=True)
        self.assertEqual(str(rxn), "2 Fe + 1.5 O2 -> Fe2O3")

    def test_balance(self):
        pass

    def test_is_identity(self):
        pass

    def test_copy(self):
        pass

    def test_reverse(self):
        pass

    def test_normalize(self):
        pass

    def test_reduce(self):
        pass

    def test_eq(self):
        pass

    def test_from_str(self):
        pass