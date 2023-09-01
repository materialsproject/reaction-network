"""Composition class used to represent a chemical composition."""
from functools import cached_property

from pymatgen.core.composition import Composition as PymatgenComposition


class Composition(PymatgenComposition):
    """
    Modified Composition class adapted from pymatgen.

    The purpose of this is to modify / extend methods for better performance within the
    rxn_network package.

    """

    def __init__(
        self, *args, strict: bool = False, **kwargs
    ):  # pylint: disable=useless-parent-delegation
        super().__init__(*args, strict=strict, **kwargs)

    @cached_property
    def reduced_formula(self) -> str:
        """
        Returns a pretty normalized formula, i.e., LiFePO4 instead of
        Li4Fe4P4O16.
        """
        return self.get_reduced_formula_and_factor()[0]
