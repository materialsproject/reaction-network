"""Composition class used to represent a chemical composition."""

from __future__ import annotations

from functools import cached_property

from pymatgen.core.composition import Composition as PymatgenComposition


class Composition(PymatgenComposition):
    """Modified Composition class adapted from pymatgen.

    The purpose of this is to modify / extend methods for better performance within the
    rxn_network package.
    """

    def __init__(self, *args, strict: bool = False, **kwargs):
        """Very flexible Composition construction, similar to the built-in Python
        dict(). Also extended to allow simple string init.

        Takes any inputs supported by the Python built-in dict function.

        1. A dict of either {Element/Species: amount},

            {string symbol:amount}, or {atomic number:amount} or any mixture
            of these. E.g., {Element("Li"): 2, Element("O"): 1},
            {"Li":2, "O":1}, {3: 2, 8: 1} all result in a Li2O composition.
        2. Keyword arg initialization, similar to a dict, e.g.,

            Composition(Li = 2, O = 1)

        In addition, the Composition constructor also allows a single
        string as an input formula. E.g., Composition("Li2O").

        Args:
            *args: Any number of 2-tuples as key-value pairs.
            strict (bool): Only allow valid Elements and Species in the Composition. Defaults to False.
            allow_negative (bool): Whether to allow negative compositions. Defaults to False.
            **kwargs: Additional kwargs supported by the dict() constructor.
        """
        super().__init__(*args, strict=strict, **kwargs)

    @cached_property
    def reduced_formula(self) -> str:
        """Returns a pretty normalized formula, i.e., LiFePO4 instead of
        Li4Fe4P4O16.
        """
        return self.get_reduced_formula_and_factor()[0]
