from functools import cached_property, lru_cache

from pymatgen.core.composition import Composition as PymatgenComposition
from pymatgen.core.composition import reduce_formula


class Composition(PymatgenComposition):
    """Customized Composition class adapted from pymatgen"""

    def __init__(self, *args, strict: bool = False, **kwargs):
        super().__init__(*args, strict=strict, **kwargs)

    @cached_property
    def reduced_formula(self) -> str:
        """
        Returns a pretty normalized formula, i.e., LiFePO4 instead of
        Li4Fe4P4O16.
        """
        return self.get_reduced_formula_and_factor()[0]