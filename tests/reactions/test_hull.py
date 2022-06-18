""" Tests for InterfaceReactionHull. """
import pytest

from pymatgen.core.composition import Element, Composition
from rxn_network.reactions.hull import InterfaceReactionHull


@pytest.fixture
def irh_basic(bao_tio2_rxns):
    return InterfaceReactionHull(
        c1=Composition("BaO"), c2=Composition("TiO2"), reactions=bao_tio2_rxns
    )


@pytest.fixture
def selected_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 2 BaO -> Ba2TiO4":
            return r


def test_stable_reactions():
    pass


def test_unstable_reactions():
    pass


def test_calculate_altitude():
    c1 = (0.0, 0.0)
    c2 = (0.5, -1.0)
    c3 = (1.0, 0.0)
    expected_altitude_1 = -1.0

    c4 = (0.25, -0.5)
    c5 = (0.4,)

    c7 = (0.33333, -0.23)
    c8 = 0.777

    assert InterfaceReactionHull.calculate_altitude(c1, c2, c3) == pytest.approx(
        expected_altitude_1
    )
    assert InterfaceReactionHull.calculate_altitude(c4, c5, c6) == pytest.approx(
        expected_altitude_2
    )
    assert InterfaceReactionHull.calculate_altitude(c7, c8, c9) == pytest.approx(
        expected_altitude_3
    )


def test_get_competition_score():
    pass


def test_get_selectivity_score():
    pass


def test_get_energy_above_hull(irh_basic, selected_rxn):
    altitude = irh_basic.calculate_altitude(
        [0, 0], [0.250, -0.2824416739], [0.3333333, -0.3837805055]
    )
    e_above_hull = irh.get_energy_above_hull(selected_rxn)
    precalculated_answer == 0.005393734008540763

    assert altitude == pytest.approx(e_above_hull)


def test_decomposition_energy_and_num_paths():
    pass


def test_plot(irh_basic):
    irh_basic.plot()
