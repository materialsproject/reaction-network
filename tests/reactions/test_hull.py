""" Tests for InterfaceReactionHull. """
import numpy as np
import pytest

from rxn_network.reactions.hull import InterfaceReactionHull


@pytest.fixture(scope="module")
def stable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 2 BaO -> Ba2TiO4":
            return r


@pytest.fixture(scope="module")
def unstable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 0.9 BaO -> 0.1 Ti10O11 + 0.9 BaO2":
            return r


def test_stable_reactions(irh_batio):
    stable_rxns = [
        "BaO -> BaO",
        "TiO2 + 2 BaO -> Ba2TiO4",
        "TiO2 + BaO -> BaTiO3",
        "TiO2 + 0.5 BaO -> 0.5 BaTi2O5",
        "TiO2 + 0.3077 BaO -> 0.07692 Ba4Ti13O30",
        "TiO2 + 0.2 BaO -> 0.2 BaTi5O11",
        "TiO2 + 0.1667 BaO -> 0.1667 BaTi6O13",
        "18 TiO2 -> O2 + 2 Ti9O17",
    ]

    stable_rxns = [
        actual_rxn
        for r in stable_rxns
        for actual_rxn in irh_batio.reactions
        if r == str(actual_rxn)
    ]
    assert irh_batio.stable_reactions == stable_rxns


def test_unstable_reactions(irh_batio):
    assert set(irh_batio.unstable_reactions) | set(irh_batio.stable_reactions) == set(
        irh_batio.reactions
    )


@pytest.mark.parametrize(
    "x1, x2, expected_length",
    [(0, 1, 9), (0.3, 1, 9), (0.8, 0.9, 4), (0.9, 1, 3), (0.3, 0.4, 2)],
)
def test_get_coords_in_range(x1, x2, expected_length, irh_batio):
    coords = irh_batio.get_coords_in_range(x1, x2)
    assert len(coords) == expected_length
    assert coords[0, 0] == pytest.approx(x1)
    assert coords[-1, 0] == pytest.approx(x2)


def test_get_energy_above_hull(irh_batio, stable_rxn, unstable_rxn):
    assert irh_batio.get_energy_above_hull(stable_rxn) == pytest.approx(0.0)
    assert irh_batio.get_energy_above_hull(unstable_rxn) == pytest.approx(
        1.0365808689708862
    )

    for r in irh_batio.unstable_reactions:
        assert (
            irh_batio.get_energy_above_hull(r) > -1e-12
        )  # some numerical error expected here

    for r in irh_batio.stable_reactions:
        assert irh_batio.get_energy_above_hull(r) == pytest.approx(0.0)


def test_hull_vertices(irh_batio):
    correct = np.array([1, 5, 19, 28, 43, 54, 57, 117])
    np.testing.assert_almost_equal(irh_batio.hull_vertices, correct)

    assert 89 not in irh_batio.hull_vertices  # 89 is above zero and not relevant


def test_plot(irh_batio):
    irh_batio.plot()
