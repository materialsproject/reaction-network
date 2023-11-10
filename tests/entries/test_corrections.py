""" Tests for entry corrections. """
import pytest
from rxn_network.entries.corrections import CARBONATE_CORRECTION, CarbonateCorrection


@pytest.fixture()
def carbonate_correction_1():
    return CarbonateCorrection(1)


@pytest.fixture()
def carbonate_correction_3():
    return CarbonateCorrection(3)


def test_num_ions(carbonate_correction_1, carbonate_correction_3):
    assert carbonate_correction_1.num_ions == 1
    assert carbonate_correction_3.num_ions == 3


def test_carbonate_correction(carbonate_correction_1):
    assert carbonate_correction_1.carbonate_correction == CARBONATE_CORRECTION
