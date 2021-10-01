""" Tests for Softplus """
import pytest
from rxn_network.costs.softplus import Softplus


@pytest.fixture(scope="module")
def softplus_with_attr():
    return Softplus(temp=800, params=["energy_per_atom"], weights=[1.0])


@pytest.fixture(scope="module")
def softplus_with_attr_and_param():
    return Softplus(
        temp=800, params=["energy_per_atom", "test_param"], weights=[0.3, 0.7]
    )


def test_evaluate(softplus_with_attr, softplus_with_attr_and_param, computed_rxn):
    r = computed_rxn.copy()
    r.data = {"test_param": 0.1}

    cost1 = softplus_with_attr.evaluate(r)
    cost2 = softplus_with_attr_and_param.evaluate(r)

    assert cost1 == pytest.approx(0.250771199)
    assert cost2 == pytest.approx(0.297691790)


def test_missing_parameter(softplus_with_attr_and_param, computed_rxn):
    with pytest.raises(ValueError) as error:
        assert softplus_with_attr_and_param.evaluate(computed_rxn)
    assert str(error.value) == "Reaction is missing parameter test_param!"
