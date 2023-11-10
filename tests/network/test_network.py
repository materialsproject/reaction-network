""" Tests for ReactionNetwork """


def test_from_dict(ymno_rn):
    """From_dict is called in the fixture, so just check that the graph is not None"""
    assert ymno_rn is not None
    assert ymno_rn.graph is not None
