# Copyright (c) 2017 Thomas Spura
"""
Test grand canonical simulation data
"""
import os
import pytest

import cpoints as cp


def get_test_file():
    """ Returns the path to the test data file.
    """
    return os.path.sep.join(__file__.split(os.path.sep)[:-1] +
                            ["GK_histo.dat"])


def test_read_gk():
    """ test reading of gk file.
    """
    data = cp.Statistics(ensemble="grand_canonical")
    data.from_mc(get_test_file())

    with pytest.raises(AssertionError):
        data = cp.Statistics(ensemble="NPT")
        data.from_mc(get_test_file())
