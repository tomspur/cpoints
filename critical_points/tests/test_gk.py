# Copyright (c) 2017 Thomas Spura
"""
Test grand canonical simulation data
"""
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import critical_points as cp
from critical_points.tests import get_test_file


def test_read_gk():
    """ test reading of gk file.
    """
    data = cp.Statistics(ensemble="grand_canonical")
    data.from_mc(get_test_file())

    data2 = cp.read_mc(get_test_file())
    print(data.data.head())
    print(data2.data.head())
    assert np.sum(data.data - data2.data).all() == 0.0

    with pytest.raises(AssertionError):
        data = cp.Statistics(ensemble="NPT")
        data.from_mc(get_test_file())


def test_cumulants():
    """ Test cumulants of gk file.
    """
    data = cp.Statistics(ensemble="grand_canonical")
    data.from_mc(get_test_file())

    print("K2/K4:", data.K2, data.K4)
    assert_almost_equal(data.K2, 1.25963602195)
    assert_almost_equal(data.K4, 1.66009790754)

    data.extrapolate(data.temperature)
    print("K2/K4/mu:", data.K2, data.K4, data.rew_obs)
    assert_almost_equal(data.K2, 1.25957559223)
    assert_almost_equal(data.K4, 1.65968547893)
    assert_almost_equal(data.rew_obs, -2.83094834914)

    with pytest.raises(NotImplementedError):
        data.extrapolate(data.temperature, field_mixing=True)
