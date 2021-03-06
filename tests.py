import pytest
import numpy as np
from MITAerEngineering.fallingSphere import discretize_time, approximate_reynolds

def test_grid():
    """
    tests the evenly spaced grid
    :return:
    :rtype:
    """
    initial_value = 0
    end_value = 25
    step = 0.25
    grid = discretize_time(initial_value = initial_value, end_value=end_value, step=step)
    assert np.allclose(grid, np.arange(0, 25, 0.25))

def test_reynolds():
    """
    Tests if the Reynolds function works
    :return:
    :rtype:
    """
    density, velocity, radius, mu_g = 9, 0.001, 0.01, 1.5E-5
    re = approximate_reynolds(density = density, velocity= velocity, radius=radius, mu_g=mu_g)
    assert re != 0
