import pytest
import numpy as np
from MIT1690.fallingSphere import (discretize_time, approximate_reynolds, discretize_time,
                                   integrate_forward_euler,integrate_midpoint_rule, plot_results, forward_euler)

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
    print(grid)
    compare = np.linspace(0, 25, int((end_value-initial_value)/step)+1)
    assert np.allclose(grid, compare)

def test_reynolds():
    """
    Tests if the Reynolds function works
    :return:
    :rtype:
    """
    density, velocity, radius, mu_g = 9, 0.001, 0.01, 1.5E-5
    re = approximate_reynolds(density = density, velocity= velocity, radius=radius, mu_g=mu_g)
    assert re != 0

def test_plot_results():
    """
    Tests the plotting fuction
    :return:
    :rtype:
    """

def test_forward_euler():
    """
    Tests the forward Euler integration scheme
    :return:
    :rtype:
    """
    initial_time = 0
    initial_velocity = 0
    end_value = 25
    timestep = 0.25
    grid = discretize_time(initial_value=initial_time, end_value=end_value, step=0.25)
    integration = integrate_forward_euler(grid, timestep, initial_velocity=initial_velocity,
                                          function = linear_explicit, numerical_method = forward_euler)
    analytical_solution = analytical_linear_function(grid)
    assert np.allclose(analytical_solution, integration)
    plot_results(grid=grid, integration=integration)

def test_linear_explicit():
    initial_time=0
    initial_velocity=0
    end_value=25
    timestep=0.25
    grid = discretize_time(initial_value=initial_time, end_value=end_value, step=0.25)
    result = np.zeros(len(grid))
    for i in range(len(grid)-1):
        result[i+1] = result[i] + 0.25 * linear_explicit(result[i])
    assert np.allclose(result, analytical_linear_function(grid))

def test_midpoint_rule():
    """
    Tests the midpoint rule
    :return:
    :rtype:
    """

def linear_explicit(x_val):
    """
    differntial equation for a linear function should be always constant
    :return:
    :rtype:
    """
    return 3

def analytical_linear_function(grid):
    """
    Provides a simple analytical test for the numerical schemes
    :return:
    :rtype:
    """
    return grid*3
