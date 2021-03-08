import numpy as np


def find_best_multistep(matrix, b):
    """
    finds the best multistep for a given taylor expansion
    :param matrix:
    :type matrix:
    :return:
    :rtype:
    """
    return np.linalg.lstsq(matrix, b)


def calculate_error(approximation, base_values):
    """
    calculates the error of a solution
    :param approximation:
    :type approximation:
    :param base_values:
    :type base_values:
    :return:
    :rtype: np.ndarray
    """
    return np.abs(approximation-base_values)


def comparison_function(x_val):
    """
    returns the explicit comparison function f(u,t)=u^2
    :param grid:
    :type grid:
    :return:
    :rtype:
    """
    return -x_val**(2)

def analytical_solution_comparison(grid):
    """
    calculates the analytical solution to the comparison function f(u,t) = u^2,
    u = (1+t)^(-1)
    :param grid: array
    :type grid:
    :return: solution
    :rtype:
    """
    return 1/(1+grid)
