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

def integrate_best_two_point_multistep(grid, timestep, initial_velocity, function, numerical_method_t0):
    """
    Just to test a given diffential equation with the "best multistep"
    :return:
    :rtype:
    """
    results=np.zeros(len(grid))
    results[0]=initial_velocity
    print("initial velocity is", initial_velocity)
    for i in range(0, len(grid) - 1):
        if i == 0:
            results[i + 1]=numerical_method_t0(prev_value=results[i],
                                               timestep=timestep,
                                               function=function)
        else:
            results[i + 1]=best_two_point(prev_value=results[i - 1],
                                         current_value=results[i],
                                         timestep=timestep,
                                         function=function)
    return results
def best_two_point(prev_value, current_value, timestep, function):
    """
    Just the besdt two point mehtod derived above
    :param prev_value:
    :type prev_value:
    :param current_value:
    :type current_value:
    :param timestep:
    :type timestep:
    :param function:
    :type function:
    :return:
    :rtype:
    """
    return -4*current_value+5*prev_value+timestep*(function(4*current_value)+function(2*prev_value))
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
