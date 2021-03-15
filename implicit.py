import numpy as np
from MIT1690.fallingSphere import discretize_time, forward_euler
from scipy import optimize
import numba



def implicit_euler(initial_conditions, time_interval, time_step, evaluation_function):
    """
    Implementation of the implcit (Backward Euler method) as asked in the script
    :param initial_conditions: vector of initial conditions
    :type initial_conditions:
    :param time_step: time_step
    :type time_step:
    :param grid: grid
    :type grid:
    :return:
    :rtype:
    """
    grid = discretize_time(initial_value=time_interval[0], end_value=time_interval[1], step=time_step)
    results_val1 = np.zeros(len(grid))
    results_val2 = np.zeros(len(grid))
    for i in range(0, len(grid)-1):
        results_val1[i+1] = results_val1[i]+time_step* evaluation_function()
        results_val2[i + 1]=results_val2[i] + time_step * evaluation_function()
    return [results_val1, results_val2]

def backward_euler(timestep, val, func):
    """

    :param timestep:
    :type timestep:
    :param val:
    :type val:
    :return:
    :rtype:
    """
    t_old = val
    t_new = newtons_method(derivative_fun=func, initial_guess=t_old)

def secant_method(points, func, epsilon = 10e-6):
    """
    Calculate the sekant method for a function to obtain the jacobian may use forward finite differences.
    :return:
    :rtype:
    """
    return optimize.approx_fprime(points, func, epsilon = epsilon)

def newtons_method(initial_guess,
                   system_of_odes,
                   derivative_fun=secant_method ,
                   numerical_itegrator=forward_euler,
                   time_step = 10e-5,
                   tol=10e-6):
    """
    The implicit function needs a numerical integrator because there is no way of knowing the value for
    a given ODE. If you do the examnple of a given analytical function and the numerical integrator represents the
    function that is divided by its derivative in the recusrive formula. It is quite important to realize the
    big difference from paper math to real calculations. It is almost never described in textbooks and lecture notes.
    :param initial_guess:passed values
    :type initial_guess:
    :param derivative_fun:
    :type derivative_fun:
    :param numerical_itegrator:
    :type numerical_itegrator:
    :param time_step:
    :type time_step:
    :param tol:
    :type tol:
    :return:
    :rtype:
    """
    new_val = numerical_itegrator(prev_value=initial_guess,  timestep = time_step, function=derivative_fun)
    err = np.abs(new_val, initial_guess)
    while err <= tol:
        old_guess = new_val.copy()
        new_val = old_guess - np.linalg.inv((np.eye(np.shape(system_of_odes))-derivative_fun(system_of_odes)))@(
            numerical_itegrator(prev_value=old_guess,
                                timestep = time_step,
                                function=derivative_fun))
        err = np.abs(new_val-old_guess)
    return new_val

def jacobian(system_of_odes, points, approximator = secant_method):
    """
    Approximates the Jacobian of a given System of ODEs
    :param system_of_odes: system of ODEs
    :type system_of_odes: array of objects
    :return: The Jacobian of the given system of ODEs
    :rtype: np.array
    """
    jac = np.zeros((len(points),len(points)))
    for i in range(len(points)):
        jac[i, :] = approximator(func = system_of_odes[i], points = points)
    return jac

def func(old_guess):
    """

    :param u:
    :type u:
    :param t:
    :type t:
    :return:
    :rtype:
    """
    u, t = old_guess
    return [-1000*u, np.sin(t)]
if __name__ == '__main__':
    initial_guess = [1,0]
    a = secant_method(old_guess=initial_guess)
    print(a)
