from MIT1690.implicit import (secant_method, jacobian, implicit_euler)
import numpy as np
def test_newton_raphson():
    """
    Test needs to test a known example of the newton raphson method
    :return:
    :rtype:
    """
    pass

def test_jacobian_constructor():
    """
    Tests the function constructing the Jacobian out of a given array for a system of ODEs
    :return:
    :rtype:
    """
    def func1(old_guess):
        """
        Helper function
        :return:
        :rtype:
        """
        return -1000 * old_guess[0] + 0*old_guess[1]

    def func2(old_guess):
        return old_guess[0] + np.sin(100*old_guess[1])

    initial_guess_1 = 1
    initial_guess_2 = 0
    points = [initial_guess_1, initial_guess_2]

    system_odes = [func1, func2]
    jac = jacobian(system_odes, points)
    assert np.allclose(jac, np.array([[-1000,  0], [1,100]])), "The calculation of the Jacobian is not correct"


def test_secant_method():
    """
    The specs are: Must return correct jacobian at the points for an analytical functions with known jacobian at the
    point
    :return:
    :rtype:
    """

    def func1(old_guess):
        """
        Helper function
        :return:
        :rtype:
        """
        return -1000 * old_guess[0] + 0* old_guess[1]

    def func2(old_guess):
        return old_guess[0] + np.sin(100*old_guess[1])
    initial_guess_1 = 1
    initial_guess_2 = 0
    initial_guess = [initial_guess_1, initial_guess_2]
    a = secant_method(points=initial_guess, func = func1)
    b = secant_method(points = initial_guess, func=func2)
    assert np.allclose(a, [-1000,  0])
    assert np.allclose(b, [1,100])
