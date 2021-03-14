import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from MIT1690.fallingSphere import (discretize_time, approximate_reynolds, discretize_time,
                                   integrate_forward_euler, integrate_midpoint_rule, plot_results,
                                   forward_euler, midpoint_rule)
from MIT1690.investigating import (find_best_multistep, calculate_error, comparison_function, analytical_solution_comparison,
                              integrate_best_two_point_multistep)
from MIT1690.pendulum import (integrate_pendulum_feu, integrate_pendulum_mpr, model)

from MIT1690.implicit import (secant_method, implicit_euler)
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)




def test_newton_raphson():
    """
    Test needs to test a known example of the newton raphson method
    :return:
    :rtype:
    """
    pass

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
        return -1000 * old_guess

    def func2(old_guess):
        return old_guess[0] + np.sin(100*old_guess[1])
    initial_guess = 1
    initial_guess2 = 0
    a = secant_method(old_guess=initial_guess, func = func1)
    b = secant_method(old_guess = [a,initial_guess2], func=func2)
    assert np.allclose(a, -1000)
    assert np.allclose(b, [1,100])


def test_pendulum_midpoint_rule():
    """

    :return:
    :rtype:
    """
    initial_time = 0
    end_time = 10
    initial_values = [0, np.pi/4]
    time_step = 0.02
    grid = discretize_time(initial_value=initial_time, end_value= end_time, step = time_step)
    results = integrate_pendulum_mpr(grid=grid, initial_values=initial_values, model=model, timestep=time_step)
    plt.plot(grid, results[0], label="$\omega$")
    plt.plot(grid, results[1], label="$\ttheta$")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("v  [m/s]")
    plt.savefig("plots/2nd_Order_Pendulum_MPR.png", dpi=300)
    plt.close()
def test_pendulum_forward_euler():
    """

    :return:
    :rtype:
    """
    initial_time = 0
    end_time = 10
    initial_values = [0, np.pi/4]
    time_step = 0.02
    grid = discretize_time(initial_value=initial_time, end_value= end_time, step = time_step)
    results = integrate_pendulum_feu(grid=grid, initial_values=initial_values, model=model, timestep=time_step)
    plt.plot(grid, results[0], label="$\omega$")
    plt.plot(grid, results[1], label="$\ttheta$")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("v  [m/s]")
    plt.savefig("plots/2nd_Order_Pendulum_FEU.png", dpi=300)
    plt.close()

def test_best_two_pint():
    """
    Tests the accuracy of forward Euler and plots the results
    :return:
    :rtype:
    """
    initial_time=0.001
    initial_velocity=1
    end_value=0.02
    timesteps=[0.001, 0.002, 0.004]
    markers = [".", "1", "h"]
    grids=[]
    integrations=[]
    analytical_solutions=[]
    for i in range(len(timesteps)):
        grids.append(discretize_time(initial_value=initial_time, end_value=end_value, step=timesteps[i]))
        integrations.append(integrate_best_two_point_multistep(grids[i], timesteps[i], initial_velocity=initial_velocity,
                                                    function=comparison_function, numerical_method_t0=forward_euler))
        analytical_solutions.append(analytical_solution_comparison(grids[i]))
        plt.plot(grids[i], integrations[i], markers[i], label= "$\Delta t$ = "+str(timesteps[i]))
    plt.plot(grids[i], analytical_solutions[i], color = "black", label="Analytical solution")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("v  [m/s]")
    plt.savefig("plots/best_two_point_compare_res.png", dpi=300)
    plt.close()
    for i in range(len(integrations)):
        plt.plot(grids[i], calculate_error(integrations[i], analytical_solutions[i]), markers[i],
                 label= "$\Delta t$ = "+str(timesteps[i]))
    plt.xlabel("t [s]")
    plt.ylabel("Integration error  [m/s]")
    plt.legend()
    plt.savefig("plots/best_two_point_errors.png", dpi=300)
    plt.close()

def test_find_best_multistep():
    """

    :return:
    :rtype:
    """
    matrix=np.array([[-1, -1, 0, 0, ],
                     [0, 1, 1, 1, ],
                     [0, -1 / 2, 0, -1, ],
                     [0, 1 / 6, 0, 1 / 2]])
    b=np.array([1, 1, 1 / 2, 1 / 6])
    result=find_best_multistep(matrix=matrix, b=b)
    assert np.allclose(result[0], np.array([4., -5., 4., 2.])), "Result is not correct"
    assert result[2] == 4, "Rank is not correct"
    assert np.allclose(result[3],
                       np.array([2.13460551, 1.23108946, 0.67337816, 0.04709258])), "Residuals are not correct"

def test_comparison_function():
    """

    :return:
    :rtype:
    """
    grid = [0.1, 0.2, 0.3]
    vals = []
    for point in grid:
        vals.append(comparison_function(point))
    assert np.allclose(np.array(vals), np.array([-0.1**(2), -0.2**(2), -0.3**(2)]))

def forward_euler_eigval_stability():
    """

    :return:
    :rtype:
    """

def test_forward_euler_accuracy():
    """
    Tests the accuracy of forward Euler and plots the results
    :return:
    :rtype:
    """
    initial_time=0.001
    initial_velocity=1
    end_value=10
    timesteps=[0.1, 0.2, 0.4]
    markers = [".", "1", "h"]
    grids=[]
    integrations=[]
    analytical_solutions=[]
    for i in range(len(timesteps)):
        grids.append(discretize_time(initial_value=initial_time, end_value=end_value, step=timesteps[i]))
        integrations.append(integrate_forward_euler(grids[i], timesteps[i], initial_velocity=initial_velocity,
                                                    function=comparison_function, numerical_method=forward_euler))
        analytical_solutions.append(analytical_solution_comparison(grids[i]))
        plt.plot(grids[i], integrations[i], markers[i], label= "$\Delta t$ = "+str(timesteps[i]))
    plt.plot(grids[i], analytical_solutions[i], color = "black", label="Analytical solution")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("v  [m/s]")
    plt.savefig("plots/forward_euler_compare_res.png", dpi=300)
    plt.close()
    for i in range(len(integrations)):
        plt.plot(grids[i], calculate_error(integrations[i], analytical_solutions[i]), markers[i],
                 label= "$\Delta t$ = "+str(timesteps[i]))
    plt.xlabel("t [s]")
    plt.ylabel("Integration error  [m/s]")
    plt.legend()
    plt.savefig("plots/forward_euler_errors.png", dpi=300)
    plt.close()

def test_forward_euler():
    """
    Tests the forward Euler integration scheme
    :return:
    :rtype:
    """
    initial_time=0
    initial_velocity=0
    end_value=2.25
    timestep=0.25
    grid=discretize_time(initial_value=initial_time, end_value=end_value, step=0.25)
    integration=integrate_forward_euler(grid, timestep, initial_velocity=initial_velocity,
                                        function=linear_explicit, numerical_method=forward_euler)
    analytical_solution=analytical_linear_function()
    assert np.allclose(analytical_solution, integration)
    plot_results(grid=grid, integration=integration, file_name="plots/forward_euler.png")


def test_linear_explicit():
    initial_time=0
    initial_velocity=0
    end_value=2.25
    timestep=0.25
    grid=discretize_time(initial_value=initial_time, end_value=end_value, step=timestep)
    result=np.zeros(len(grid))
    result[0]=initial_velocity
    for i in range(len(grid) - 1):
        result[i + 1]=result[i] + 0.25 * linear_explicit(result[i])
    assert np.allclose(result, analytical_linear_function())


def test_midpoint_rule_accuracy():
    """
    Tests the accuracy of forward Euler and plots the results
    :return:
    :rtype:
    """
    initial_time=0.001
    initial_velocity=1
    end_value=8
    timesteps=[0.1, 0.2, 0.4]
    markers = [".", "1", "h"]
    grids=[]
    integrations=[]
    analytical_solutions=[]
    for i in range(len(timesteps)):
        grids.append(discretize_time(initial_value=initial_time, end_value=end_value, step=timesteps[i]))
        integrations.append(integrate_midpoint_rule(grids[i], timesteps[i], initial_velocity=initial_velocity,
                                                    function=comparison_function, numerical_method_t0=forward_euler))
        analytical_solutions.append(analytical_solution_comparison(grids[i]))
        plt.plot(grids[i], integrations[i], markers[i], label= "$\Delta t$ = "+str(timesteps[i]))
    plt.plot(grids[i], analytical_solutions[i], color = "black", label="Analytical solution")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("v  [m/s]")
    plt.savefig("plots/midpoint_rule_compare_res.png", dpi=300)
    plt.close()
    for i in range(len(integrations)):
        plt.plot(grids[i], calculate_error(integrations[i], analytical_solutions[i]), markers[i],
                 label= "$\Delta t$ = "+str(timesteps[i]))
    plt.xlabel("t [s]")
    plt.ylabel("Integration error  [m/s]")
    plt.legend()
    plt.savefig("plots/midpoint_rule_errors.png", dpi=300)
    plt.close()

def test_midpoint_rule():
    """
    Tests the midpoint rule
    :return:
    :rtype:
    """
    initial_time=0
    initial_velocity=0
    end_value=2.25
    timestep=0.25
    grid=discretize_time(initial_value=initial_time, end_value=end_value, step=0.25)
    integration=integrate_midpoint_rule(grid, timestep, initial_velocity=initial_velocity,
                                        function=linear_explicit, numerical_method_t0=forward_euler)
    analytical_solution=analytical_linear_function()
    print(integration)
    assert np.allclose(analytical_solution, integration)
    plot_results(grid=grid, integration=integration, file_name="plots/midpoint_rule.png")


def test_grid():
    """
    tests the evenly spaced grid
    :return:
    :rtype:
    """
    initial_value=0
    end_value=25
    step=0.25
    grid=discretize_time(initial_value=initial_value, end_value=end_value, step=step)
    print(grid)
    compare=np.linspace(0, 25, int((end_value - initial_value) / step) + 1)
    assert np.allclose(grid, compare)


def test_reynolds():
    """
    Tests if the Reynolds function works
    :return:
    :rtype:
    """
    density, velocity, radius, mu_g=9, 0.001, 0.01, 1.5E-5
    re=approximate_reynolds(density=density, velocity=velocity, radius=radius, mu_g=mu_g)
    assert re != 0


def linear_explicit(x_val):
    """
    differntial equation for a linear function should be always constant
    :return:
    :rtype:
    """
    return 3


def analytical_linear_function():
    """
    Provides a simple analytical test for the numerical schemes
    :return:
    :rtype:
    """
    return [0., 0.75, 1.5, 2.25, 3., 3.75, 4.5, 5.25, 6., 6.75]
