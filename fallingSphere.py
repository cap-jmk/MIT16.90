import numpy as np
import numba
import matplotlib.pyplot as plt

#TODO parallelize


def integrate_midpoint_rule(grid, timestep, initial_velocity, function, numerical_method_t0):
    """
    Calculates midpoint rule
    :param prev_value: value at n-1
    :type prev_value:
    :param time_step: time_step
    :type time_step:
    :param function: explicit function
    :type function:
    :return: new value
    :rtype:
    """
    results=np.zeros(len(grid))
    results[0]=initial_velocity
    print("initial velocity is", initial_velocity)
    for i in range(1, len(grid) - 1):
        if i == 0:
            results[i + 1]=numerical_method_t0(prev_value=results[i],
                                            timestep=timestep,
                                            function=function)
        else:
            results[i + 1] = midpoint_rule(prev_value=results[i-1],
                                           timestep=timestep,
                                           function=function)
    return results

def midpoint_rule(prev_value,  timestep, function):
    """
    Just the formula to the midpoint rule
    :param prev_value: t_n-1 is something else than in forward euler, first script I see, ecplicitly stating this
    :type prev_value:
    :param timestep: times_step
    :type timestep:
    :param function: explicit function
    :type function:
    :return:
    :rtype:
    """
    return prev_value *2*timestep*function(x_val =prev_value)

def integrate_forward_euler(grid, timestep, initial_velocity, function, numerical_method):
    """
    integration scheme for forward euler method
    :param grid:
    :type grid:
    :param timestep:
    :type timestep:
    :param initial_velocity:
    :type initial_velocity:
    :param function:
    :type function:
    :param numerical_method:
    :type numerical_method:
    :return:
    :rtype:
    """
    results = np.zeros(len(grid))
    results[0] = initial_velocity
    print("initial velocity is", initial_velocity)
    for i in range(0, len(grid)-1):
        results[i+1] = numerical_method(prev_value = results[i],
                                                                timestep= timestep,
                                                                function=function)
    return results

def forward_euler(prev_value,  timestep, function):
    """
    Just the formula for the forward euler method
    :param prev_value:
    :type prev_value:
    :param timestep:
    :type timestep:
    :param function:
    :type function:
    :return:
    :rtype:
    """
    return  prev_value + timestep * function(x_val=prev_value)

def calculate_function(x_val):
    """
    calculates the function for the discretized timestep
    :return:
    :rtype:
    """
    density = 0.9
    radius = 0.01
    velocity = x_val
    mu_g = 1.99E-5
    g = 9.8
    mass_particle = 0.0038
    reynolds = approximate_reynolds(density=density, velocity=velocity, mu_g= mu_g, radius=radius)
    drag_coefficient = approximate_drag_coefficient(Re=reynolds)
    return g - 1/(mass_particle*2) * approximate_D(density=density, radius=radius, velocity=velocity, drag_coefficient=drag_coefficient)

def approximate_drag_coefficient(Re):
    """

    :param Re:
    :type Re:
    :return:
    :rtype:
    """
    print("Reynolds", Re)
    if Re == 0:
        C_d = 0
    else:
        C_d = 24/Re + 6/(1+Re**(1/2))+0.4

    return C_d

def approximate_D(density, radius, velocity, drag_coefficient):
    D = 2 * density *np.pi*radius**2*velocity**2*drag_coefficient
    return D

def approximate_reynolds(density, velocity, radius, mu_g):
    Re =2 * density * velocity * radius / mu_g
    return Re

def discretize_time(initial_value, end_value, step = 0.25):
    """

    :param initial_value:
    :type initial_value:
    :param end_value:
    :type end_value:
    :param step:
    :type step:
    :return:
    :rtype:
    """
    grid = np.linspace(initial_value, end_value, int((end_value-initial_value)/step)+1)
    return grid

def plot_results(grid, integration):
    """

    :param grid:
    :type grid:
    :param integration:
    :type integration:
    :return:
    :rtype:
    """
    plt.plot(grid, integration)
    plt.xlabel("Time t [s]")
    plt.ylabel("Velocity v [m/s]")
    plt.savefig("forward_euler.png", dpi=300)
