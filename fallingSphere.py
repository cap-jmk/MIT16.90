import numpy as np
import numba
import matplotlib.pyplot as plt

#TODO parallelize

def integrate(grid, timestep, initial_velocity):
    results = np.zeros(len(grid))
    results[0] = initial_velocity
    print("initial velocity is", initial_velocity)
    for i in range(1, len(grid)-1):
        results[i+1] = results[i] + timestep * calculate_function(x_val=results[i])
    return results

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
    
    :return: 
    :rtype: 
    """
    grid = np.arange(start= initial_value, stop=end_value, step=step)
    return grid

def plot_results(grid, integration):
    plt.plot(grid, integration)
    plt.xlabel("Time t [s]")
    plt.ylabel("Velocity v")
    plt.savefig("forward_euler.png", dpi=300)

if __name__ == '__main__':
    initial_time = 0
    intial_velocity = 0.001
    end_value = 25
    timestep = 0.25
    grid = discretize_time(initial_value=initial_time, end_value=end_value, step=0.25)
    integration = integrate(grid, timestep=timestep, initial_velocity=intial_velocity)
    plot_results(grid=grid, integration=integration)
