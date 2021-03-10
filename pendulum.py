import numpy as np



def integrate_pendulum_mpr(grid, timestep, initial_values, model):
    variable_1 = np.zeros((len(grid)))
    variable_2 = np.zeros((len(grid)))
    variable_1[0]= initial_values[0]
    variable_2[0] = initial_values[1]
    for i in range(0, len(grid) - 1):
        if i == 0:
            calculate_new = model(variable_1[i],variable_2[i])
            variable_1[i+1] = variable_1[i] + timestep*calculate_new[0]
            variable_2[i+1] = variable_2[i] + timestep*calculate_new[1]
        else:
            calculate_new = model(variable_1[i],variable_2[i])
            variable_1[i+1] = variable_1[i-1] + 2*timestep*calculate_new[0]
            variable_2[i+1] = variable_2[i-1] + 2*timestep*calculate_new[1]
    return [variable_1, variable_2]


def integrate_pendulum_feu(grid, timestep, initial_values, model):
    variable_1 = np.zeros((len(grid)))
    variable_2 = np.zeros((len(grid)))
    variable_1[0]= initial_values[0]
    variable_2[0] = initial_values[1]
    for i in range(0, len(grid) - 1):
        calculate_new = model(variable_1[i],variable_2[i])
        variable_1[i+1] = variable_1[i] + timestep*calculate_new[0]
        variable_2[i+1] = variable_2[i] + timestep*calculate_new[1]
    return [variable_1, variable_2]

def model(omega_prev,theta_prev):
    """
    for 2 dimensional system the function needs to integrate 2 equations, hence there are two different
    coupled updates!
    :return:
    :rtype:
    """
    g = 9.81
    L = 1
    omega = -g/L*np.sin(theta_prev)
    theta = omega_prev
    return [omega, theta]
