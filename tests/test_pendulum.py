import matplotlib.pyplot as plt
from MIT1690.pendulum import (integrate_pendulum_feu, integrate_pendulum_mpr, model)
from MIT1690.fallingSphere import (discretize_time, approximate_reynolds, discretize_time,
                                   integrate_forward_euler, integrate_midpoint_rule, plot_results,
                                   forward_euler, midpoint_rule)


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
