import numpy as np
import matplotlib.pyplot as plt
from config import initialize_constants
from solver import solve_robot, initial_guess_gen
from equations_mode1 import equations
import plotter
from optimizer import optimize_robot

def main():
    config = initialize_constants()

    Ft_values = np.arange(1, 101, 1)

    theta_solutions = solve_robot(config, Ft_values, equations, elastic_model= 'linear')

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    for Ft in [20, 30, 40, 60, 80, 100]:
        theta_plot = theta_solutions[Ft-1]

        plotter.plot_robot(theta_plot, axs[2], Ft)

    plt.tight_layout()
    plt.show()

    # optimal_params, optimized_theta2 = optimize_robot(config, Ft_values, (10e-3, 15e-3))
    # print(f"Optimal Parameters: {optimal_params}")
    # print(f"Optimal Maximum Buckling of joint 2: {np.degrees(optimized_theta2)} degrees")

if __name__ == "__main__":
    main()