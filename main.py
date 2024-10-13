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

    # Finding theta angles for the Ft range
    theta_solutions = solve_robot(config, Ft_values, equations, elastic_model="linear")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    for Ft in [20, 30, 40, 60, 80, 100]:
        theta_plot = theta_solutions[Ft - 1]

        plotter.plot_robot(theta_plot, axs[2], Ft, config["L"])

    plt.tight_layout()
    plt.show()

    # # Design Optimization
    # R2_bounds = (2.43e-3, 7e-3)
    # L_bounds = (7e-3, 18e-3)
    # optimal_params, optimized_theta2 = optimize_robot(
    #     config, Ft_values, R2_bounds, L_bounds, elastic_model="linear"
    # )
    # print(f"Optimal R2: {optimal_params[0] * 1000:.2f}mm")
    # print(f"Optimal L: {optimal_params[1] * 1000}mm")
    # print(
    #     f"Optimal Maximum Buckling of joint 2: {np.degrees(optimized_theta2)} degrees"
    # )


if __name__ == "__main__":
    main()
