import numpy as np
import matplotlib.pyplot as plt

from configs import initialize_constants
from solver import solve_robot
from equations_mode1 import equations
import plotter
from optimizer import *
from utils import tendon_disp


def main():
    config = initialize_constants()

    measured_Ft = [0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                    31.90615836, 38.47507331, 47.85923754]
    Ft_values = np.append(np.linspace(0, 60, 200), measured_Ft)
    Ft_values = np.sort(Ft_values)

    # Finding theta angles for the Ft range
    theta_solutions = solve_robot(config, Ft_values, equations, elastic_model="non-linear")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    for Ft in measured_Ft:
        index = np.where(Ft_values == Ft)[0][0]
        theta_plot = theta_solutions[index]
        plotter.plot_robot(theta_plot, axs[2], Ft, config)

    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    plotter.plot_measured_robot(measured_directory, axs[2])

    plt.tight_layout()
    plt.show()

    # Plot the error of the max displacement force
    index = np.where(Ft_values == 47.85923754)[0][0]
    theta_plot = theta_solutions[index]
    error_summary = plotter.measure_error(theta_plot, 47.85923754, config, measured_directory)
    print(f"Mean Error: {error_summary['mean_error']}")
    print(f"Max Error: {error_summary['max_error']}")


    # Plot Ft vs tendon disp
    tendon_displacements = [tendon_disp(theta_solutions[i], config) for i in range(len(Ft_values))]
    plt.figure(figsize=(8, 6))
    plt.plot(tendon_displacements, Ft_values, label="Tendon Displacement", color="b")
    plt.ylabel("$F_t$")
    plt.xlabel("Tendon Displacement")
    plt.title("Tendon Displacement vs $F_t$")
    plt.grid(True)
    plt.legend()
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

    # # Model Optimization
    # R1_bounds = (5e-3, 8e-3)
    # R2_bounds = (2.5e-3, 7e-3)
    # E_bounds = (5e9, 10e9)
    # A1_bounds = (3e-3, 3.6e-3)
    # A2_bounds = (1e-3, 1.6e-3)
    # mu_bounds = (0.05, 0.4)
    # k_bounds = (1, 15)
    # d_bounds = (0.2e-3, 4e-3)
    # elastic_model = "non-linear"
    # optimal_params, optimized_error = optimize_model(
    #     config, measured_Ft, measured_directory, elastic_model, R1_bounds, R2_bounds, E_bounds, A1_bounds, A2_bounds, mu_bounds
    # )
    # print(f"Optimal R1: {optimal_params[0] * 1000}mm")
    # print(f"Optimal R2: {optimal_params[1] * 1000}mm")
    # print(f"Optimal E: {optimal_params[2]}")
    # print(f"Optimal A1: {optimal_params[3] * 1000}mm")
    # print(f"Optimal A2: {optimal_params[4] * 1000}mm")
    # print(f"Optimal mu: {optimal_params[5]}")
    # if elastic_model == "linear":
    #     print(f"Optimal k: {optimal_params[6]}")
    # elif elastic_model == "non-linear-L":
    #     print(f"Optimal d: {optimal_params[6]}")
    # print(
    #     f"Optimized Error: {optimized_error}"
    # )

if __name__ == "__main__":
    main()
