import numpy as np
import matplotlib.pyplot as plt
from config import initialize_constants
from solver import solve_robot, initial_guess_gen
from equations_mode1 import equations
import plotter
from optimizer import *
from utils import tendon_disp


def main():
    config = initialize_constants()

    measured_Ft = [1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                    31.90615836, 38.47507331, 47.85923754]
    Ft_values = np.append(np.arange(1, 101, 1), measured_Ft)
    Ft_values = np.sort(Ft_values)

    # Finding theta angles for the Ft range
    theta_solutions = solve_robot(config, Ft_values, equations, elastic_model="non-linear")

    print(tendon_disp(theta_solutions[49], config))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    for Ft in measured_Ft:
        index = np.where(Ft_values == Ft)[0][0]
        theta_plot = theta_solutions[index]
        plotter.plot_robot(theta_plot, axs[2], Ft, config["L"])

    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    plotter.plot_measured_robot(measured_directory, axs[2])

    plt.tight_layout()
    plt.show()

    # Plot the error of the max displacement force
    index = np.where(Ft_values == 47.85923754)[0][0]
    theta_plot = theta_solutions[index]
    error_summary = plotter.measure_error(theta_plot, 47.85923754, config["L"], measured_directory)
    print(f"Mean Error: {error_summary['mean_error']}")
    print(f"Max Error: {error_summary['max_error']}")


    # Plot Ft vs tendon disp
    tendon_displacements = [tendon_disp(theta_solutions[i], config) for i in range(len(Ft_values))]
    plt.figure(figsize=(8, 6))
    plt.plot(tendon_displacements, Ft_values, label="Tendon Displacement", color="b")
    plt.xlabel("$F_t$")
    plt.ylabel("Tendon Displacement")
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

    R2_bounds = (2.67e-3, 7e-3)
    E_bounds = (5e9, 70e9)
    optimal_params, optimized_error = optimize_model(
        config, measured_Ft, R2_bounds, E_bounds, measured_directory, elastic_model="non-linear"
    )
    print(f"Optimal R2: {optimal_params[0] * 1000}mm")
    print(f"Optimal E: {optimal_params[1]}")
    print(
        f"Optimized Error: {optimized_error}"
    )


if __name__ == "__main__":
    main()
