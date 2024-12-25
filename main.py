import numpy as np
import matplotlib.pyplot as plt

from configs import initialize_constants
import solver2
import plotter
import utils

def main():
    config = initialize_constants()

    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    measured_Ft = [0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 
                    13.13782991, 16.8914956, 22.52199413, 29.09090909, 38.47507331, 47.85923754]
    Ft_values = np.append(np.linspace(0, 60, 100), measured_Ft)
    Ft_values = np.sort(Ft_values)

    # Finding theta angles for the Ft range
    theta_solutions, F_solutions = solver2.solve_robot(config, Ft_values)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    modeled_x, modeled_y = [], []
    for Ft in measured_Ft:
        index = np.where(Ft_values == Ft)[0][0]
        theta_plot = theta_solutions[index]
        F_plot = F_solutions[index]
        plotter.plot_robot(theta_plot, axs[2], Ft, config, F_plot)

        x_m, y_m = utils.theta_to_xy(theta_plot, config, F_plot)
        modeled_x.append(x_m)
        modeled_y.append(y_m)
    x_measured, y_measured = utils.read_measurements(measured_directory)

    plotter.plot_measured_robot(config, measured_directory, axs[2])

    errors = np.full_like(x_measured, 0)
    for i in range(len(x_measured)):
        error = np.full_like(x_measured[i], 0)
        for k, (mx, my, x, y) in enumerate(zip(modeled_x[i], modeled_y[i], x_measured[i], y_measured[i])):
            error[k] = np.sqrt((mx - x) ** 2 + (my - y) ** 2)/(config['L']*1000*(k+1))
        errors[i] = error

    mean_error = np.mean(np.array(errors).ravel())
    print(f"last error: {errors[-1][-1]}")

    print(f"Mean Error: {mean_error}")

    plt.tight_layout()
    plt.show()

    # Plot Ft vs tendon disp
    tendon_displacements = [utils.tendon_disp(theta_solutions[i], config, F_solutions[i])*1000 for i in range(len(Ft_values))]
    plt.figure(figsize=(8, 6))
    plt.plot(tendon_displacements, Ft_values, label="Tendon Displacement", color="b")
    plt.ylabel("$F_t$")
    plt.xlabel("Tendon Displacement")
    plt.title("Tendon Displacement vs $F_t$")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
