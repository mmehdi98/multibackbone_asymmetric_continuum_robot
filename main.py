import numpy as np
import matplotlib.pyplot as plt
from config import initialize_constants
from solver import solve_robot, initial_guess_gen
from equations_mode1 import equations
import plotter

def main():
    config = initialize_constants()

    Ft_values = np.arange(1, 101, 1)

    th_p_guess = 0.4
    th_a_guess = 0.2

    theta_solutions = solve_robot(config, Ft_values, equations, th_p_guess= th_p_guess, th_a_guess= th_a_guess)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    for Ft in [20, 30, 40, 60, 80, 100]:
        theta_plot = solve_robot(config, [Ft], equations, th_p_guess= th_p_guess, th_a_guess= th_a_guess)

        plotter.plot_robot(theta_plot, axs[2], Ft)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()