import numpy as np
from scipy.optimize import minimize

import utils
from configs import initialize_constants, initialize_constants_optimization
import solver2

def main():
    config = initialize_constants()
    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    measured_Ft = np.array([0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                29.09090909, 38.47507331, 47.85923754])
    R1_bounds = (5e-3, 8e-3)
    R2_bounds = (2e-3, 6e-3)
    E_bounds = (1e9, 70e9)
    A1_bounds = (2.5e-3, 4e-3)
    A2_bounds = (0.5e-3, 1.8e-3)
    mu_bounds = (0.01, 0.5)
    clearance_bounds = (0.2e-3, 0.6e-3)
    bounds = [R1_bounds, R2_bounds, E_bounds, A1_bounds, A2_bounds, mu_bounds, clearance_bounds]
    initial_params = np.array(
        [
            5.4e-3, #R1
            3e-3, #R2
            10e9, #E
            3.6e-3, #A1
            1.4e-3, #A2
            0.14, #mu
            0.4e-3, #clearance
        ]
    )
    optimal_params, optimized_error = optimize_model(
        config, measured_Ft, measured_directory, initial_params, bounds
    )
    print(f"Optimal R1: {optimal_params[0] * 1000}mm")
    print(f"Optimal R2: {optimal_params[1] * 1000}mm")
    print(f"Optimal E: {optimal_params[2]}")
    print(f"Optimal A1: {optimal_params[3] * 1000}mm")
    print(f"Optimal A2: {optimal_params[4] * 1000}mm")
    print(f"Optimal mu: {optimal_params[5]}")
    print(f"Optimal clearance = {optimal_params[6] * 1000}mm")

    print(
        f"Optimized Error: {optimized_error}"
    )

def optimize_model(
    config,
    measured_Ft,
    measured_directory,
    initial_params,
    bounds,
):

    x_measured, y_measured = utils.read_measurements(measured_directory)

    def objective_function(params):
        Ft_values = np.append(np.arange(0, measured_Ft[-1], 100), measured_Ft)
        Ft_values = np.sort(Ft_values)

        config = initialize_constants_optimization(params[0], params[1], params[2], params[3], params[4], params[5], params[6])
        theta, F_solutions = solver2.solve_robot(config, Ft_values)

        modeled_x, modeled_y = np.full_like(x_measured, 0), np.full_like(y_measured, 0)
        for i, Ft in enumerate(measured_Ft):
            index = np.where(Ft_values == Ft)[0][0]
            angles = theta[index]
            F = F_solutions[index]
            x_m, y_m = utils.theta_to_xy(angles, config, F)
            modeled_x[i] = x_m
            modeled_y[i] = y_m

        if (
            x_measured is None
            or len(modeled_x) != len(x_measured)
            or len(modeled_x[0]) != len(x_measured[0])
        ):
            raise ValueError("Mismatch in data length or missing measurement data.")

        errors = np.full_like(x_measured, 0)
        for i in range(len(x_measured)):
            error = np.full_like(x_measured[i], 0)
            for k, (mx, my, x, y) in enumerate(zip(modeled_x[i], modeled_y[i], x_measured[i], y_measured[i])):
                error[k] = np.sqrt((mx - x) ** 2 + (my - y) ** 2)/(config['L']*1000*(k+1))
            errors[i] = error

        mean_error = np.mean(np.array(errors).ravel())

        print(f"error: {mean_error}, mu:{config["mu"]}")

        return mean_error

    result = minimize(
        objective_function, initial_params, method="Nelder-Mead", bounds=bounds
    )

    return result.x, result.fun

if __name__ == "__main__":
    main()