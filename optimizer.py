import numpy as np
from scipy.optimize import minimize

# from solver import solve_robot
# from equations_mode1 import equations
import utils
import matplotlib.pyplot as plt
from configs import initialize_constants_optimization
from optimizer_equations import *


def optimize_model(
    config,
    measured_Ft,
    measured_directory,
    elastic_model,
    R_1_bounds,
    R_2_bounds,
    E_bounds,
    A1_bounds,
    A2_bounds,
    mu_bounds,
    k_bounds=(1, 15), 
    d_bounds=(0.2e-3, 4e-3),
):

    x_measured, y_measured = utils.read_measurements(measured_directory)

    modeled_x, modeled_y = [[15 * i for i in range(16)]], [[0] * 16]
    diff = []
    for i in range(len(x_measured)):
        diff.append(
            [
                (x_measured[i][k] - modeled_x[0][k]) ** 2
                + (y_measured[i][k] - modeled_y[0][k]) ** 2
                for k in range(len(x_measured[i]))
            ]
        )

    def objective_function(params):
        Ft_values = np.append(np.arange(0, measured_Ft[-1], 100), measured_Ft)
        Ft_values = np.sort(Ft_values)

        if elastic_model == "non-linear":
            config = initialize_constants_optimization(
                params[0], params[1], params[2], params[3], params[4], params[5]
            )
            theta = solve_robot(
                config, Ft_values, equations, elastic_model=elastic_model
            )
        elif elastic_model == "non-linear-L":
            config = initialize_constants_optimization(
                params[0], params[1], params[2], params[3], params[4], params[5]
            )
            theta = solve_robot(
                config, Ft_values, equations, elastic_model=elastic_model, d=params[6]
            )
        elif elastic_model == "linear":
            config = initialize_constants_optimization(
                params[0], params[1], params[2], params[3], params[4], params[5]
            )
            theta = solve_robot(
                config, Ft_values, equations, elastic_model=elastic_model, k=params[6]
            )
        else:
            raise ValueError("Elastic model not recognized")

        modeled_x, modeled_y = [], []
        for Ft in measured_Ft:
            index = np.where(Ft_values == Ft)[0][0]
            angles = theta[index]
            x_m, y_m = utils.theta_to_xy(angles, config)
            modeled_x.append(x_m)
            modeled_y.append(y_m)

        if (
            x_measured is None
            or len(modeled_x) != len(x_measured)
            or len(modeled_x[0]) != len(x_measured[0])
        ):
            raise ValueError("Mismatch in data length or missing measurement data.")

        errors = []
        for i in range(10, len(x_measured)):
            for mx, my, x, y, d in zip(
                modeled_x[i], modeled_y[i], x_measured[i], y_measured[i], diff[i]
            ):
                error = []
                if (d - 0) < 0.005:
                    error.append(0)
                else:
                    error.append(np.sqrt((mx - x) ** 2 + (my - y) ** 2)/(config['L']*1000*15))
            errors.append(error)

        mean_error_list = []
        for error in errors:
            mean_error_list.append(np.mean(errors))

        mean_error = np.mean(mean_error_list)

        print(f"error: {mean_error}, mu:{config["mu"]}")

        return mean_error

    match elastic_model:
        case "non-linear":
            bounds = [R_1_bounds, R_2_bounds, E_bounds, A1_bounds, A2_bounds, mu_bounds]
            initial_params = np.array(
                [
                    R_1_bounds[0],
                    R_2_bounds[0],
                    E_bounds[0],
                    A1_bounds[0],
                    A2_bounds[0],
                    mu_bounds[0],
                ]
            )
        case "non-linear-L":
            bounds = [
                R_1_bounds,
                R_2_bounds,
                E_bounds,
                A1_bounds,
                A2_bounds,
                mu_bounds,
                d_bounds,
            ]
            initial_params = np.array(
                [
                    R_1_bounds[0],
                    R_2_bounds[0],
                    E_bounds[0],
                    A1_bounds[0],
                    A2_bounds[0],
                    mu_bounds[0],
                    d_bounds[0],
                ]
            )
        case "linear":
            bounds = [
                R_1_bounds,
                R_2_bounds,
                E_bounds,
                A1_bounds,
                A2_bounds,
                mu_bounds,
                k_bounds,
            ]
            initial_params = np.array(
                [
                    R_1_bounds[0],
                    R_2_bounds[0],
                    E_bounds[0],
                    A1_bounds[0],
                    A2_bounds[0],
                    mu_bounds[0],
                    k_bounds[0],
                ]
            )

    result = minimize(
        objective_function, initial_params, method="Nelder-Mead", bounds=bounds
    )

    return result.x, result.fun


# def optimize_buckling(config, Ft_values, R_2_bounds, L_bounds, * , elastic_model):

#     def objective_function(params):
#         config["R_2"] = params[0]
#         config["L"] = params[1]

#         theta2 = solve_robot(config, Ft_values, equations, elastic_model= elastic_model)[:,1]

#         min_theta = np.min(theta2)

#         return -min_theta

#     bounds = [R_2_bounds, L_bounds]
#     initial_params = np.array([R_2_bounds[0], L_bounds[0]])

#     result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds= bounds)

#     return result.x, -result.fun
