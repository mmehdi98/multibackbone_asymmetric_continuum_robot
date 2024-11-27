import numpy as np
from scipy.optimize import minimize
from solver import solve_robot
from equations_mode1 import equations
import utils
import matplotlib.pyplot as plt
from config import initialize_constants_optimization

def optimize_buckling(config, Ft_values, R_2_bounds, L_bounds, * , elastic_model):
    
    def objective_function(params):
        config["R_2"] = params[0]
        config["L"] = params[1]

        theta2 = solve_robot(config, Ft_values, equations, elastic_model= elastic_model)[:,1]

        min_theta = np.min(theta2)

        return -min_theta

    bounds = [R_2_bounds, L_bounds]
    initial_params = np.array([R_2_bounds[0], L_bounds[0]])

    result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds= bounds)

    return result.x, -result.fun
    

def optimize_model(config, measured_Ft, R_2_bounds, E_bounds, measured_directory, *, elastic_model):

    x_measured, y_measured = utils.read_measurements(measured_directory)

    modeled_x, modeled_y = [[15*i for i in range(16)]], [[0]*16]
    diff = []
    for i in range(len(x_measured)):
        diff.append([(x_measured[i][k]-modeled_x[0][k])**2 + (y_measured[i][k]-modeled_y[0][k])**2 for k in range(len(x_measured[i]))])

    def objective_function(params):
        config = initialize_constants_optimization(params[0], params[1])
        Ft_values = (np.append(np.arange(1, measured_Ft[-1], 50), measured_Ft))
        Ft_values = np.sort(Ft_values)

        theta = solve_robot(config, Ft_values, equations, elastic_model= elastic_model)

        modeled_x, modeled_y = [[15*i for i in range(16)]], [[0]*16]
        for Ft in measured_Ft:
            index = np.where(Ft_values == Ft)[0][0]
            angles = theta[index]
            x_m, y_m = utils.theta_to_xy(angles, config['L'])
            modeled_x.append(x_m)
            modeled_y.append(y_m)

        if x_measured is None or len(modeled_x) != len(x_measured) or len(modeled_x[0]) != len(x_measured[0]):
            raise ValueError("Mismatch in data length or missing measurement data.")

        errors = []
        for i in range(len(x_measured)):
            for mx, my, x, y, d in zip(modeled_x[i], modeled_y[i], x_measured[i], y_measured[i], diff[i]):
                error = []
                if (d-0) < 0.0005:
                    error.append(0)
                else:
                    error.append(np.sqrt((mx - x)**2 + (my - y)**2)/d)
            errors.append(error)

        mean_error_list = []
        for error in errors:
            mean_error_list.append(np.mean(errors))

        mean_error = np.mean(mean_error_list)

        print(mean_error)

        return mean_error

    bounds = [R_2_bounds, E_bounds]
    initial_params = np.array([R_2_bounds[0], E_bounds[0]])

    result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds= bounds)

    return result.x, result.fun

# def optimize_model(config, Ft_values, R_2_bounds, measured_directory, *, elastic_model):
#     # Read measured data
#     x_measured, y_measured = utils.read_measurements(measured_directory, test_num="max_disp")
#     x_measured = x_measured[0]
#     y_measured = y_measured[0]

#     def plot_robot(theta, modeled_x, modeled_y, iteration, error):
#         """Plot the robot configuration during optimization."""
#         plt.figure(figsize=(8, 6))
#         plt.plot(x_measured, y_measured, '-o', label='Measured Trajectory', color='orange')
#         plt.plot(modeled_x, modeled_y, '-o', label='Modeled Trajectory', color='blue')
#         plt.title(f"Iteration {iteration} | Error: {error:.4f}")
#         plt.xlabel('X Position (mm)')
#         plt.ylabel('Y Position (mm)')
#         plt.axis('equal')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     def objective_function(params):
#         config = initialize_constants_optimization(params[0])

#         # Solve for joint angles based on the current configuration
#         theta = solve_robot(config, Ft_values, equations, elastic_model=elastic_model)[-1]

#         # Convert joint angles to X-Y positions
#         modeled_x, modeled_y = utils.theta_to_xy(theta, config['L'])

#         if x_measured is None or len(modeled_x) != len(x_measured):
#             raise ValueError("Mismatch in data length or missing measurement data.")
    
#         # Calculate errors
#         errors = [
#             np.sqrt((mx - x)**2 + (my - y)**2)
#             for mx, my, x, y in zip(modeled_x, modeled_y, x_measured, y_measured)
#         ]

#         # Plot the robot configuration for visualization
#         iteration = objective_function.iteration
#         plot_robot(theta, modeled_x, modeled_y, iteration, np.mean(errors))
#         print(config["R_2"])
#         objective_function.iteration += 1

#         return np.mean(errors)

#     # Initialize iteration counter
#     objective_function.iteration = 1

#     # Set up optimization
#     bounds = [R_2_bounds]
#     initial_params = np.array([R_2_bounds[0]])

#     result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds=bounds)

#     return result.x, result.fun