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
    

def optimize_model(config, Ft_values, R_2_bounds, E_bounds, measured_directory, *, elastic_model):

    x_measured, y_measured = utils.read_measurements(measured_directory, test_num="max_disp")
    x_measured = x_measured[0]
    y_measured = y_measured[0]

    def objective_function(params):
        config = initialize_constants_optimization(params[0], params[1])

        theta = solve_robot(config, Ft_values, equations, elastic_model= elastic_model)[-1]

        modeled_x, modeled_y = utils.theta_to_xy(theta, config['L'])

        if x_measured is None or len(modeled_x) != len(x_measured):
            raise ValueError("Mismatch in data length or missing measurement data.")
    
        errors = [
            np.sqrt((mx - x)**2 + (my - y)**2)
            for mx, my, x, y in zip(modeled_x, modeled_y, x_measured, y_measured)
        ]

        mean_error = np.mean(errors)

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