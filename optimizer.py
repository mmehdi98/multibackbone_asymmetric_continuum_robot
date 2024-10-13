import numpy as np
from scipy.optimize import minimize
from solver import solve_robot
from equations_mode1 import equations

def optimize_robot(config, Ft_values, R_2_bounds, L_bounds, * , elastic_model):
    
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
    

