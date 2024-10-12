import numpy as np
from scipy.optimize import minimize
from solver import solve_robot
from equations_mode1 import equations

def optimize_robot(config, Ft_values, L_bounds):
    
    def objective_function(L):
        config["L"] = L

        theta2 = solve_robot(config, Ft_values, equations)[:,1]

        min_theta = np.min(theta2)

        return -min_theta

    bounds = [L_bounds]
    initial_params = np.array(L_bounds[0])

    result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds= bounds)

    return result.x, -result.fun
    

