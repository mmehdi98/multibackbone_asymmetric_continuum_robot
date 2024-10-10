import numpy as np
from scipy.optimize import fsolve


def initial_guess_gen(n, Ft, th_p_guess= 0.4, th_a_guess= 0.1):
    theta_guess = [th_p_guess, th_a_guess] * int(n/2) + [th_p_guess]
    Ft_guess = [Ft*(0.5+0.5*np.exp(-0.1*i)) for i in range(n-1)]

    return theta_guess+Ft_guess

def solve_robot(config, Ft_values, equations, initial_guess_func, th_p_guess= 0.4, th_a_guess= 0.1):
    theta_solutions = []

    for Ft in Ft_values:
        initial_guess = initial_guess_func(config["num"], Ft, th_p_guess= th_p_guess, th_a_guess= th_a_guess)

        solution = fsolve(lambda vars: equations(vars, config, Ft), initial_guess)

        theta_solutions.append(solution[:config["num"]]) 

    return np.array(theta_solutions)