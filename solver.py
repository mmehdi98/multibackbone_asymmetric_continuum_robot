import numpy as np
from scipy.optimize import root


def initial_guess_gen(n, Ft, th_p_guess=0.4, th_a_guess=0.1):
    theta_guess = [th_p_guess, th_a_guess] * int(n / 2) + [th_p_guess]
    Ft_guess = [Ft * (0.5 + 0.5 * np.exp(-0.1 * i)) for i in range(n - 1)]

    return theta_guess + Ft_guess


def solve_robot(
    config,
    Ft_values,
    equations,
    elastic_model,
    th_p_guess=0.5,
    th_a_guess=-0.1,
    method="hybr",
):
    theta_solutions = []
    initial_guess = initial_guess_gen(
        config["num"], 0, th_p_guess=th_p_guess, th_a_guess=th_a_guess
    )

    for Ft in Ft_values:
        solution = root(
            lambda vars: equations(vars, config, Ft, elastic_model),
            initial_guess,
            method=method,
        )

        theta_solutions.append(solution.x[: config["num"]])
        initial_guess = list(solution.x)

        # if solution.success:
        #     theta_solutions.append(solution.x[:config["num"]])
        #     initial_guess = list(solution.x)
        # else:
        #     raise ValueError(f"Root-finding failed for Ft={Ft}: {solution.message}")

    return np.array(theta_solutions)
