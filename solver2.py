import numpy as np
from scipy.optimize import root
import utils

def solve_robot(
    config,
    Ft_values,
    method="hybr",
):
    theta_solutions = np.zeros((len(Ft_values), config["num"]))
    F_solutions = np.zeros((len(Ft_values), config["num"]))
    initial_guess = np.array([0]*(2*config["num"]-1))

    for i, Ft in enumerate(Ft_values):
        solution = root(
            lambda vars: equations(vars, config, Ft),
            initial_guess,
            method=method,
        )

        theta_solutions[i] = solution.x[: config["num"]]
        F_solutions[i] = np.append(solution.x[config["num"]:], Ft)
        initial_guess = np.array(solution.x)

        # if solution.success:
        #     theta_solutions.append(solution.x[:config["num"]])
        #     initial_guess = list(solution.x)
        # else:
        #     raise ValueError(f"Root-finding failed for Ft={Ft}: {solution.message}")

    return theta_solutions, F_solutions

def elastic_moment(E, I, L, th, R_left, R_right, alpha, betha):
    # th = abs(th)
    vertical_dist_left = (R_left * (1 - np.cos(alpha - th)))
    vertical_dist_right = (R_right * (1 - np.cos(betha + th)))

    return (
        (E * I) * np.sin(th)
        * (1 / (vertical_dist_left) 
            + 1 / (vertical_dist_right)
        )
    )

def equations(vars, constants, Ft):
    L = constants["L"]
    mu = constants["mu"]
    E = constants["E"]
    I = constants["I"]

    n = int(len(vars)/2)+1
    th_array = np.array(vars[0:n])
    Fr_array = np.array(vars[n:])
    Fr_array = np.insert(Fr_array, 0, 0)
    # phi_array = np.array([th if abs(th)>0.016 else 0 for th in th_array])
    phi_array = utils.calc_phi(th_array, constants)

    sum_phi_array = np.cumsum(phi_array)

    M = np.zeros(n)
    Fy = np.zeros(n)

    for i in range(n):
        th = th_array[i]
        # phi = phi_array[i]
        sum_phi = sum_phi_array[i]
        Fr = Fr_array[i]

        At = constants["At"][i]
        Ac = constants["Ac"][i]
        Lt = constants["Lt"][i]
        Lc = constants["Lc"][i]
        R_left = constants["R_left"][i]
        R_right = constants["R_right"][i]
        alpha = constants["alpha"][i]
        betha = constants["betha"][i]
        R = R_left if th >= 0 else R_right

        # The last joint
        if i != n-1:
            th_up = th_array[i+1]
            Fr_up = Fr_array[i+1]
            At_up = constants["At"][i+1]
            Ac_up = constants["Ac"][i+1]
            R_up = constants["R_left"][i+1] if th_array[i+1] >= 0 else constants["R_right"][i+1]

            # Protagonist joints
            if i & 1 == 0:
                M[i] = (
                    Ft*np.exp(-mu * sum_phi)*(1-np.exp(-mu * th))/mu
                    * (
                        Lt/2 + R_left*(1-np.cos(alpha)) - R*(1-np.cos(th)) + mu*(At - R*np.sin(th))
                    )
                    + Fr_up*(At - At_up + R_up*th_up - R*np.sin(th))
                    - elastic_moment(E, I, L, th, R_left, R_right, alpha, betha)
                )

                Fy[i] = (
                    Fr * np.cos(th)
                    - Fr_up
                    - Ft*np.exp(-mu * sum_phi)*(1-np.exp(-mu * th))
                )

            # Antagonist joints
            else:
                th = -th
                M[i] = abs(
                    Ft*np.exp(-mu * sum_phi)*(1-np.exp(-mu * th))/mu
                    * (
                        Lt/2 + R_left*(1-np.cos(alpha)) - R*(1-np.cos(th)) + mu*(At - R*np.sin(th))
                    )
                    + Fr_up*(Ac - Ac_up - R_up*th_up - R*np.sin(th))
                    - elastic_moment(E, I, L, th, R_left, R_right, alpha, betha)
                )

                Fy[i] = (
                    Fr * np.cos(th)
                    - Fr_up
                    - Ft*np.exp(-mu * sum_phi)*(1-np.exp(-mu * th))
                )

        else:
            M[i] = (
                Ft*np.exp(-mu * sum_phi)*(1-np.exp(-mu * th))/mu
                * (
                    Lt/2 + R_left*(1-np.cos(alpha)) - R*(1-np.cos(th)) + mu*(At - R*np.sin(th))
                )
                + Ft*np.exp(-mu * sum_phi)*(At - R*np.sin(th))
                - elastic_moment(E, I, L, th, R_left, R_right, alpha, betha)
            )

            Fy[i] = (
                Fr * np.cos(th)
                - Ft*np.exp(-mu * sum_phi)
            )



    return np.concatenate((M, Fy[1:]))
