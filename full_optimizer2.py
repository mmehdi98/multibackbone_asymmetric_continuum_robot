import numpy as np
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt

import utils
from configs import initialize_constants

def main():
    config = initialize_constants()
    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    measured_Ft = [0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                29.09090909, 38.47507331, 47.85923754]
    R1_bounds = (5e-3, 8e-3)
    R2_bounds = (2e-3, 5e-3)
    E_bounds = (1e9, 35e9)
    A1_bounds = (2.8e-3, 4e-3)
    A2_bounds = (0.6e-3, 1.6e-3)
    mu_bounds = (0.01, 0.5)
    clearance_bounds = (0.2e-3, 0.5e-3)
    k_bounds = (1, 15)
    d_bounds = (0.2e-3, 4e-3)
    elastic_model = "non-linear"
    optimal_params, optimized_error = optimize_model(
        config, measured_Ft, measured_directory, elastic_model, R1_bounds, R2_bounds, E_bounds, A1_bounds, A2_bounds, mu_bounds, clearance_bounds
    )
    print(f"Optimal R1: {optimal_params[0] * 1000}mm")
    print(f"Optimal R2: {optimal_params[1] * 1000}mm")
    print(f"Optimal E: {optimal_params[2]}")
    print(f"Optimal A1: {optimal_params[3] * 1000}mm")
    print(f"Optimal A2: {optimal_params[4] * 1000}mm")
    print(f"Optimal mu: {optimal_params[5]}")
    print(f"Optimal clearance = {optimal_params[6] * 1000}mm")
    if elastic_model == "linear":
        print(f"Optimal k: {optimal_params[7]}")
    elif elastic_model == "non-linear-L":
        print(f"Optimal d: {optimal_params[7]}")
    print(
        f"Optimized Error: {optimized_error}"
    )

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
    clearance_bounds,
    k_bounds=(1, 15), 
    d_bounds=(0.2e-3, 4e-3),
):

    x_measured, y_measured = utils.read_measurements(measured_directory)
    # theta_measured = utils.xy_to_theta(x_measured, y_measured)
    # for i in range(len(x_measured)):
    #     x_measured[i], y_measured[i] = utils.th2xy_measurements(theta_measured[i], config["L"])

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
            theta, F_solutions = solve_robot(
                config, Ft_values, equations, elastic_model=elastic_model, clearance=params[6]
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
            F = F_solutions[index]
            x_m, y_m = theta_to_xy(angles, config, F)
            modeled_x.append(x_m)
            modeled_y.append(y_m)

        if (
            x_measured is None
            or len(modeled_x) != len(x_measured)
            or len(modeled_x[0]) != len(x_measured[0])
        ):
            raise ValueError("Mismatch in data length or missing measurement data.")

        errors = []
        for i in range(len(x_measured)):
            for k, (mx, my, x, y) in enumerate(zip(modeled_x[i], modeled_y[i], x_measured[i], y_measured[i])):
                error = []
                error.append(np.sqrt((mx - x) ** 2 + (my - y) ** 2)/(config['L']*1000*(k+1)))
            errors.append(error)

        # mean_error_list = []
        # for error in errors:
        #     mean_error_list.append(np.mean(errors))

        # mean_error = np.mean(mean_error_list)

        mean_error = np.mean(np.array(errors).ravel())

        print(f"error: {mean_error}, mu:{config["mu"]}")

        return mean_error

    match elastic_model:
        case "non-linear":
            bounds = [R_1_bounds, R_2_bounds, E_bounds, A1_bounds, A2_bounds, mu_bounds, clearance_bounds]
            # initial_params = np.array(
            #     [
            #         (R_1_bounds[0]+R_1_bounds[1])/2,
            #         (R_2_bounds[0]+R_2_bounds[1])/2,
            #         (E_bounds[0]+E_bounds[1])/2,
            #         (A1_bounds[0]+A1_bounds[1])/2,
            #         (A2_bounds[0]+A2_bounds[1])/2,
            #         (mu_bounds[0]+mu_bounds[1])/2,
            #     ]
            initial_params = np.array(
                [
                    5.4e-3, #R1
                    3e-3, #R2
                    10e9, #E
                    3.6875e-3, #A1
                    1.308859e-3, #A2
                    0.16089, #mu
                    0.4, #clearance
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
                    (R_1_bounds[0]+R_1_bounds[1])/2,
                    (R_2_bounds[0]+R_2_bounds[1])/2,
                    (E_bounds[0]+E_bounds[1])/2,
                    (A1_bounds[0]+A1_bounds[1])/2,
                    (A2_bounds[0]+A2_bounds[1])/2,
                    (mu_bounds[0]+mu_bounds[1])/2,
                    (d_bounds[0]+d_bounds[1])/2,
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
                    (R_1_bounds[0]+R_1_bounds[1])/2,
                    (R_2_bounds[0]+R_2_bounds[1])/2,
                    (E_bounds[0]+E_bounds[1])/2,
                    (A1_bounds[0]+A1_bounds[1])/2,
                    (A2_bounds[0]+A2_bounds[1])/2,
                    (mu_bounds[0]+mu_bounds[1])/2,
                    (k_bounds[0]+k_bounds[1])/2,
                ]
            )

    result = minimize(
        objective_function, initial_params, method="Nelder-Mead", bounds=bounds
    )

    return result.x, result.fun

def initialize_constants_optimization(R_1, R_2, E, A1, A2, mu):
    constants = {
        "num" : 15, # The number of joints
        "L" : 15e-3, # Total length of a joint
        "A_1" : A1, # Larger distance to the point of max length
        "A_2" : A2, # Smaller distance to the point of max length
        "R_1" : R_1, # Larger radius
        "R_2" : R_2, # Smaller radius
        "E" : E, # Modulus of elasticity
        "r" : 0.3e-3, # Radius of the backbone
        "mu" : mu, # Friction coefficient
    }

    constants["I"] = (np.pi * constants["r"]**4) / 4

    At = np.zeros(constants["num"])
    Ac = np.zeros(constants["num"])
    R_left = np.zeros(constants["num"])
    R_right = np.zeros(constants["num"])
    for i in range(constants["num"]):
        if i & 1 == 0:
            At[i] = constants["A_1"]
            Ac[i] = constants["A_2"]
            R_left[i] = constants["R_1"]
            R_right[i] = constants["R_2"]
        else:
            At[i] = constants["A_2"]
            Ac[i] = constants["A_1"]
            R_left[i] = constants["R_2"]
            R_right[i] = constants["R_1"]

    constants["At"] = At
    constants["Ac"] = Ac
    constants["R_left"] = R_left
    constants["R_right"] = R_right

    alpha = np.array([np.arcsin(At[i] / R_left[i]) for i in range(constants["num"])])
    betha = np.array([np.arcsin(Ac[i] / R_right[i]) for i in range(constants["num"])])

    constants["alpha"] = alpha
    constants["betha"] = betha

    Lt = np.array([R_left[i] * np.cos(a) + (constants["L"] - R_left[i]) for i, a in enumerate(alpha)])
    Lc = np.array([R_right[i] * np.cos(b) + (constants["L"] - R_right[i]) for i, b in enumerate(betha)])

    constants["Lt"] = Lt
    constants["Lc"] = Lc

    return constants


import numpy as np
from scipy.optimize import root

def elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, k, d, model= 'non-linear'):
    vertical_dist_left = (R_left[i] * (1 - np.cos(alpha[i] - th[i])))
    vertical_dist_right = (R_right[i] * (1 - np.cos(betha[i] + th[i])))

    # k = 10.5
    # d = 2.3e-3
    match model:
        case 'linear':
            return (
                (E * I / L) * k * th[i]
                )
        case 'non-linear':
            return (
                (E * I) * np.sin(th[i])
                * (1 / (vertical_dist_left) 
                   + 1 / (vertical_dist_right)
                )
            )
        case 'non-linear-L':
            if th[i]==0:
                return 0
            else:
                return (
                    (E * I) * np.sin(th[i]) * th[i]
                    * (
                        1 / (vertical_dist_left * th[i] + d * np.sin(th[i])) 
                        + 1 / (vertical_dist_right * th[i] + d * np.sin(th[i]))
                    )
                )

                
def equations(vars, constants, Ft, clearance, elastic_model, k, d):
    L = constants["L"]
    mu = constants["mu"]
    E = constants["E"]
    I = constants["I"]
    At = constants["At"]
    Ac = constants["Ac"]
    Lt = constants["Lt"]
    Lc = constants["Lc"]
    R_left = constants["R_left"]
    R_right = constants["R_right"]
    alpha = constants["alpha"]
    betha = constants["betha"]

    n = int(len(vars)/2)+1
    th = np.array(vars[0:n])
    Fr = np.array(vars[n:])
    Fr = np.insert(Fr, 0, 0)
    phi = np.array([calculate_phi(th[i], R_left[i], R_right[i], alpha[i], betha[i], Lt[i], Lc[i], At[i], Ac[i], clearance) for i in range(len(th))])

    sum_th = np.cumsum(phi)

    M = np.zeros(n)
    Fy = np.zeros(n)

    for i in range(n):
        # The last joint
        if i == n-1:
            M[i] = (
                np.exp(-mu * sum_th[i])
                * (
                    Ft
                    * (
                        (1+mu*th[i])*(At[i] - R_left[i]*np.sin(th[i]))
                        + th[i] * (Lt[i] / 2 + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i])))
                    )
                )
                - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, k, d, elastic_model)
            )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Ft * np.exp(-mu * sum_th[i]) * (1 + mu*th[i])
            )
        # Protagonist joints
        elif i & 1 == 0:
            M[i] = (
                np.exp(-mu * sum_th[i]) * th[i]
                * (
                    Ft
                    * (
                        Lt[i] / 2
                        + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i]))
                        + mu * (At[i] - R_left[i] * np.sin(th[i]))
                    )
                )
                + Fr[i+1] * ((At[i] - At[i+1]) + (R_left[i+1] * th[i+1] - R_left[i] * np.sin(th[i])))
                - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, k, d, elastic_model)
            )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Fr[i+1]
                - Ft * mu * np.exp(-mu * sum_th[i]) * th[i]
            )
        # Antagonist joints
        else:
            M_reverse =(
                np.exp(-mu * sum_th[i]) * th[i]
                * (
                    Ft
                    * (
                        Lt[i] / 2
                        + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i]))
                        + mu * (At[i] - R_left[i] * np.sin(th[i]))
                    )
                )
                - Fr[i+1] * ((Ac[i] - Ac[i+1]) - (R_left[i+1] * th[i+1] - R_left[i] * np.sin(th[i])))
            )
            # Reverse motion
            if M_reverse <0:
                th[i] = -th[i]
                M[i] = (
                    Fr[i+1] * (Ac[i] - (Ac[i+1] + R_left[i+1] * th[i+1] + R_right[i] * np.sin(th[i])))
                    - np.exp(-mu * sum_th[i]) * th[i] * Ft
                    * (
                        L - R_right[i] - Lt[i] / 2
                        + R_right[i] * np.cos(th[i])
                        + mu * (At[i] + R_right[i] * np.sin(th[i]))
                    )
                    - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, k, d, elastic_model)
                )
            #Normal motion
            elif M_reverse > 0:
                M[i] = (
                    np.exp(-mu * sum_th[i]) * th[i] * Ft
                    * (
                        Lt[i] / 2
                        + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i]))
                        + mu * (At[i] - R_left[i] * np.sin(th[i]))
                    )
                    - Fr[i+1] * ((Ac[i] - Ac[i+1]) - (R_left[i+1] * th[i+1] - R_left[i] * np.sin(th[i])))
                    - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, k, d, elastic_model)
                )

            else:
                pass

            # M[i] = (
            #     np.exp(-mu * sum_th[i]) * th[i]
            #     * (
            #         Ft
            #         * (
            #             Lt[i] / 2
            #             + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i]))
            #             + mu * (At[i] - R_left[i] * np.sin(th[i]))
            #         )
            #     )
            #     - Fr[i+1] * ((Ac[i] - Ac[i+1]) + (R_left[i+1] * th[i+1] - R_left[i] * np.sin(th[i])))
            #     - (
            #         (E * I) * np.sin(th[i])
            #         * (1 / (R_left[i] * (1 - np.cos(alpha[i] - th[i]))) 
            #         + 1 / (R_right[i] * (1 - np.cos(betha[i] + th[i]))))
            #     )
            # )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Fr[i+1]
                - Ft * mu * np.exp(-mu * sum_th[i]) * th[i]
            )

    return np.concatenate((M, Fy[1:]))

def initial_guess_gen(n, Ft, th_p_guess=0.4, th_a_guess=0.1):
    # theta_guess = [th_p_guess, th_a_guess] * int(n / 2) + [th_p_guess]
    # Ft_guess = [Ft * (0.5 + 0.5 * np.exp(-0.1 * i)) for i in range(n - 1)]
    theta_guess = [0]*n
    Ft_guess = [0]*(n-1)

    return theta_guess + Ft_guess


def solve_robot(
    config,
    Ft_values,
    equations,
    elastic_model,
    th_p_guess=0.5,
    th_a_guess=-0.1,
    method="hybr",
    clearance=0.4,
    k= 10.5,
    d= 2.3e-3
):
    theta_solutions = []
    F_solutions = []
    initial_guess = initial_guess_gen(
        config["num"], 0, th_p_guess=th_p_guess, th_a_guess=th_a_guess
    )

    for Ft in Ft_values:
        solution = root(
            lambda vars: equations(vars, config, Ft, clearance, elastic_model, k=k, d=d),
            initial_guess,
            method=method,
        )

        theta_solutions.append(solution.x[: config["num"]])
        F_solutions.append(np.append(solution.x[config["num"]:], Ft))
        initial_guess = list(solution.x)

        # if solution.success:
        #     theta_solutions.append(solution.x[:config["num"]])
        #     initial_guess = list(solution.x)
        # else:
        #     raise ValueError(f"Root-finding failed for Ft={Ft}: {solution.message}")

    return np.array(theta_solutions), np.array(F_solutions)

def theta_to_xy(theta, config, F_list):
    theta = np.ravel(theta)
    num = np.size(theta)
    R_left = config['R_left']
    R_right = config['R_right']
    Gama = (config['A_1'] - config['A_2'])/2 * 1000

    x_coords = [0]
    y_coords = [0]

    tf_total = np.eye(3)
    
    for i in range(num):
        delta_L = F_list[i]*config['L']/(20e6*np.pi*(3.1e-3)**2)
        L = (config['L']- delta_L)*1000
        if theta[i] >= 0:
            R = R_left[i] * 1000
        else:
            R = R_right[i] * 1000

        x = R*(1-np.cos(theta[i])) + L*np.cos(theta[i]) - Gama*np.sin(theta[i])
        y = R*(theta[i]-np.sin(theta[i])) - Gama + L*np.sin(theta[i]) + Gama*np.cos(theta[i])
        tf = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), x],
            [np.sin(theta[i]),  np.cos(theta[i]), y],
            [0,                0,               1]
        ])
        
        tf_total = tf_total @ tf
        
        new_x = tf_total[0, 2]
        new_y = tf_total[1, 2]
        
        x_coords.append(new_x)
        y_coords.append(new_y)

    return x_coords, y_coords

def calculate_phi(theta, R_left, R_right, alpha, betha, Lt, Lc, At, Ac, clearance):
    if theta > 0:
        R = R_left
        angle = alpha
        Lf = Lt
        A = At

        Ly = Lc
    elif theta < 0:
        theta = -theta
        R = R_right
        angle = betha
        Lf = Lc
        A = Ac

        Ly = Lt
    else:
        return 0
    
    x1 = R*theta + R*np.sin(angle-theta) + clearance*np.cos(theta)
    y1 = R*(1-np.cos(angle-theta)) - clearance*np.sin(theta)
    p1 = (x1, y1)

    x2 = x1 + Lf*np.sin(theta) - clearance*np.cos(theta)
    y2 = y1 + Lf*np.cos(theta) + clearance*np.sin(theta)
    p2 = (x2, y2)

    x3 = A + clearance
    y3 = 0
    p3 = (x3, y3)

    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return 0
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    
    phi = np.arccos(1-((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)/(2*radius**2))

    return phi

if __name__ == "__main__":
    main()