import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import plotter
from utils import tendon_disp, calculate_phi

def main():
    config = initialize_constants()

    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    measured_Ft = [0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                    29.09090909, 38.47507331, 47.85923754]
    Ft_values = np.append(np.linspace(0, 60, 100), measured_Ft)
    Ft_values = np.sort(Ft_values)

    # Finding theta angles for the Ft range
    theta_solutions = solve_robot(config, Ft_values, equations, elastic_model="non-linear")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    for Ft in measured_Ft:
        index = np.where(Ft_values == Ft)[0][0]
        theta_plot = theta_solutions[index]
        plotter.plot_robot(theta_plot, axs[2], Ft, config)

    plotter.plot_measured_robot(config, measured_directory, axs[2])

    plt.tight_layout()
    plt.show()

    # Plot Ft vs tendon disp
    tendon_displacements = [tendon_disp(theta_solutions[i], config) for i in range(len(Ft_values))]
    plt.figure(figsize=(8, 6))
    plt.plot(tendon_displacements, Ft_values, label="Tendon Displacement", color="b")
    plt.ylabel("$F_t$")
    plt.xlabel("Tendon Displacement")
    plt.title("Tendon Displacement vs $F_t$")
    plt.grid(True)
    plt.legend()
    plt.show()

def initialize_constants():
    constants = {
        "num" : 15, # The number of joints
        "L" : 15e-3, # Total length of a joint
        "A_1" : 3.5907e-3, # Larger distance to the point of max length
        "A_2" : 1.2e-3, # Smaller distance to the point of max length
        "R_1" : 6.2682e-3, # Larger radius
        "R_2" : 2.8472e-3, # Smaller radius
        "E" : 7.04869e9, # Modulus of elasticity
        "r" : 0.3e-3, # Radius of the backbone
        "mu" : 0.22889, # Friction coefficient
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

def elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, model= 'non-linear'):
    vertical_dist_left = (R_left[i] * (1 - np.cos(alpha[i] - th[i])))
    vertical_dist_right = (R_right[i] * (1 - np.cos(betha[i] + th[i])))

    k = 10
    d = -2e-3
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

def equations(vars, constants, Ft, elastic_model):
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
    phi = np.array([calculate_phi(th[i], R_left[i], R_right[i], alpha[i], betha[i], Lt[i], Lc[i], At[i], Ac[i]) for i in range(len(th))])

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
                        (0.9*At[i] - R_left[i]*np.sin(th[i]))
                        +(mu*th[i])*(At[i] - R_left[i]*np.sin(th[i]))
                        + th[i] * (Lt[i] / 2 + R_left[i] * (np.cos(th[i]) - np.cos(alpha[i])))
                    )
                )
                - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, elastic_model)
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
                - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, elastic_model)
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
                    - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, elastic_model)
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
                    - elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, elastic_model)
                )

            else:
                pass

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Fr[i+1]
                - Ft * mu * np.exp(-mu * sum_th[i]) * th[i]
            )

    return np.concatenate((M, Fy[1:]))

if __name__ == "__main__":
    main()