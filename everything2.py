import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import plotter
from utils import tendon_disp, calculate_phi, read_measurements

def main():
    config = initialize_constants()

    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    measured_Ft = [0.0, 1.876832845, 2.815249267, 3.753665689, 4.692082111, 5.630498534, 7.507331378, 10.32258065, 13.13782991, 16.8914956, 22.52199413,
                    29.09090909, 38.47507331, 47.85923754]
    Ft_values = np.append(np.linspace(0, 60, 100), measured_Ft)
    Ft_values = np.sort(Ft_values)

    # Finding theta angles for the Ft range
    theta_solutions, F_solutions = solve_robot(config, Ft_values, equations, elastic_model="non-linear")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot theta vs Ft
    plotter.plot_theta(Ft_values, theta_solutions, axs[0])

    # Plot cumulative sum of theta vs Ft
    plotter.plot_theta_sum(Ft_values, theta_solutions, axs[1])

    # Plot the robot for given forces
    E_plastic = 20e6
    modeled_x, modeled_y = [], []
    for Ft in measured_Ft:
        index = np.where(Ft_values == Ft)[0][0]
        theta_plot = theta_solutions[index]
        F_plot = F_solutions[index]
        plot_robot(theta_plot, axs[2], Ft, config, F_plot, E_plastic)

        x_m, y_m = theta_to_xy(theta_plot, config, F_plot, E_plastic)
        modeled_x.append(x_m)
        modeled_y.append(y_m)
    x_measured, y_measured = read_measurements(measured_directory)

    errors = []
    for i in range(10, len(x_measured)):
        for k, (mx, my, x, y) in enumerate(zip(modeled_x[i], modeled_y[i], x_measured[i], y_measured[i])):
            error = []
            error.append(np.sqrt((mx - x) ** 2 + (my - y) ** 2)/(config['L']*1000*(k+1)))
        errors.append(error)
    mean_error = np.mean(np.array(errors).ravel())
    print(f"last error: {errors[-1][-1]}")

    print(f"Mean Error: {mean_error}")

    plot_measured_robot(config, measured_directory, axs[2])

    plt.tight_layout()
    plt.show()

    # Plot Ft vs tendon disp
    tendon_displacements = [tendon_disp(theta_solutions[i], config, F_solutions[i], E_plastic)*1000 for i in range(len(Ft_values))]
    plt.figure(figsize=(8, 6))
    plt.plot(tendon_displacements, Ft_values, label="Tendon Displacement", color="b")
    plt.ylabel("$F_t$")
    plt.xlabel("Tendon Displacement")
    plt.title("Tendon Displacement vs $F_t$")
    plt.grid(True)
    plt.legend()
    plt.show()

# The phi calculation from y=0, clearance = 0.49497274317149775mm and length reduction in kinematics
def initialize_constants():
    constants = {
        "num" : 15, # The number of joints
        "L" : 15e-3, # Total length of a joint
        "A_1" : 3.6321744229799937e-3, # Larger distance to the point of max length
        "A_2" : 1.4128105036353378e-3, # Smaller distance to the point of max length
        "R_1" : 5.377409026631593e-3, # Larger radius
        "R_2" : 2.956017023426858e-3, # Smaller radius
        "E" : 9.957668366e9, # Modulus of elasticity
        "r" : 0.3e-3, # Radius of the backbone
        "mu" : 0.14890120697639153, # Friction coefficient
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
    F_solutions = []
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
        F_solutions.append(np.append(solution.x[config["num"]:], Ft))
        initial_guess = list(solution.x)

        # if solution.success:
        #     theta_solutions.append(solution.x[:config["num"]])
        #     initial_guess = list(solution.x)
        # else:
        #     raise ValueError(f"Root-finding failed for Ft={Ft}: {solution.message}")

    return np.array(theta_solutions), np.array(F_solutions)

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
                        (At[i] - R_left[i]*np.sin(th[i]))
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

def theta_to_xy(theta, config, F_list, E_plastic):
    theta = np.ravel(theta)
    num = np.size(theta)
    R_left = config['R_left']
    R_right = config['R_right']
    Gama = (config['A_1'] - config['A_2'])/2 * 1000

    x_coords = [0]
    y_coords = [0]

    tf_total = np.eye(3)
    
    for i in range(num):
        delta_L = F_list[i]*config['L']/(E_plastic*np.pi*(3.1e-3)**2)
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

def plot_measured_robot(config, directory, ax):

    x, y = read_measurements(directory)
    # theta = utils.xy_to_theta(x, y)
    # print(theta)
    # for i in range(len(x)):
    #     x[i], y[i] = utils.th2xy_measurements(theta[i], config["L"])

    for i in range(len(x)):
        ax.plot(x[i], y[i], marker='o', color='orange')

def plot_robot(theta, ax, Ft, config, F_list, E_plastic):
    
    x_coords, y_coords = theta_to_xy(theta, config, F_list, E_plastic)

    # Plot the robot
    ax.plot(x_coords, y_coords, '-o', label=f"{Ft}N", color='blue')
    ax.text(x_coords[-1], y_coords[-1], f'{Ft:.2f}N', fontsize=12, ha='left', va='bottom')
    ax.axis('equal')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.set_title(f'Robot Configurations')
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_ylim(0, 0.8)
    # ax.set_xticks(np.arange(-0.1, 0.8, 0.1))
    # ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    ax.grid(True)
    # ax.legend()

def tendon_disp(thetas, config, F_list, E_plastic):
    R_left = config["R_left"]
    R_right = config["R_right"]
    alpha = config["alpha"]
    At = config["At"]
    tendon_disp = 0

    sum_d1 = 0
    for i, theta in enumerate(thetas):
        delta_L = F_list[i]*config['L']/(E_plastic*np.pi*(3.1e-3)**2)
        tendon_disp +=delta_L
        x0 = At[i]
        y0 = 0
        x1 = At[i]
        y1 = R_left[i]*(1-np.cos(alpha[i]))
        d1 = euclidean_dist((x0, y0), (x1, y1))
        sum_d1 += d1 * 1000
        if theta >= 0:
            x2 = R_left[i] * (theta + np.sin(alpha[i]-theta))
            y2 = R_left[i] * (1-np.cos(alpha[i]-theta))
            d2 = euclidean_dist((x0, y0), (x2, y2))
            tendon_disp += np.absolute(d2 - d1)
        else:
            x2 = At[i]*np.cos(alpha[i]/2 + theta) + R_right[i] * (np.sin(theta) - theta)
            y2 = At[i]*np.sin(alpha[i]/2 + theta) + R_right[i] * (1-np.cos(theta))
            d2 = euclidean_dist((x0, y0), (x2, y2))
            tendon_disp -= np.absolute(d2 - d1)
        # print(f"{i}: {d1 - d2}")
    # print(f"sum_d1: {sum_d1}")

    return tendon_disp

def euclidean_dist(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

if __name__ == "__main__":
    main()