import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Ft = 10  # Newton
Fc = 0

A_left = 4.4e-3
A_right = 1e-3

# initial_guess = [0.5, 0.1, 0.5, 0.1, 0.5, Ft*0.95, Ft*0.8, Ft*0.75, Ft*0.7]
# initial_guess = [0.5, 0.1, 0.5, Ft*0.95, Ft*0.8]

num = 7
At = np.zeros(num)
Ac = np.zeros(num)
for i in range(num):
    if i & 1 == 0:
        At[i] = A_left
        Ac[i] = A_right
    else:
        Ac[i] = A_left
        At[i] = A_right

# At = np.array([4.4e-3, 1e-3, 4.4e-3])
# Ac = np.array([1e-3, 4.4e-3, 1e-3])

R = 5e-3
alpha = np.array([np.arcsin(A / R) for A in At])
betha = np.array([np.arcsin(A / R) for A in Ac])

print(np.degrees(alpha))
print(np.degrees(betha))

print(alpha)
print(betha)

Lr = 2e-3
Lt = np.array([Lr + R * np.cos(a) for a in alpha])
Lc = np.array([Lr + R * np.cos(b) for b in betha])

E = 7.5e9
r = 0.4e-3
I = (np.pi * r**4) / 4
mu = 0.1

def main():

    Ft_values = np.arange(1, 101, 1)
    theta_solutions = []

    for Ft in Ft_values:
        initial_guess = initial_guess_gen(Ft)

        solution = fsolve(lambda vars: equations(vars, Ft), initial_guess)

        theta_solutions.append(solution[:num]) 

    theta_solutions = np.array(np.degrees(theta_solutions))
    theta_solutions_sum = np.cumsum(theta_solutions, axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for i in range(num):
        axs[0].plot(Ft_values, theta_solutions[:, i], label=f"Theta {i+1}")
    

    axs[0].set_xlabel('Ft (N)')
    axs[0].set_ylabel('Theta (degrees)')
    axs[0].set_title('Variation of Theta with Ft')
    axs[0].legend()
    axs[0].grid(True)

    for i in range(num):
        axs[1].plot(Ft_values, theta_solutions_sum[:, i], label=f"Theta {i+1}")
    
    axs[1].set_xlabel('Ft (N)')
    axs[1].set_ylabel('Cumulative Theta (degrees)')
    axs[1].set_title('Cumulative Sum of Theta with Ft')
    axs[1].legend()
    axs[1].grid(True)

    for Ft in [20, 30, 40, 60, 80, 100]:
        initial_guess = initial_guess_gen(Ft)

        solution = fsolve(lambda vars: equations(vars, Ft), initial_guess)

        theta_plot = (solution[:num])
        plot_robot(theta_plot, axs[2], Ft)

    plt.tight_layout()
    plt.show()

    # solution = fsolve(equations, initial_guess)

    # for i, s in enumerate(solution):
    #     if i < num:
    #         print(f"Theta{i+1} = {s:.4f}")
    #     else:
    #         print(f"Fr{i+2-num} = {s:.4f}")

    # # print(f"Solution:\nth1 = {solution[0]:.4f}\nth2 = {solution[1]:.4f}\nth3 = {solution[2]:.4f}\nFr2 = {solution[3]:.4f}\nFr3 = {solution[4]:.4f}")

    # print(np.isclose(equations(solution), np.zeros(len(initial_guess))))
    # print(equations(solution))
    # # print(check_fx(solution))


def equations(vars, Ft):
    n = int(len(vars)/2)+1
    th = np.array(vars[0:n])
    Fr = np.array(vars[n:])
    Fr = np.insert(Fr, 0, 0)

    sum_th = np.cumsum(th)

    M = np.zeros(n)
    Fy = np.zeros(n)

    for i in range(n):
        if i == n-1:
            M[i] = (
                np.exp(-mu * sum_th[i])
                * (
                    Ft
                    * (
                        (1+mu*th[i])*(At[i] - R*np.sin(th[i]))
                        + th[i] * (Lt[i] / 2 + R * (np.cos(th[i]) - np.cos(alpha[i])))
                    )
                    - Fc
                    * (
                        (1+mu*th[i])*(Ac[i] + R*np.sin(th[i]))
                        - th[i] * (Lc[i] / 2 + R * (np.cos(th[i]) - np.cos(betha[i])))
                    )
                )
                - (
                    (E * I / R)
                    * np.sin(th[i])
                    * (1 / (1 - np.cos(alpha[i] - th[i])) + 1 / (1 - np.cos(betha[i] + th[i])))
                )
            )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                + np.exp(-mu * sum_th[i]) * (Fc - Ft) * (1 + mu*th[i])
            )
        elif i & 1 == 0:
            M[i] = (
                np.exp(-mu * sum_th[i]) * th[i]
                * (
                    Ft
                    * (
                        Lt[i] / 2
                        + R * (np.cos(th[i]) - np.cos(alpha[i]))
                        + mu * (At[i] - R * np.sin(th[i]))
                    )
                    - Fc
                    * (
                        Lc[i] / 2
                        + R * (np.cos(th[i]) - np.cos(betha[i]))
                        - mu * (Ac[i] + R * np.sin(th[i]))
                    )
                )
                + Fr[i+1] * ((At[i] - At[i+1]) + (R * th[i+1] - R * np.sin(th[i])))
                - (
                    (E * I / R)
                    * np.sin(th[i])
                    * (1 / (1 - np.cos(alpha[i] - th[i])) + 1 / (1 - np.cos(betha[i] + th[i])))
                )
            )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Fr[i+1]
                + mu * np.exp(-mu * sum_th[i]) * th[i] * (Fc - Ft)
            )
        else:
            M[i] = (
                np.exp(-mu * sum_th[i]) * th[i]
                * (
                    Ft
                    * (
                        Lt[i] / 2
                        + R * (np.cos(th[i]) - np.cos(alpha[i]))
                        + mu * (At[i] - R * np.sin(th[i]))
                    )
                    - Fc
                    * (
                        Lc[i] / 2
                        + R * (np.cos(th[i]) - np.cos(betha[i]))
                        - mu * (Ac[i] + R * np.sin(th[i]))
                    )
                )
                + Fr[i+1] * ((Ac[i] - Ac[i+1]) + (R * th[i+1] - R * np.sin(th[i])))
                - (
                    (E * I / R)
                    * np.sin(th[i])
                    * (1 / (1 - np.cos(alpha[i] - th[i])) + 1 / (1 - np.cos(betha[i] + th[i])))
                )
            )

            Fy[i] = (
                Fr[i] * np.cos(th[i])
                - Fr[i+1]
                + mu * np.exp(-mu * sum_th[i]) * th[i] * (Fc - Ft)
            )

    # X-Force Equilibrium
    # Fx1 = (
    #     np.exp(-mu * th1) * th1
    #     * (np.cos(th1) - mu*np.sin(th1)) * (Ft - Fc)
    #     - Fr2 * np.sin(th1)
    # )

    # Fx2 = (
    #     np.exp(-mu * (th1 + th2)) * th2
    #     * (np.cos(th2) - mu*np.sin(th2)) * (Ft - Fc)
    #     - Fr3 * np.sin(th2)
    # )

    return np.concatenate((M, Fy[1:]))

def check_fx(vars):
    th1, th2, th3, Fr2, Fr3 = vars
    # X-Force Equilibrium
    Fx1 = (
        np.exp(-mu * th1) * th1
        * (np.cos(th1) - mu*np.sin(th1)) * (Ft - Fc)
        - Fr2 * np.sin(th1)
    )

    Fx2 = (
        np.exp(-mu * (th1 + th2)) * th2
        * (np.cos(th2) - mu*np.sin(th2)) * (Ft - Fc)
        - Fr3 * np.sin(th2)
    )
    return [Fx1, Fx2]

def plot_robot(theta_radians, ax, Ft):

    link_lengths = [0.1 for _ in range(num)]

    # theta_radians = np.radians(theta_degrees)  # Convert angles to radians
    x_coords = [0]  # x-coordinate of the base (origin)
    y_coords = [0]  # y-coordinate of the base (origin)

    # Calculate the position of each joint
    for i, length in enumerate(link_lengths):
        x_coords.append(x_coords[-1] + length * np.cos(np.sum(theta_radians[:i+1])))
        y_coords.append(y_coords[-1] + length * np.sin(np.sum(theta_radians[:i+1])))

    # Plot the robot
    ax.plot(x_coords, y_coords, '-o', label=f"{Ft}N")
    ax.text(x_coords[-1], y_coords[-1], f'{Ft}N', fontsize=12, ha='left', va='bottom')
    ax.axis('equal')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.set_title(f'Robot Configurations')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_ylim(0, 0.8)
    ax.set_xticks(np.arange(-0.1, 0.8, 0.1))
    ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    ax.grid(True)
    ax.legend()

def initial_guess_gen(Ft):
    return [0.8, 0.12, 0.8, 0.12, 0.8, 0.12, 0.8, Ft * 0.95, Ft * 0.8, Ft * 0.75, Ft * 0.7, Ft * 0.65, Ft * 0.6]

if __name__ == "__main__":
    main()
