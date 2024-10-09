import numpy as np
from scipy.optimize import fsolve

Ft = 10  # Newton
Fc = 5

A_left = 4.4e-3
A_right = 1e-3

initial_guess = [0.5, 0.1, 0.5, 0.1, 0.5, Ft*0.95, Ft*0.8, Ft*0.75, Ft*0.7]
# initial_guess = [0.5, 0.1, 0.5, Ft*0.95, Ft*0.8]

num = int(len(initial_guess)/2)+1
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
betha = np.array([np.arccos(A / R) for A in Ac])

print(alpha)
print(betha)

Lr = 2e-3
Lt = np.array([Lr + R * np.cos(a) for a in alpha])
Lc = np.array([Lr + R * np.cos(b) for b in betha])

E = 7.5e9
r = 0.4e-3
I = (np.pi * r**4) / 4
mu = 0.4

def main():

    solution = fsolve(equations, initial_guess)

    for i, s in enumerate(solution):
        if i < num:
            print(f"Theta{i+1} = {s:.4f}")
        else:
            print(f"Fr{i+2-num} = {s:.4f}")

    # print(f"Solution:\nth1 = {solution[0]:.4f}\nth2 = {solution[1]:.4f}\nth3 = {solution[2]:.4f}\nFr2 = {solution[3]:.4f}\nFr3 = {solution[4]:.4f}")

    print(np.isclose(equations(solution), np.zeros(len(initial_guess))))
    print(equations(solution))
    # print(check_fx(solution))


def equations(vars):
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

if __name__ == "__main__":
    main()
