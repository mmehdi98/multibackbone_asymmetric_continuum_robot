import numpy as np
from scipy.optimize import fsolve

Ft = 1000  # Newton
Fc = 0

At = np.array([0, 4.4e-3, 1e-3, 4.4e-3])
Ac = np.array([0, 1e-3, 4.4e-3, 1e-3])

R = 7e-3
alpha = np.array([np.arcsin(A / R) for A in At])
betha = np.array([np.arccos(A / R) for A in Ac])

Lr = 0
Lt = np.array([Lr + R * np.cos(a) for a in alpha])
Lc = np.array([Lr + R * np.cos(b) for b in betha])

E = 70e9
r = 0.4e-3
I = (np.pi * r**4) / 4
mu = 0.7

def main():
    initial_guess = [0, 0, 0, 0, 0]

    solution = fsolve(equations, initial_guess)

    print(f"Solution:\nth1 = {np.degrees(solution[0])}\nth2 = {np.degrees(solution[1])}\nth3 = {np.degrees(solution[2])}\nFr2 = {solution[3]}\nFr3 = {solution[4]}")


def equations(vars):
    th1, th2, th3, Fr2, Fr3 = vars
    eq1 = (
        np.exp(-mu * th1) * th1
        * (
            Ft
            * (
                Lt[1] / 2
                + R * (np.cos(th1) - np.cos(alpha[1]))
                + mu * (At[1] - R * np.sin(th1))
            )
            - Fc
            * (
                Lc[1] / 2
                + R * (np.cos(th1) - np.cos(betha[1]))
                - mu * (Ac[1] + R * np.sin(th1))
            )
        )
        + Fr2 * ((At[1] - At[2]) + (R * th2 - R * np.sin(th1)))
        - (
            (E * I / R)
            * np.sin(th1)
            * (1 / (1 - np.cos(alpha[1] - th1)) + 1 / (1 - np.cos(betha[1] + th1)))
        )
    )
    eq2 = (
        np.exp(-mu * th1) * th1
        * (np.cos(th1) - mu*np.sin(th1)) * (Ft - Fc)
        - Fr2 * np.sin(th1)
    )
    eq3 = (
        np.exp(-mu * (th1 + th2)) * th2
        * (
            Ft
            * (
                Lt[2] / 2
                + R * (np.cos(th2) - np.cos(alpha[2]))
                + mu * (At[2] - R * np.sin(th2))
            )
            - Fc
            * (
                Lc[2] / 2
                + R * (np.cos(th2) - np.cos(betha[2]))
                - mu * (Ac[2] + R * np.sin(th2))
            )
        )
        + Fr3 * ((Ac[2] - Ac[3]) + (R * th3 - R * np.sin(th2)))
        - (
            (E * I / R)
            * np.sin(th2)
            * (1 / (1 - np.cos(alpha[2] - th2)) + 1 / (1 - np.cos(betha[2] + th2)))
        )
    )
    eq4 = (
        np.exp(-mu * (th1 + th2)) * th2
        * (np.cos(th2) - mu*np.sin(th2)) * (Ft - Fc)
        - Fr3 * np.sin(th2)
    )
    eq5 = (
        np.exp(-mu * (th1 + th2 + th3))
        * (
            Ft
            * (
                (1+mu*th3)*(At[3] - R*np.sin(th3))
                + th3 * (Lt[3] / 2 + R * (np.cos(th3) - np.cos(alpha[3])))
            )
            - Fc
            * (
                (1+mu*th3)*(Ac[3] + R*np.sin(th3))
                - th3 * (Lc[3] / 2 + R * (np.cos(th3) - np.cos(betha[3])))
            )
        )
        - (
            (E * I / R)
            * np.sin(th3)
            * (1 / (1 - np.cos(alpha[3] - th3)) + 1 / (1 - np.cos(betha[3] + th3)))
        )
    )
    return [eq1, eq2, eq3, eq4, eq5]


if __name__ == "__main__":
    main()
