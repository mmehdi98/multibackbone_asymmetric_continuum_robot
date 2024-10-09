import numpy as np
from scipy.optimize import fsolve

Ft = 100  # Newton
Fc = 5

At = np.array([0, 4.4e-3, 1e-3, 4.4e-3])
Ac = np.array([0, 1e-3, 4.4e-3, 1e-3])

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
    initial_guess = [0.5, 0.1, 0.5, Ft*0.95, Ft*0.8]

    solution = fsolve(equations, initial_guess)

    print(f"Solution:\nth1 = {solution[0]:.4f}\nth2 = {solution[1]:.4f}\nth3 = {solution[2]:.4f}\nFr2 = {solution[3]:.4f}\nFr3 = {solution[4]:.4f}")

    print(np.isclose(equations(solution), [0.0, 0.0, 0.0, 0.0, 0.0]))
    print(equations(solution))


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
        np.exp(-mu * (th1 + th2)) * th2
        * (np.cos(th2) + mu*np.sin(th2)) * (Fc - Ft)
        + Fr2
        - Fr3 * np.cos(th2)
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
        np.exp(-mu * (th1 + th2 + th3))
        * (
            (1+mu*th3) * np.cos(th3)
            + th3 * np.sin(th3)
        )
        * (Fc - Ft)
        + Fr3
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
