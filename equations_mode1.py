import numpy as np
from utils import elastic_moment

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

    sum_th = np.cumsum(th)

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