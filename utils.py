import numpy as np

def elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, model= 'non-linear'):
    vertical_dist_left = (R_left[i] * (1 - np.cos(alpha[i] - th[i])))
    vertical_dist_right = (R_right[i] * (1 - np.cos(betha[i] + th[i])))

    k = 10
    d = 0.9e-3
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
            return (
                (E * I) * np.sin(th[i]) * th[i]
                * (1 / (vertical_dist_left * th[i] + d * np.sin(th[i])) 
                   + 1 / (vertical_dist_right * th[i] + d * np.sin(th[i]))
                )
            )
    
            # return (
            #     (E * I) * np.sin(th[i])
            #     * (1 / (R_left[i] * (1 - np.cos(alpha[i] - th[i]))) 
            #     + 1 / (R_right[i] * (1 - np.cos(betha[i] + th[i]))))
            # )