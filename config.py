import numpy as np

def initialize_constants():
    constants = {
        "num" : 15, # The number of joints
        "L" : 15e-3, # Total length of a joint
        "A_1" : 3e-3, # Larger distance to the point of max length
        "A_2" : 1.2e-3, # Smaller distance to the point of max length
        "R_1" : 6.41e-3, # Larger radius
        "R_2" : 2.43e-3, # Smaller radius
        "E" : 70e9, # Modulus of elasticity
        "r" : 0.3e-3, # Radius of the backbone
        "mu" : 0.1, # Friction coefficient
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