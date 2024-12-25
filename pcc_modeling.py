import numpy as np
import matplotlib.pyplot as plt

from configs import initialize_constants
import utils

def main():
    measured_directory = 'F:\\Measurements\\Test_3\\protagonist_motion\\loading\\1\\Coordinates_1.json'
    x_measured, y_measured = utils.read_measurements(measured_directory)

    config = initialize_constants()
    # r0 = (config["A_1"] + config["A_2"])/2
    r0 = 6e-3
    tendon_disp = [0.0, 1.994662002, 3.529017389, 4.909937236, 6.367574853, 7.82521247, 9.206132318, 10.5103344, 11.73781871, 13.04202078, 
                    14.11606955, 14.88324725, 15.95729602, 16.72447371]
    
    plt.figure()
    for m, delta_L in enumerate(tendon_disp):
        kappa = delta_L*0.001/(r0*config["L"]*15)
        x_coords = []
        y_coords = []
        for i in range(config["num"]):
            s = config["L"]*(i+1)
            x, y = calculate_position(kappa, s)
            x_coords.append(x*1000)
            y_coords.append(y*1000)
        plt.plot(x_coords, y_coords, '-o', color='blue')
        plt.plot(x_measured[m], y_measured[m], marker='o', color='orange')
        plt.axis('equal')
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.title(f'Robot Configurations')
        plt.xlabel('X position (mm)')
        plt.ylabel('Y position (mm)')
        plt.grid(True)
    plt.show()

def calculate_position(kappa, s):
    if kappa == 0:
        x = s
        y = 0
    else:
        x = (1/kappa)*np.sin(kappa*s)
        y = (1/kappa)*(1-np.cos(kappa*s))

    return x, y

if __name__ == "__main__":
    main()