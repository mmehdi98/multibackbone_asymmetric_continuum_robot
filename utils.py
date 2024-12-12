import numpy as np
import json

def euclidean_dist(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def theta_to_xy(theta, config, F_list):
    theta = np.ravel(theta)
    num = np.size(theta)
    R_left = config['R_left']
    R_right = config['R_right']
    Gama = (config['A_1'] - config['A_2'])/2 * 1000
    E_plastic = config["E_plastic"]

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

def read_measurements(directory, mode="protagonist", test_num=None):
    with open(directory, 'r') as file:
        data = json.load(file)

    test_keys = [key for key in data.keys() if key.startswith('test_') and not key.endswith('_ply_filename')]
    test_numbers = [int(key.split('_')[1]) for key in test_keys]

    match mode:
        case "protagonist":
            ref_test = min(test_numbers)
            max_disp_test = max(test_numbers)
        case "antagonist":
            ref_test = max(test_numbers)
            max_disp_test = min(test_numbers)
        case _:
            raise ValueError("mode is not valid")

    for test_name, points in data.items():
        if test_name.endswith(f"{ref_test}"):
            x_ref = points[0][0]
            y_ref = points[0][1]

    x = []
    y = []
    if test_num == None:
        for test_name, points in data.items():
            if test_name.endswith('_ply_filename'):
                continue 
            
            x.append([(point[0] - x_ref)*1000 for point in points])
            y.append([-(point[1] - y_ref)*1000 for point in points])
    elif test_num == "max_disp":
        for test_name, points in data.items():
            if test_name.endswith(f"{max_disp_test}"):
                x.append([(point[0] - x_ref)*1000 for point in points])
                y.append([-(point[1] - y_ref)*1000 for point in points])
    else:
        for test_name, points in data.items():
            if test_name.endswith(f"{test_num}"):
                x.append([(point[0] - x_ref)*1000 for point in points])
                y.append([-(point[1] - y_ref)*1000 for point in points])

    return x, y

def calculate_phi(theta, config):
    clearance = config["clearance"]
    phi = np.full_like(theta, 0)
    for i, th in enumerate(theta):
        if th > 0:
            R = config["R_left"][i]
            angle = config["alpha"][i]
            Lf = config["Lt"][i]
            A = config["At"][i]

            Ly = config["Lc"][i]
        elif th < 0:
            th = -th
            R = config["R_right"][i]
            angle = config["betha"][i]
            Lf = config["Lc"][i]
            A = config["Ac"][i]

            Ly = config["Lt"][i]
        else:
            phi[i] = 0
            continue
        
        x1 = R*th + R*np.sin(angle-th) + clearance*np.cos(th)
        y1 = R*(1-np.cos(angle-th)) - clearance*np.sin(th)
        p1 = (x1, y1)

        x2 = x1 + Lf*np.sin(th) - clearance*np.cos(th)
        y2 = y1 + Lf*np.cos(th) + clearance*np.sin(th)
        p2 = (x2, y2)

        x3 = A + clearance
        y3 = 0
        p3 = (x3, y3)

        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if abs(det) < 1.0e-6:
            phi[i] = 0
            continue
        
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        
        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        
        phi[i] = np.arccos(1-((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)/(2*radius**2))

    return phi

def tendon_disp(thetas, config, F_list):
    R_left = config["R_left"]
    R_right = config["R_right"]
    alpha = config["alpha"]
    At = config["At"]
    E_plastic = config["E_plastic"]
    tendon_disp = 0

    for i, theta in enumerate(thetas):
        delta_L = F_list[i]*config['L']/(E_plastic*np.pi*(3.1e-3)**2)
        tendon_disp +=delta_L
        x0 = At[i]
        y0 = 0
        x1 = At[i]
        y1 = R_left[i]*(1-np.cos(alpha[i]))
        d1 = euclidean_dist((x0, y0), (x1, y1))
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

    return tendon_disp