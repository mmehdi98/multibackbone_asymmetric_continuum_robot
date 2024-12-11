import numpy as np
import json

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
    
            # return (
            #     (E * I) * np.sin(th[i])
            #     * (1 / (R_left[i] * (1 - np.cos(alpha[i] - th[i]))) 
            #     + 1 / (R_right[i] * (1 - np.cos(betha[i] + th[i]))))
            # )

def euclidean_dist(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def tendon_disp(thetas, config):
    R_left = config["R_left"]
    R_right = config["R_right"]
    alpha = config["alpha"]
    At = config["At"]
    tendon_disp = 0

    sum_d1 = 0
    for i, theta in enumerate(thetas):
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

def theta_to_xy(theta, config):
    theta = np.ravel(theta)
    num = np.size(theta)
    L = config['L'] * 1000
    R_left = config['R_left']
    R_right = config['R_right']
    Gama = (config['A_1'] - config['A_2'])/2 * 1000

    x_coords = [0]
    y_coords = [0]

    tf_total = np.eye(3)
    
    for i in range(num):
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

def th2xy_measurements(theta, length):
    theta = np.ravel(theta)
    num = np.size(theta)
    length = length * 1000

    x_coords = [0]
    y_coords = [0]

    tf_total = np.eye(3)
    
    for i in range(num):
        tf = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), length * np.cos(theta[i])],
            [np.sin(theta[i]),  np.cos(theta[i]), length * np.sin(theta[i])],
            [0,                0,               1]
        ])
        
        tf_total = tf_total @ tf
        
        new_x = tf_total[0, 2]
        new_y = tf_total[1, 2]
        
        x_coords.append(new_x)
        y_coords.append(new_y)

    return x_coords, y_coords

def xy_to_theta(full_x, full_y):
    theta = []
    for x_coords, y_coords in zip(full_x, full_y):
        th = []
        for i in range(1, len(x_coords)):
            if i == 1:
                th.append(np.arctan(y_coords[i]/x_coords[i]))
            else:
                # v1 = np.sqrt((x_coords[i]-x_coords[i-1])**2 + (y_coords[i]-y_coords[i-1])**2)
                # v2 = np.sqrt((x_coords[i-1]-x_coords[i-2])**2 + (y_coords[i-1]-y_coords[i-2])**2)
                det = - (x_coords[i]-x_coords[i-1])*(y_coords[i-1]-y_coords[i-2]) + (x_coords[i-1]-x_coords[i-2])*(y_coords[i]-y_coords[i-1])
                dot = (x_coords[i]-x_coords[i-1])*(x_coords[i-1]-x_coords[i-2]) + (y_coords[i]-y_coords[i-1])*(y_coords[i-1]-y_coords[i-2])
                th.append(np.arctan2(det,dot))
        theta.append(th)
    return theta

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

def calculate_phi(theta, R_left, R_right, alpha, betha, Lt, Lc, At, Ac):
    if theta > 0:
        R = R_left
        angle = alpha
        Lf = Lt
        A = At

        Ly = Lc
    elif theta < 0:
        theta = -theta
        R = R_right
        angle = betha
        Lf = Lc
        A = Ac

        Ly = Lt
    else:
        return 0
    
    x1 = R*theta + R*np.sin(angle-theta)
    y1 = R*(1-np.cos(angle-theta))
    p1 = (x1, y1)

    x2 = x1 + Lf*np.sin(theta)
    y2 = y1 + Lf*np.cos(theta)
    p2 = (x2, y2)

    x3 = A
    y3 = -Ly
    p3 = (x3, y3)

    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return 0
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    
    phi = np.arccos(1-((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)/(2*radius**2))

    return phi