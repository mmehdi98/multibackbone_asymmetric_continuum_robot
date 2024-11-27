import numpy as np
import json

def elastic_moment(i, E, I, L, th, R_left, R_right, alpha, betha, model= 'non-linear'):
    vertical_dist_left = (R_left[i] * (1 - np.cos(alpha[i] - th[i])))
    vertical_dist_right = (R_right[i] * (1 - np.cos(betha[i] + th[i])))

    k = 10.5
    d = 2.3e-3
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

def theta_to_xy(theta, length):
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