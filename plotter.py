import numpy as np
import matplotlib.pyplot as plt
import json
import utils

def plot_robot(theta, ax, Ft, length):
    
    x_coords, y_coords = utils.theta_to_xy(theta, length)

    # Plot the robot
    ax.plot(x_coords, y_coords, '-o', label=f"{Ft}N", color='blue')
    ax.text(x_coords[-1], y_coords[-1], f'{Ft:.2f}N', fontsize=12, ha='left', va='bottom')
    ax.axis('equal')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.set_title(f'Robot Configurations')
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_ylim(0, 0.8)
    # ax.set_xticks(np.arange(-0.1, 0.8, 0.1))
    # ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    ax.grid(True)
    # ax.legend()

def plot_theta(Ft_values, theta_solutions, ax):
    theta_solutions = np.degrees(theta_solutions)
    num = np.size(theta_solutions[0])
    for i in range(num):
        ax.plot(Ft_values, theta_solutions[:, i], label=f"Theta {i+1}")
    
    ax.set_xlabel('Ft (N)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('Variation of Theta with Ft')
    ax.legend()
    ax.grid(True)

def plot_theta_sum(Ft_values, theta_solutions, ax):
    theta_solutions = np.degrees(theta_solutions)
    num = np.size(theta_solutions[0])
    theta_solutions_sum = np.cumsum(theta_solutions, axis= 1)
    for i in range(num):
        ax.plot(Ft_values, theta_solutions_sum[:, i], label=f"Theta {i+1}")
    
    ax.set_xlabel('Ft (N)')
    ax.set_ylabel('Cumulative Theta (degrees)')
    ax.set_title('Cumulative Sum of Theta with Ft')
    ax.legend()
    ax.grid(True)

def plot_measured_robot(directory, ax):

    x, y = utils.read_measurements(directory)

    for i in range(len(x)):
        ax.plot(x[i], y[i], marker='o', color='orange')

    # with open(directory, 'r') as file:
    #     data = json.load(file)

    # for test_name, points in data.items():
    #     if test_name.endswith('2051'):
    #         x_ref = points[0][0]
    #         y_ref = points[0][1]

    # for test_name, points in data.items():
    #     if test_name.endswith('_ply_filename'):
    #         continue 
        
    #     x = [(point[0] - x_ref)*1000 for point in points]
    #     y = [-(point[1] - y_ref)*1000 for point in points]
        
        # ax.plot(x, y, marker='o', label=test_name, color='orange')
    # ax.legend()


def measure_error(theta, Ft, length, measured_directory):

    modeled_x, modeled_y = utils.theta_to_xy(theta, length)

    x_measured, y_measured = utils.read_measurements(measured_directory, test_num="max_disp")
    x_measured = x_measured[0]
    y_measured = y_measured[0]
    
    # # Get the measured robot's coordinates
    # with open(measured_directory, 'r') as file:
    #     data = json.load(file)

    # x_measured, y_measured = None, None
    # for test_name, points in data.items():
    #     if test_name.endswith('2051'):  # Assuming this is the reference test
    #         x_ref, y_ref = points[0][0], points[0][1]
    #         break

    # for test_name, points in data.items():
    #     if test_name.endswith('2320'):
    #         x_measured = [(point[0] - x_ref) * 1000 for point in points]
    #         y_measured = [-(point[1] - y_ref) * 1000 for point in points]
    #         break
    
    if x_measured is None or len(modeled_x) != len(x_measured):
        raise ValueError("Mismatch in data length or missing measurement data.")
    
    # Calculate the errors
    errors = [
        np.sqrt((mx - x)**2 + (my - y)**2)
        for mx, my, x, y in zip(modeled_x, modeled_y, x_measured, y_measured)
    ]

    error_summary = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'errors': errors
    }

    # Plot the errors
    fig, ax = plt.subplots()
    ax.plot(modeled_x, modeled_y, '-o', label='Modeled Robot', color='blue')
    ax.plot(x_measured, y_measured, '-o', label='Measured Robot', color='orange')
    
    # Add error vectors
    for mx, my, x, y in zip(modeled_x, modeled_y, x_measured, y_measured):
        ax.plot([mx, x], [my, y], '--r', alpha=0.6)  # Error vector
    
    ax.legend()
    ax.set_title(f'Error Analysis (Ft={Ft}N)')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.axis('equal')
    ax.grid(True)
    
    plt.show()
    
    return error_summary
