import numpy as np
import matplotlib.pyplot as plt

def plot_robot(theta, ax, Ft):
    theta = np.ravel(theta)
    num = np.size(theta)
    length = 10

    x_coords = [0]
    y_coords = [0] 

    tf_total = np.zeros((num, 3, 3))

    for i in range(num):
        tf = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), length * np.cos(theta[i-1])],
            [np.sin(theta[i]), np.cos(theta[i]), length * np.sin(theta[i-1])],
            [0, 0, 1]
        ])
        if i == 0:
            tf_total[i] = tf
        else:
            tf_total[i] = tf_total[i-1] @ tf
        vector = np.array([
            [length * np.cos(theta[i])],
            [length * np.sin(theta[i])],
            [1]
        ])

        new_x = (tf_total[i] @ vector)[0, 0]
        new_y = (tf_total[i] @ vector)[1, 0]

        x_coords.append(new_x)
        y_coords.append(new_y)

    # Plot the robot
    ax.plot(x_coords, y_coords, '-o', label=f"{Ft}N")
    ax.text(x_coords[-1], y_coords[-1], f'{Ft}N', fontsize=12, ha='left', va='bottom')
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
    ax.legend()

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