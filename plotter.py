import numpy as np
import matplotlib.pyplot as plt

# def plotter(Ft_values, theta_solutions):
#     num = np.size(theta_solutions[1])
#     theta_solutions_sum = np.cumsum(theta_solutions, axis=1)


#     for Ft in [20, 30, 40, 60, 80, 100]:
#         initial_guess = initial_guess_gen(Ft)

#         solution = fsolve(lambda vars: equations(vars, Ft), initial_guess)

#         theta_plot = (solution[:num])
#         plot_robot(theta_plot, axs[1, 0], Ft)

    # for i in range(int(num/2)):
    #     axs[1, 1].plot(Ft_values, reverse_momentum[:, i], label=f"M{2*i+1}")
    

    # axs[1, 1].set_xlabel('Ft (N)')
    # axs[1, 1].set_ylabel('Theta (degrees)')
    # axs[1, 1].set_title('Reverse Momentum')
    # axs[1, 1].legend()
    # axs[1, 1].grid(True)



def plot_robot(theta_radians, ax, Ft):
    num = np.size(theta_radians)
    link_lengths = [0.1 for _ in range(num)]

    # theta_radians = np.radians(theta_degrees)  # Convert angles to radians
    x_coords = [0]  # x-coordinate of the base (origin)
    y_coords = [0]  # y-coordinate of the base (origin)

    # Calculate the position of each joint
    for i, length in enumerate(link_lengths):
        x_coords.append(x_coords[-1] + length * np.cos(np.sum(theta_radians[:i+1])))
        y_coords.append(y_coords[-1] + length * np.sin(np.sum(theta_radians[:i+1])))

    # Plot the robot
    ax.plot(x_coords, y_coords, '-o', label=f"{Ft}N")
    ax.text(x_coords[-1], y_coords[-1], f'{Ft}N', fontsize=12, ha='left', va='bottom')
    ax.axis('equal')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.set_title(f'Robot Configurations')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_ylim(0, 0.8)
    ax.set_xticks(np.arange(-0.1, 0.8, 0.1))
    ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    ax.grid(True)
    ax.legend()

def plot_theta(Ft_values, theta_solutions, ax):
    num = np.size(theta_solutions[0])
    for i in range(num):
        ax.plot(Ft_values, theta_solutions[:, i], label=f"Theta {i+1}")
    
    ax.set_xlabel('Ft (N)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('Variation of Theta with Ft')
    ax.legend()
    ax.grid(True)

def plot_theta_sum(Ft_values, theta_solutions, ax):
    num = np.size(theta_solutions[0])
    theta_solutions_sum = np.cumsum(theta_solutions, axis= 1)
    for i in range(num):
        ax.plot(Ft_values, theta_solutions_sum[:, i], label=f"Theta {i+1}")
    
    ax.set_xlabel('Ft (N)')
    ax.set_ylabel('Cumulative Theta (degrees)')
    ax.set_title('Cumulative Sum of Theta with Ft')
    ax.legend()
    ax.grid(True)