import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyswarms as ps

# Constants
NOISE = 1e-11
P = 1
P1 = 0.5 * P
RX = (16, 25)
TX = (0, 25)


# define lists to store progress
bfgs_progress = []
slsqp_progress = []
pso_progress = []


def read_file(file_path):
    with open(file_path, "r") as file:
        data = [complex(line.rstrip('\n').replace('i', 'j')) for line in file]
    return np.array(data)


def calculate_data_rate(x, U1_G, U1_H, progress_list):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
    data_rate = np.log2(1 + U1_SNR)
    progress_list.append(data_rate)
    return -data_rate


def calculate_data_rate_pso(x, U1_G, U1_H, progress_list):
    n_particles = x.shape[0]
    data_rate = []
    for i in range(n_particles):
        theta = np.diag(np.exp(1j * x[i]))
        U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
        U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
        rate = np.log2(1 + U1_SNR)
        data_rate.append(-rate)
    progress_list.append(np.max(data_rate))
    return np.array(data_rate)


def calculate_distance(point1, point2):
    x1, y1 = point1
    print("Point1:", x1, y1)
    x2, y2 = point2
    print("Point2:", x2, y2)
    distance = round(np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2))
    return distance


def check_supplementary(angle_array1, angle_array2):
    angle_array1 = np.degrees(angle_array1) % 360
    angle_array2 = np.degrees(angle_array2) % 360
    diff = np.abs(angle_array1 - angle_array2)
    return np.isclose(diff, 180, atol=5)  # atol specifies the tolerance


def get_initial_guess(U1_G):
    return np.zeros(len(U1_G))


def get_bounds(U1_G):
    max_bound = 2 * np.pi * np.ones(len(U1_G))
    min_bound = np.zeros(len(U1_G))
    return (min_bound, max_bound)


def optimize_phase_shifts(U1_G, U1_H):
    # Make an initial guess for the phase shifts
    x0 = get_initial_guess(U1_G)
    bounds = get_bounds(U1_G)

    # Call the minimize function with BFGS
    res1 = minimize(calculate_data_rate, x0, args=(U1_G, U1_H, bfgs_progress),
                    method='BFGS', tol=0.001)

    # Call the minimize function with SLSQP
    res2 = minimize(calculate_data_rate, x0, args=(U1_G, U1_H, slsqp_progress),
                    method='SLSQP', bounds=bounds, tol=0.001)

    # Initialize the optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=len(U1_G),
                                        options=options, bounds=bounds)

    # Perform optimization with PSO
    cost, pos = optimizer.optimize(
        calculate_data_rate_pso, iters=4000, args=(U1_G, U1_H, pso_progress))

    phase_shifts_bfgs_rad = res1.x
    phase_shifts_slsqp_rad = res2.x
    phase_shifts_pso_rad = pos

    phase_shifts_bfgs_deg = np.rad2deg(phase_shifts_bfgs_rad)
    phase_shifts_slsqp_deg = np.rad2deg(phase_shifts_slsqp_rad)
    phase_shifts_pso_deg = np.rad2deg(phase_shifts_pso_rad)

    return (phase_shifts_bfgs_rad, phase_shifts_slsqp_rad, phase_shifts_pso_rad,
            phase_shifts_bfgs_deg, phase_shifts_slsqp_deg, phase_shifts_pso_deg)


def write_results_to_file(phase_shifts_bfgs_rad, phase_shifts_slsqp_rad, phase_shifts_pso_rad,
                          phase_shifts_bfgs_deg, phase_shifts_slsqp_deg, phase_shifts_pso_deg,
                          bfgs_fun, slsqp_fun, pso_cost, folder_name):
    with open(f'{folder_name}/results.txt', 'w') as f:
        df_bfgs = pd.DataFrame({'Phase shift (BFGS) in radians': phase_shifts_bfgs_rad,
                                'Phase shift (BFGS) in degrees': phase_shifts_bfgs_deg})
        df_slsqp = pd.DataFrame({'Phase shift (SLSQP) in radians': phase_shifts_slsqp_rad,
                                 'Phase shift (SLSQP) in degrees': phase_shifts_slsqp_deg})
        df_pso = pd.DataFrame({'Phase shift (PSO) in radians': phase_shifts_pso_rad,
                               'Phase shift (PSO) in degrees': phase_shifts_pso_deg})

        f.write(df_bfgs.to_string())
        f.write('\n')
        f.write(f'Optimized max data rate (BFGS): {-bfgs_fun}\n')
        f.write(df_slsqp.to_string())
        f.write('\n')
        f.write(f'Optimized max data rate (SLSQP): {-slsqp_fun}\n')
        f.write(df_pso.to_string())
        f.write('\n')
        f.write(f'Optimized max data rate (PSO): {abs(pso_cost)}\n')


def plot_data_rate_progress(bfgs_progress, slsqp_progress, pso_progress, folder_name, x_str, y_str, z):
    plt.figure(figsize=(12, 6))
    plt.plot(bfgs_progress, label='BFGS')
    plt.plot(slsqp_progress, label='SLSQP')
    plt.plot([-i for i in pso_progress], label='PSO')
    plt.xlabel('Iteration')
    plt.ylabel('Data Rate')
    plt.title(
        f'Data Rate vs Iteration for RIS location: ({x_str}, {y_str}, {z})')
    plt.legend()
    plt.savefig(f"{folder_name}/data_rate_vs_iteration.png")


def plot_heatmap(phase_shifts_bfgs_deg, phase_shifts_slsqp_deg, folder_name, x_str, y_str, z):
    df_bfgs = pd.DataFrame(phase_shifts_bfgs_deg, columns=['Phase Shift'])
    df_bfgs['Method'] = 'BFGS'
    df_slsqp = pd.DataFrame(phase_shifts_slsqp_deg, columns=['Phase Shift'])
    df_slsqp['Method'] = 'SLSQP'
    df = pd.concat([df_bfgs, df_slsqp])

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.pivot(columns='Method', values='Phase Shift'), fmt=".1f")
    plt.title(
        f'Optimal Phase Shifts for RIS location: ({x_str}, {y_str}, {z})')
    plt.savefig(f"{folder_name}/optimal_phase_shifts.png")


def main():
    # Ask for user input
    folder_name = input("Enter directory name: ")
    x_str, y_str = folder_name.split('_')
    z = 2
    x = int(x_str)
    y = int(y_str)
    RIS = (x, y)

    U1_D = read_file(f"{folder_name}/D.txt")
    U1_G = read_file(f"{folder_name}/G.txt")
    U1_H = read_file(f"{folder_name}/H.txt")

    # Perform optimization
    (phase_shifts_bfgs_rad, phase_shifts_slsqp_rad, phase_shifts_pso_rad,
     phase_shifts_bfgs_deg, phase_shifts_slsqp_deg, phase_shifts_pso_deg) = optimize_phase_shifts(U1_G, U1_H)

    # Check if the phase shifts are supplementary
    are_supplementary = check_supplementary(
        phase_shifts_bfgs_rad, phase_shifts_slsqp_rad)

    # Write results to a file
    bfgs_fun = calculate_data_rate(phase_shifts_bfgs_rad, U1_G, U1_H, [])
    slsqp_fun = calculate_data_rate(phase_shifts_slsqp_rad, U1_G, U1_H, [])
    write_results_to_file(phase_shifts_bfgs_rad, phase_shifts_slsqp_rad, phase_shifts_pso_rad,
                          phase_shifts_bfgs_deg, phase_shifts_slsqp_deg, phase_shifts_pso_deg,
                          bfgs_fun, slsqp_fun, optimizer.cost_history[-1], folder_name)

    # Append the PSO progress to the graph
    plot_data_rate_progress(bfgs_progress, slsqp_progress, optimizer.cost_history,
                            folder_name, x_str, y_str, z)

    # plot the heatmap
    plot_heatmap(phase_shifts_bfgs_deg, phase_shifts_slsqp_deg,
                 folder_name, x_str, y_str, z)


if __name__ == "__main__":
    main()
