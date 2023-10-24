import numpy as np
from scipy.optimize import minimize
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyswarms as ps
import matplotlib.animation as animation

# Constants
NOISE = 1e-11
P = 1
P1 = 0.5 * P
RX = (16, 25)
TX = (0, 25)


def get_data_rate_constants():
    return NOISE, P1


def get_coordinates():
    return RX, TX


def read_complex_file(file_path):
    with open(file_path, "r") as file:
        data = [complex(line.rstrip('\n').replace('i', 'j')) for line in file]
    return np.array(data)


def get_user_inputs():
    folder_name = input("Enter directory name: ")
    x_str, y_str = folder_name.split('_')
    x, y, z = int(x_str), int(y_str), 2
    RIS = (x, y)

    U1_D = read_complex_file(f"{folder_name}/D.txt")
    U1_G = read_complex_file(f"{folder_name}/G.txt")
    U1_H = read_complex_file(f"{folder_name}/H.txt")

    return folder_name, RIS, U1_D, U1_G, U1_H, x_str, y_str, z


folder_name, RIS, U1_D, U1_G, U1_H, x_str, y_str, z = get_user_inputs()


# define lists to store progress
bfgs_progress = []
slsqp_progress = []
pso_progress = []


def calculate_data_rate(x, U1_G, U1_H, is_pso=False):
    if is_pso:
        n_particles = x.shape[0]
        data_rate = []
        for i in range(n_particles):
            theta = np.diag(np.exp(1j * x[i]))
            U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
            U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
            rate = np.log2(1 + U1_SNR)
            data_rate.append(-rate)
        pso_progress.append(np.max(data_rate))
        return np.array(data_rate)
    else:
        theta = np.diag(np.exp(1j * x))
        U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
        U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
        data_rate = np.log2(1 + U1_SNR)
        return data_rate


def calculate_data_rate_bfgs(x):
    data_rate = calculate_data_rate(x, U1_G, U1_H)
    bfgs_progress.append(data_rate)
    return -data_rate


def calculate_data_rate_slsqp(x):
    data_rate = calculate_data_rate(x, U1_G, U1_H)
    slsqp_progress.append(data_rate)
    return -data_rate


def calculate_data_rate_pso(x):
    return calculate_data_rate(x, U1_G, U1_H, is_pso=True)


def calculate_distance(point1, point2):
    points = zip(point1, point2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return round(np.sqrt(sum(diffs_squared_distance)))


def check_supplementary(angle_array1, angle_array2, tolerance=5):
    # Convert angles to degrees
    angle_array1 = np.degrees(angle_array1) % 360
    angle_array2 = np.degrees(angle_array2) % 360

    # Calculate the absolute difference
    diff = np.abs(angle_array1 - angle_array2)

    # Check if the differences are close to 180
    return np.isclose(diff, 180, atol=tolerance)


def perform_optimization(
    data_rate_func, initial_guess, bounds, method='BFGS', tol=0.001
):
    return minimize(data_rate_func, initial_guess, method=method, bounds=bounds, tol=tol)


def perform_pso_optimization(data_rate_func, dimensions, bounds, options, n_particles=10, iters=4000):
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(data_rate_func, iters=iters)
    return cost, pos


# Make an initial guess for the phase shifts
initial_guess = np.zeros(len(U1_G))
bounds = [(0, 2 * np.pi)] * len(U1_G)


# Call the minimize function with BFGS
res1 = perform_optimization(
    calculate_data_rate_bfgs, initial_guess, bounds, method='BFGS')

# Call the minimize function with SLSQP
res2 = perform_optimization(
    calculate_data_rate_slsqp, initial_guess, bounds, method='SLSQP')

phase_shifts_bfgs_rad = res1.x
phase_shifts_slsqp_rad = res2.x

# Check if the phase shifts are supplementary
are_supplementary = check_supplementary(
    phase_shifts_bfgs_rad, phase_shifts_slsqp_rad)


def create_phase_shift_df(rad_values, deg_values, method):
    return pd.DataFrame({
        f'Phase shift ({method}) in radians': rad_values,
        f'Phase shift ({method}) in degrees': deg_values
    })


def write_results_to_file(file_name, df, method, fun_value):
    with open(file_name, 'a') as f:
        f.write(df.to_string())
        f.write('\n')
        f.write(f'Optimized max data rate ({method}): {abs(fun_value)}\n')


# Create DataFrames
df_bfgs = create_phase_shift_df(res1.x, np.rad2deg(res1.x), 'BFGS')
df_slsqp = create_phase_shift_df(res2.x, np.rad2deg(res2.x), 'SLSQP')

# Write results to files
results_file = f'{folder_name}/results.txt'
write_results_to_file(results_file, df_bfgs, 'BFGS', res1.fun)
write_results_to_file(results_file, df_slsqp, 'SLSQP', res2.fun)


def plot_data_rate(progress_bfgs, progress_slsqp,  ris_location, folder_name):
    plt.figure(figsize=(12, 6))
    plt.plot(progress_bfgs, label='BFGS')
    plt.plot(progress_slsqp, label='SLSQP')
    plt.xlabel('Iteration')
    plt.ylabel('Data Rate')
    plt.title(f'Data Rate vs Iteration for RIS location: {ris_location}')
    plt.legend()
    plt.savefig(f"{folder_name}/data_rate_vs_iteration.png")


def create_heatmap(phase_shifts_bfgs, phase_shifts_slsqp, ris_location, folder_name):
    df_bfgs = pd.DataFrame(np.rad2deg(phase_shifts_bfgs),
                           columns=['Phase Shift'])
    df_bfgs['Method'] = 'BFGS'
    df_slsqp = pd.DataFrame(np.rad2deg(
        phase_shifts_slsqp), columns=['Phase Shift'])
    df_slsqp['Method'] = 'SLSQP'

    df = pd.concat([df_bfgs, df_slsqp])  # Include PSO data

    # plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.pivot(columns='Method', values='Phase Shift'), fmt=".1f")
    plt.title(f'Optimal Phase Shifts for RIS location: {ris_location}')
    plt.savefig(f"{folder_name}/optimal_phase_shifts.png")


# Plot data rate
ris_location = (int(x_str), int(y_str), z)
plot_data_rate(bfgs_progress, slsqp_progress,
               ris_location, folder_name)

# Create heatmap
create_heatmap(res1.x, res2.x, ris_location, folder_name)
complex_numbers_G = U1_G[:10]
complex_numbers_H = U1_H[:10]

# Create a new figure
plt.figure(figsize=(8, 8))

# Plot the numbers from matrix G in one color
plt.scatter(complex_numbers_G.real, complex_numbers_G.imag,
            color='blue', label='G')

# Plot the numbers from matrix H in another color
plt.scatter(complex_numbers_H.real,
            complex_numbers_H.imag, color='red', label='H')

# Add labels and a legend
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("Constellation Diagram for G and H")
plt.legend()
plt.savefig(f'{folder_name}/constellation_diagram.png')


def plot_phase_shifts_3d(phase_shifts, plot_title):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    grid_size = int(np.sqrt(len(phase_shifts)))

    x, y = np.meshgrid(range(grid_size), range(grid_size))
    z = np.zeros_like(x)
    dx = dy = np.ones_like(x)
    dz = phase_shifts.reshape((grid_size, grid_size))

    colors = plt.cm.viridis(
        (phase_shifts - phase_shifts.min()) / (phase_shifts.ptp()))

    ax.bar3d(x.ravel(), y.ravel(), z.ravel(), dx.ravel(),
             dy.ravel(), dz.ravel(), color=colors)

    ax.set_title(plot_title)
    plt.savefig(f"{folder_name}/{plot_title.replace(' ', '_')}.png")


plot_phase_shifts_3d(res1.x, '3D Phase Shifts for BFGS')
plot_phase_shifts_3d(res2.x, '3D Phase Shifts for SLSQP')
