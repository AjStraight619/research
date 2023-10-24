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
        data = []
        for line in file:
            data.append(complex(line.rstrip('\n').replace('i', 'j')))
    return np.array(data)


folder_name = input("Enter directory name: ")
x_str, y_str = folder_name.split('_')
z = 2
x = int(x_str)
y = int(y_str)
RIS = (x, y)


U1_D = read_file(f"{folder_name}/D.txt")
U1_G = read_file(f"{folder_name}/G.txt")
U1_H = read_file(f"{folder_name}/H.txt")


def calculate_data_rate_bfgs(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
    data_rate = np.log2(1 + U1_SNR)
    bfgs_progress.append(data_rate)  # negative because we are minimizing
    return -data_rate


def calculate_data_rate_slsqp(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / NOISE
    data_rate = np.log2(1 + U1_SNR)
    slsqp_progress.append(data_rate)  # negative because we are minimizing
    return -data_rate


def calculate_data_rate_pso(x):
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


def calculate_distance(point1, point2):
    x1, y1 = point1
    print("Point1:", x1, y1)
    x2, y2 = point2
    print("Point2:", x2, y2)
    distance = round(np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2))
    return distance


def check_supplementary(angle_array1, angle_array2):
    # Convert angles to degrees
    angle_array1 = np.degrees(angle_array1) % 360
    angle_array2 = np.degrees(angle_array2) % 360

    # Calculate the absolute difference
    diff = np.abs(angle_array1 - angle_array2)

    # Check if the differences are close to 180
    return np.isclose(diff, 180, atol=5)  # atol specifies the tolerance


# Make an initial guess for the phase shifts
x0 = np.zeros(len(U1_G))

bounds = [(0, 2 * np.pi)] * len(U1_G)

max_bound = 2 * np.pi * np.ones(len(U1_G))
min_bound = np.zeros(len(U1_G))
bounds_pso = (min_bound, max_bound)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# Initialize the optimizer
optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=len(U1_G), options=options, bounds=bounds_pso)

# Perform optimization
cost, pos = optimizer.optimize(calculate_data_rate_pso, iters=4000)


phase_shifts_pso_rad = pos
phase_shifts_pso_deg = np.rad2deg(phase_shifts_pso_rad)

# Call the minimize function with BFGS
res1 = minimize(calculate_data_rate_bfgs, x0, method='BFGS', tol=0.001)

# Call the minimize function with SLSQP
res2 = minimize(calculate_data_rate_slsqp, x0,
                method='SLSQP', bounds=bounds, tol=0.001)

phase_shifts_bfgs_rad = res1.x
phase_shifts_slsqp_rad = res2.x

# Check if the phase shifts are supplementary
are_supplementary = check_supplementary(
    phase_shifts_bfgs_rad, phase_shifts_slsqp_rad)

# # Print results
# for i, is_supplementary in enumerate(are_supplementary):
#     print(
#         f'Angles at index {i} are {"close to being supplementary" if is_supplementary else "not close to being supplementary"}')

# dist_RIS_to_Rx = calculate_distance(RIS, RX)
# dist_RIS_to_Tx = calculate_distance(RIS, TX)
# print("RIS to Rx:", dist_RIS_to_Rx)
# print("RIS to Tx:", dist_RIS_to_Tx)


with open(f'{folder_name}/results.txt', 'w') as f:
    diag_elements_bfgs_rad = res1.x
    diag_elements_bfgs_deg = np.rad2deg(diag_elements_bfgs_rad)
    df_bfgs = pd.DataFrame({'Phase shift (BFGS) in radians': diag_elements_bfgs_rad,
                            'Phase shift (BFGS) in degrees': diag_elements_bfgs_deg})

    diag_elements_slsqp_rad = res2.x
    diag_elements_slsqp_deg = np.rad2deg(diag_elements_slsqp_rad)
    df_slsqp = pd.DataFrame({'Phase shift (SLSQP) in radians': diag_elements_slsqp_rad,
                             'Phase shift (SLSQP) in degrees': diag_elements_slsqp_deg})

    f.write(df_bfgs.to_string())
    f.write('\n')
    f.write(f'Optimized max data rate (BFGS): {-res1.fun}\n')
    f.write(df_slsqp.to_string())
    f.write('\n')
    f.write(f'Optimized max data rate (SLSQP): {-res2.fun}\n')

# Create a DataFrame for the phase shifts calculated by PSO
diag_elements_pso_rad = phase_shifts_pso_rad
diag_elements_pso_deg = phase_shifts_pso_deg
df_pso = pd.DataFrame({'Phase shift (PSO) in radians': diag_elements_pso_rad,
                       'Phase shift (PSO) in degrees': diag_elements_pso_deg})

with open(f'{folder_name}/results.txt', 'a') as f:
    f.write(df_pso.to_string())
    f.write('\n')
    f.write(f'Optimized max data rate (PSO): {abs(cost)}\n')


# Append the PSO progress to the graph
plt.figure(figsize=(12, 6))
plt.plot(bfgs_progress, label='BFGS')
plt.plot(slsqp_progress, label='SLSQP')
plt.plot([-i for i in optimizer.cost_history], label='PSO')
plt.xlabel('Iteration')
plt.ylabel('Data Rate')
plt.title(f'Data Rate vs Iteration for RIS location: ({x_str}, {y_str}, {z})')
plt.legend()
plt.savefig(f"{folder_name}/data_rate_vs_iteration.png")

# heatmap
# convert the angles to degrees
phase_shifts_bfgs = np.rad2deg(res1.x)
phase_shifts_slsqp = np.rad2deg(res2.x)
phase_shifts_pso = np.rad2deg(phase_shifts_pso_rad)

# prepare a dataframe for the heatmap
df_bfgs = pd.DataFrame(phase_shifts_bfgs, columns=['Phase Shift'])
df_bfgs['Method'] = 'BFGS'
df_slsqp = pd.DataFrame(phase_shifts_slsqp, columns=['Phase Shift'])
df_slsqp['Method'] = 'SLSQP'
df = pd.concat([df_bfgs, df_slsqp])
df_pso = pd.DataFrame(phase_shifts_pso, columns=['Phase Shift'])
df_pso['Method'] = 'PSO'
df = pd.concat([df_bfgs, df_slsqp, df_pso])  # Include PSO data

# plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.pivot(columns='Method', values='Phase Shift'),
            fmt=".1f")
plt.title(f'Optimal Phase Shifts for RIS location: ({x_str}, {y_str}, {z})')
plt.savefig(f"{folder_name}/optimal_phase_shifts.png")
