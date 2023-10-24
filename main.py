import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_file(file_path):
    with open(file_path, "r") as file:
        data = []
        for line in file:
            try:
                data.append(complex(line.rstrip('\n').replace('i', 'j')))
            except ValueError:
                print(f"Problem with line: '{line}'")
    return np.array(data)


folder_name = input("Enter directory name: ")

U1_D = read_file(f"{folder_name}/D.txt")
U1_G = read_file(f"{folder_name}/G.txt")
U1_H = read_file(f"{folder_name}/H.txt")

print(len(U1_G))

Noise = 1e-11
P = 1
P1 = 0.5 * P

# Define a list to store data rate progress for BFGS
bfgs_progress = []
# Define a list to store data rate progress for SLSQP
slsqp_progress = []


def calculate_data_rate_bfgs(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
    data_rate = np.log2(1 + U1_SNR)
    bfgs_progress.append(data_rate)
    return -data_rate


def calculate_data_rate_slsqp(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
    data_rate = np.log2(1 + U1_SNR)
    slsqp_progress.append(data_rate)
    return -data_rate


x0 = np.zeros(len(U1_G))

bounds = [(0, 2 * np.pi)] * len(U1_G)

res1 = minimize(calculate_data_rate_bfgs, x0, method='BFGS')
res2 = minimize(calculate_data_rate_slsqp, x0, method='SLSQP', bounds=bounds)

with open(f'{folder_name}/results.txt', 'w') as f:
    f.write(
        f'Optimized phase shift combination (BFGS) [in radians]: {res1.x}\n')
    f.write(
        f'Optimized phase shift combination (BFGS) [in degrees]: {np.rad2deg(res1.x)}\n')
    f.write(f'Optimized max data rate (BFGS): {-res1.fun}\n')
    f.write(
        f'Optimized phase shift combination (SLSQP) [in radians]: {res2.x}\n')
    f.write(
        f'Optimized phase shift combination (SLSQP) [in degrees]: {np.rad2deg(res2.x)}\n')
    f.write(f'Optimized max data rate (SLSQP): {-res2.fun}\n')

plt.figure(figsize=(12, 6))
plt.plot(bfgs_progress, label='BFGS')
plt.plot(slsqp_progress, label='SLSQP')
plt.xlabel('Iteration')
plt.ylabel('Data Rate')
plt.legend()
plt.savefig(f"{folder_name}/data_rate_vs_iteration.png")

phase_values = np.linspace(0, 2 * np.pi, 100)
phase_grid = np.meshgrid(phase_values, phase_values)

Z_bfgs = np.array([calculate_data_rate_bfgs([x, y])
                   for x, y in zip(np.ravel(phase_grid[0]), np.ravel(phase_grid[1]))])
Z_bfgs = Z_bfgs.reshape(phase_grid[0].shape)

Z_slsqp = np.array([calculate_data_rate_slsqp([x, y])
                    for x, y in zip(np.ravel(phase_grid[0]), np.ravel(phase_grid[1]))])
Z_slsqp = Z_slsqp.reshape(phase_grid[0].shape)


fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(phase_grid[0], phase_grid[1], Z_bfgs.reshape(
    phase_grid[0].shape), cmap='viridis')
ax1.set_xlabel('Phase shift 1')
ax1.set_ylabel('Phase shift 2')
ax1.set_zlabel('Data rate')
ax1.set_title('Data rate for BFGS')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(phase_grid[0], phase_grid[1], Z_slsqp.reshape(
    phase_grid[0].shape), cmap='viridis')
ax2.set_xlabel('Phase shift 1')
ax2.set_ylabel('Phase shift 2')
ax2.set_zlabel('Data rate')
ax2.set_title('Data rate for SLSQP')

plt.savefig(f"{folder_name}/phase_shift_vs_data_rate.png")
plt.show()
