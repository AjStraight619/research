import numpy as np
from itertools import product
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt


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

# Rest of your code...

# Phase shift values to try for each RIS element
# 16 equally spaced values from 0 to pi
phase_shifts = np.linspace(0, np.pi, num=16)

# identity_matrix = np.eye(4)

# Generate all possible combinations of phase shifts for the 4 RIS elements
# all_combinations = product(phase_shifts, repeat=4)

max_data_rate = 0
best_combination = None

Noise = 1e-11
P = 1
P1 = 0.5 * P

# for combination in all_combinations:
#     # Construct theta matrix for this combination
#     theta = np.diag(np.exp(1j * np.array(combination)))

#     # Calculate pathloss for this combination
#     U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)

#     U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
#     data_rate = np.log2(1 + U1_SNR)

#     if data_rate > max_data_rate:
#         max_data_rate = data_rate
#         best_combination = combination

# print('Best phase shift combination (Brute force):', best_combination)
# print('Max data rate (Brute force):', max_data_rate)

# U1_PathLoss2 = np.matmul(np.matmul(U1_G.conj().T, identity_matrix), U1_H)
# U1_SNR = P1 * np.abs(U1_PathLoss2) / Noise
# data_rate2 = np.log2(1 + U1_SNR)

# print("Identity matrix: ", data_rate2)

# New optimization method


# Define a list to store data rate progress for BFGS
bfgs_progress = []
# Define a list to store data rate progress for SLSQP
slsqp_progress = []


def calculate_data_rate_bfgs(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
    data_rate = np.log2(1 + U1_SNR)
    bfgs_progress.append(data_rate)  # add current data rate to the list
    return -data_rate  # return negative data rate for minimization


def calculate_data_rate_slsqp(x):
    theta = np.diag(np.exp(1j * x))
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, theta), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
    data_rate = np.log2(1 + U1_SNR)
    slsqp_progress.append(data_rate)  # add current data rate to the list
    return -data_rate  # return negative data rate for minimization

# calculate using the identity matrix because this is essentially what we were doing before


def calculate_data_rate_identity_matrix(identity_matrix):
    U1_PathLoss = np.matmul(np.matmul(U1_G.conj().T, identity_matrix), U1_H)
    U1_SNR = P1 * np.abs(U1_PathLoss) / Noise
    data_rate = np.log2(1 + U1_SNR)
    return data_rate


# Make an initial guess for the phase shifts
x0 = np.zeros(len(U1_G))

identity_matrix = np.eye(len(U1_G))

bounds = [(0, 2 * np.pi)] * len(U1_G)

# Call the minimize function with BFGS
res1 = minimize(calculate_data_rate_bfgs, x0, method='BFGS')

# Call the minimize function with SLSQP
res2 = minimize(calculate_data_rate_slsqp, x0, method='SLSQP', bounds=bounds)

og_data_rate = calculate_data_rate_identity_matrix(identity_matrix)


with open(f'{folder_name}/results.txt', 'w') as f:
    # f.write(
    #     f'Best phase shift combination (Brute force) [in radians]: {best_combination}\n')
    # f.write(f'Max data rate (Brute force): {max_data_rate}\n')
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
    f.write(f'Identity matrix data rate: {og_data_rate}')


plt.figure(figsize=(12, 6))
plt.plot(bfgs_progress, label='BFGS')
plt.plot(slsqp_progress, label='SLSQP')
plt.xlabel('Iteration')
plt.ylabel('Data Rate')
plt.legend()

# Save the figure before displaying it
os.makedirs(folder_name, exist_ok=True)
plt.savefig(f"{folder_name}/data_rate_vs_iteration.png")
