import numpy as np

U1_H = np.array([
    -1.50424749287885e-05 - 6.67491707955173e-05j,
    1.95843996093799e-05 + 6.56050588580419e-05j,
    -2.4034563884205e-05 - 6.4141519797769e-05j,
    2.83702333923129e-05 + 6.23659679927715e-05j,
    -3.25692692380274e-05 - 6.02874416974569e-05j,
    3.66102440288187e-05 + 5.79165556401655e-05j,
    -4.04725541718313e-05 - 5.52654453357838e-05j,
    4.41365278383051e-05 + 5.23477034064287e-05j,
    -1.5100354052841e-05 - 6.66172240395925e-05j,
    1.96391380045901e-05 + 6.54717361752412e-05j
])

U1_G = np.array([
    0.000172202724775537 - 4.0240058235643e-05j,
    -7.22844267383715e-05 + 0.000161393935319863j,
    -8.46373894409372e-05 - 0.000155272512105493j,
    0.000174814137806918 + 2.67031445667175e-05j,
    -0.000127132410625358 + 0.000122924329876199j,
    -2.08059693725409e-05 - 0.000175613645100249j,
    0.000152336756392146 + 8.98139931126949e-05j,
    -0.00016373473682334 + 6.68129977198956e-05j,
    0.000172202724775537 - 4.0240058235643e-05j,
    -7.22844267383715e-05 + 0.000161393935319863j
])

theta1 = np.eye(len(U1_G))  # Identity matrix
# theta2 = np.ones((len(U1_G), len(U1_G))) # Matrix filled with ones

theta_degrees = np.arange(0, 361, 45)

for theta_degree in theta_degrees:
    theta_percent = theta_degree / 360
    theta3 = np.zeros((10, 10), dtype=complex)
    np.fill_diagonal(theta3, np.exp(2*np.pi*theta_percent*1j))
    print(theta3)
print(theta1)

UH1 = -1.50424749287885e-05 - 6.67491707955173e-05j
THETA1 = theta3[1]
print(theta3[0])
print("ABS for theta * uh1: ", np.abs(np.dot(UH1, THETA1)))
print(UH1 * THETA1)
print(abs(UH1 * THETA1))


U1_PathLoss1 = np.dot(np.dot(U1_G.conj().T, theta1), U1_H.conj())
# U1_PathLoss2 = np.dot(np.dot(U1_G.conj().T, theta2), U1_H.conj())
U1_PathLoss3 = np.dot(np.dot(U1_G.conj().T, theta3), U1_H.conj())

print("U1_PathLoss for theta1: ", U1_PathLoss1)
# print("U1_PathLoss for theta2: ", U1_PathLoss2)
print("U1_PathLoss for theta3: ", U1_PathLoss3)
print("ABS U1_PathLoss for theta1: ", np.abs(U1_PathLoss1))
# print("ABS U1_PathLoss for theta2: ", np.abs(U1_PathLoss2))
print("ABS U1_PathLoss for theta3: ", np.abs(U1_PathLoss3))


magnitude_factor = 0.5  # reduce the signal by half for every 90 degrees
theta_degrees = np.arange(0, 361, 45)
for theta_degree in theta_degrees:
    theta_percent = theta_degree / 360
    magnitude_reduction = magnitude_factor ** (theta_degree // 90)
    theta = np.zeros((10, 10), dtype=complex)
    np.fill_diagonal(theta, magnitude_reduction *
                     np.exp(2*np.pi*theta_percent*1j))


# print(np.abs(U1_G[0] * U1_H[0]))
# print(np.abs(U1_G[1] * U1_H[0]))
# print(np.abs(U1_G[0] * U1_H[1]))
# print(np.abs(U1_G[1] * U1_H[1]))


# n = 10
# Noise = 1e-11
# P = 1
# P1 = 0.5 * P

# theta_percent_values = [i / 360 for i in range(0, 360, 45)]
# theta = np.diag([np.exp(2 * np.pi * theta_percent_values[i % len(theta_percent_values)] * 1j) for i in range(n)])

# # We should reshape U1_H to be 2D array (10, 1)
# U1_H = U1_H.reshape(-1, 1)

# datarates = []

# for i in range(n):
#     # create a theta matrix with only the i-th diagonal element set
#     theta_single = np.zeros_like(theta, dtype=np.complex128)
#     theta_single[i, i] = theta[i, i]

#     # calculate path loss for a single angle
#     result = np.dot(U1_G, np.dot(theta_single, U1_H))
#         # Calculate path loss for a single angle
#     U1_PathLoss = np.abs(result)

#     # Calculate the SNR
#     U1_SNR = P1 * U1_PathLoss / Noise

#     # Calculate the data rate
#     R1 = np.log2(1 + U1_SNR)

#     # Append the data rate to the list
#     datarates.append(R1)

# # Now print the calculated data rates
# for i, rate in enumerate(datarates):
#     print(f"Data rate for angle {theta_percent_values[i % len(theta_percent_values)] * 360} degrees: {rate}")
