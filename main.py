import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_file(file):
    with open(file, 'r') as f:
        if file in ['H.txt', 'G.txt']:
            data = [complex(line.rstrip('\n').replace('i', 'j')) for line in f]
            data = [data[i:i + 64] for i in range(0, len(data), 64)]
        elif file == 'D.txt':
            data = [complex(line.rstrip('\n').replace('i', 'j')) for line in f]
        else:
            data = [tuple(map(float, line.split(','))) for line in f]
    return data

# Open output.txt in write mode
output_file = open("output.txt", "w")

# Read data from the specified files
U1_Rx = read_file("Rx.txt")
U1_RIS = read_file("RIS.txt")
U1_Tx = read_file("Tx.txt")
U1_D = read_file("D.txt")
U1_G = read_file("G.txt")
U1_H = read_file("H.txt")

# Set some initial values
n = 64
theta_percent = 0.125
Noise = 1e-11
P = 1
P1 = 0.5 * P

# Initialize theta to a 2D array of zeroes
theta = np.zeros((n, n), dtype=complex)
np.fill_diagonal(theta, np.exp(2*np.pi*theta_percent*1j))

# Initialize lists to hold distances, data rates
U1_distances = []
datarates = []

for i in range(236):

    # Calculate the distance and add it to the list of distances
    U1_distances.append(np.linalg.norm(np.array(U1_Rx[i]) - np.array(U1_RIS[i])))

    # Calculate the path loss
    U1_PathLoss = abs(np.dot(np.dot(U1_G[i], theta), np.transpose(U1_H[i])) + U1_D[i])

    # Calculate the SNR
    U1_SNR = P1*U1_PathLoss/Noise

    # Calculate the data rate
    R1 = np.log2(1 + U1_SNR)

    # Add the data rate to its list
    datarates.append(R1)

# Calculate max_distance and max_dataRate
print(len(datarates), file=output_file)
print(len(U1_distances), file=output_file)
max_index = np.argmax(datarates)
print(max_index, file=output_file)
max_distance = U1_distances[max_index]
max_dataRate = datarates[max_index]

# Create a DataFrame
data_dict = {'Distance': U1_distances, 'Data Rate': datarates}
df = pd.DataFrame(data_dict)

# Save DataFrame to a text file
df.to_csv('data_rates.txt', sep=' ', index=False)

# Plotting
plt.plot(U1_distances, datarates, '--gs', linewidth=2, markersize=10, markeredgecolor='b', markerfacecolor=[0.5,0.5,0.5])
plt.text(max_distance, max_dataRate, 'Max Data Rate = {:.2f}'.format(max_dataRate), fontsize=14, ha='center', va='bottom')

plt.xlabel('Distance')
plt.ylabel('Data Rates')

# Save the figure before showing it
plt.savefig('data_rates_plot.png', format='png', dpi=300)

# Show
plt.show()

# Finally, close the output file
output_file.close()















