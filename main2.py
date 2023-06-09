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
user_input = input("Enter a decimal between 0-1 to test: ")
n = 64
theta_percent = float(user_input)
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
    distance = np.linalg.norm(np.array(U1_Rx[i]) - np.array(U1_RIS[i]))
    U1_distances.append(distance)

    # Calculate the path loss
    U1_PathLoss = abs(np.dot(np.dot(U1_G[i], theta), np.transpose(U1_H[i])) + U1_D[i])

    # Calculate the SNR
    U1_SNR = P1 * U1_PathLoss / Noise

    # Calculate the data rate
    data_rate = np.log2(1 + U1_SNR)

    # Add the data rate to its list
    datarates.append(data_rate)

# Calculate max_distance and max_dataRate
print(len(datarates), file=output_file)
print(len(U1_distances), file=output_file)
max_index = np.argmax(datarates)
print(max_index, file=output_file)
max_distance = U1_distances[max_index]
max_dataRate = datarates[max_index]

# Create an array of dictionaries
data_array = []
for i in range(236):
    data_dict = {"Distance": U1_distances[i], "Data Rate": datarates[i]}
    data_array.append(data_dict)

    
# Generate the output file name based on the theta percent
output_file_name = "data_at_{:.3f}.txt".format(theta_percent)

# Append index, theta percent, and maximum data rate to the output file
with open(output_file_name, "w") as output_file:
    for data_dict in data_array:
        output_file.write(str(data_dict) + "\n")
    output_file.write("Max Index: {}\n".format(max_index))
    output_file.write("Theta Percent: {}\n".format(theta_percent))
    output_file.write("Max Data Rate: {}\n".format(max_dataRate))

# # Create a DataFrame
# data_dict = {'Distance': U1_distances, 'Data Rate': datarates}
# df = pd.DataFrame(data_dict)

# # Save DataFrame to a text file
# df.to_csv('data_rates.txt', sep=' ', index=False)

# Plotting
plt.plot(U1_distances, datarates, '--gs', linewidth=2, markersize=10, markeredgecolor='b', markerfacecolor=[0.5,0.5,0.5])
plt.text(max_distance, max_dataRate, 'Max Data Rate = {:.2f}'.format(max_dataRate), fontsize=14, ha='center', va='bottom')

plt.xlabel('Distance')
plt.ylabel('Data Rates')

# Set the graph title
plt.title('Graph for Theta Percent: {:.3f}'.format(theta_percent))

# Save the figure before showing it
plt.savefig('data_rates_plot_{}.png'.format(theta_percent), format='png', dpi=300)

# Show the graph
plt.show()
