import numpy as np
import matplotlib.pyplot as plt
import os


def read_file(positionFolder, file):
    path = os.path.join(positionFolder, file)
    print(path)  # this line will help us debug the path
    with open(path, 'r') as f:
        if file in ['H.txt', 'G.txt']:
            data = [complex(line.rstrip('\n').replace('i', 'j')) for line in f]
            data = [data[i:i + 64] for i in range(0, len(data), 64)]
        elif file == 'D.txt':
            data = [complex(line.rstrip('\n').replace('i', 'j')) for line in f]
        else:
            data = [tuple(map(float, line.split(','))) for line in f]
    return [data]  # Return the data as a list with a single element

# def calculate_datarates(U1_G, U1_H, U1_D, theta, P1, Noise, d_percent, n=64):
#     U1_G = np.array(U1_G).reshape(n, 1)  # Reshape U1_G to (n, 1)
#     U1_H = np.array(U1_H).reshape(n, 1)  # Reshape U1_H to (n, 1)
#     U1_D = np.array(U1_D).reshape(1, 1)  # Reshape U1_D to (1, 1)

#     print(np.shape(U1_G))
#     print(np.shape(U1_H))
#     print(np.shape(U1_D))
#     print(np.shape(theta))

#     datarates = []
#     theta_values = []


#     for i in range(n):

#         U1_G = np.array(U1_G).reshape(64, 1)  # Reshape U1_G to (64, 1)
#         U1_H = np.array(U1_H).reshape(64, 1)  # Reshape U1_H to (64, 1)
#         U1_D = np.array(U1_D).reshape(1, 1)  # Reshape U1_D to (1, 1)

#         U1_PathLoss = np.dot(np.dot(U1_G.conj().T, theta), U1_H.conj()) + d_percent * np.abs(U1_D)
#         # U1_PathLoss = np.abs(U1_PathLoss)
#         U1_SNR = P1 * U1_PathLoss / Noise
#         R1 = np.log2(1 + U1_SNR)
#         datarates.append(R1)

#         # Print intermediate values
#         if i < 10:
#             print(f"Iteration {i + 1}:")
#             print(f"U1_G: {U1_G[i]}")
#             print(f"U1_H: {U1_H[i]}")
#             print(f"Theta: {theta[i, i]}")
#             print(f"U1_PathLoss: {U1_PathLoss}")
#             print(f"absU1_PathLoss: {np.abs(U1_PathLoss)}")
#             print(f"U1_SNR: {U1_SNR}")
#             print(f"Data Rate: {R1}")
#             print("-------------------------")

#         theta_values.append(np.angle(theta[i, i], deg=True))


#     return theta_values, datarates

def calculate_datarates(U1_G, U1_H, U1_D, theta, P1, Noise, d_percent):
    n_iter = len(U1_G)  # Assuming U1_G and U1_H have the same length

    # Iterate over the number of iterations
    for i in range(n_iter):
        # Get the i-th matrix from U1_G and U1_H
        U1_G_i = np.squeeze(U1_G[i])
        U1_H_i = np.squeeze(U1_H[i])
        U1_PathLoss = np.dot(np.dot(U1_G_i.conj().T, theta), U1_H_i.conj()) + d_percent * np.abs(U1_D)
        # print("Theta from calc function: ", theta)

        # Calculate the absolute value of PathLoss
        absU1_PathLoss = np.abs(U1_PathLoss)

        # Calculate SNR
        U1_SNR = P1 * U1_PathLoss / Noise

        # Calculate data rate
        DataRate = np.log2(1 + U1_SNR)
        
        # Print out the values
        # print("-------------------------")
        # print(f"Iteration {i+1}:")
        # print(f"U1_G at iteration {i+1}: {U1_G_i}")
        # print(f"U1_H at iteration {i+1}: {U1_H_i}")
        # print(f"U1_PathLoss: {U1_PathLoss}")
        # print(f"absU1_PathLoss: {absU1_PathLoss}")
        # print(f"U1_SNR: {U1_SNR}")
        # print(f"Data Rate: {DataRate}")

    return DataRate



def write_output_file(theta_degrees, datarates, position, d_percent):
    max_index = np.argmax(datarates)
    max_datarate = datarates[max_index]
    max_theta = theta_degrees[max_index]
    max_theta_percent = max_theta / 360

    data_array = [{"Theta Percent": theta_degrees[i] / 360, "Theta degrees": theta_degrees[i], "Data Rate": datarates[i]} for i in range(len(datarates))]

    # Make sure the directory exists, if not, create it.
    if not os.path.exists(position):
        os.makedirs(position)

    output_file_name = f"{position}/data_at_{max_theta:.1f}_d_{d_percent}.txt"
    with open(output_file_name, "w") as output_file:
        for data_dict in data_array:
            output_file.write(str(data_dict) + "\n")
        output_file.write("Max Index: {}\n".format(max_index + 1))
        output_file.write("Theta Degrees for highest data rate: {}\n".format(max_theta))
        output_file.write("Theta Percent for the highest data rate: {}\n".format(max_theta_percent))
        output_file.write("Max Data Rate: {}\n".format(max_datarate))
        output_file.write("Line of sight (d_percent): {}\n".format(d_percent))

def generate_plot(theta_degrees, datarates, position, d_percent):
    max_index = np.argmax(datarates)
    max_theta = theta_degrees[max_index]
    max_dataRate = datarates[max_index]

    plt.plot(theta_degrees, datarates, '--gs', linewidth=2, markersize=10, markeredgecolor='b', markerfacecolor=[0.5,0.5,0.5])
    plt.text(max_theta, max_dataRate, 'Max Data Rate = {:.2f}'.format(max_dataRate), fontsize=14, ha='center', va='bottom')

    plt.xlabel('Theta (degrees)')
    plt.ylabel('Data Rates')
    plt.title('Graph for Max Theta: {:.1f}'.format(max_theta))

    plt.savefig(f'{position}/data_rates_plot_{max_theta}_d_{d_percent}.png', format='png', dpi=300)
    plt.show()


def main():
    # Locations for Rx
    locations = {
        "bottom middle": "position_25_18_2",
        "top middle": "position_25_32_2",
        "left middle": "position_18_25_2",
        "right middle": "position_32_25_2",
        "bottom left corner": "position_18_18_2",
        "top left corner": "position_18_32_2",
        "top right corner": "position_32_32_2",
        "bottom right corner": "position_32_18_2",
        "27_23": "position_27_23_2",
        "22_23": "position_22_23_2",
        "25_29": "position_25_29_2",
        "8_20": "position_8_20_2",
        "-8_20": "position_-8_20_2",
        "8_20_1e": "position_8_20_2_(1eRIS)"
    }
    
    while True:
        print(f"Available locations: {', '.join(locations.keys())}")
        chosen_location = input("Choose a location to test from (25, 30 or type 'exit' to quit): ").strip()

        if chosen_location.lower() == "exit":
            break
        elif chosen_location not in locations:
            print("Not a valid location. Please try again.")
            continue
            
        position = locations[chosen_location]
        
        U1_Rx = read_file(position, "Rx.txt")
        U1_RIS = read_file(position, "RIS.txt")  
        U1_Tx = read_file(position, "Tx.txt")  
        U1_D = read_file(position, "D.txt")
        U1_G = read_file(position, "G.txt")
        U1_H = read_file(position, "H.txt")

        n = 64
        Noise = 1e-11
        P = 1
        P1 = 0.5 * P
        theta = np.zeros((n, n), dtype=complex)
        d_percent = 0 # 0 if there is no line of sight, 1 if full line of sight [0:1]

        theta_degrees = np.arange(0, 361, 45)  # Angles from 0 to 360 in steps of 45
        datarates_all = []
        
        # create new theta matrix
        

        for theta_degree in theta_degrees:
            theta_percent = theta_degree / 360
            np.fill_diagonal(theta, np.exp(2*np.pi*theta_percent*1j))
            print("Abs of theta from main:", np.abs(theta))
    
            datarates = calculate_datarates(U1_G, U1_H, U1_D, theta, P1, Noise, d_percent)

            datarates_all.append(datarates)

        # Compute average data rate for each angle
        avg_datarates = [np.mean(datarates) for datarates in datarates_all]
        write_output_file(theta_degrees, avg_datarates, position, d_percent)
        generate_plot(theta_degrees, avg_datarates, position, d_percent)


if __name__ == "__main__":
    main()

