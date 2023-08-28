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
    return data

# def calculate_theta_and_datarates(U1_Rx, U1_RIS, U1_G, U1_H, U1_D, theta, n, P1, Noise, d_percent):
#     datarates = []
#     theta_values = []

#     for i in range(236):
#         U1_PathLoss = abs(np.dot(np.dot(U1_G[0], theta), np.transpose(U1_H[0])) + d_percent * U1_D[0])
#         U1_SNR = P1 * U1_PathLoss / Noise
#         data_rate = np.log2(1 + U1_SNR)

#         # Add printing for the first 5 iterations
#         if i < 5:
#             print(f"Iteration {i + 1}:")
#             print(f"U1_PathLoss = {U1_PathLoss}")
#             print(f"U1_SNR = {U1_SNR}")
#             print(f"data_rate = {data_rate}")
#             print("-------------------------")

#         datarates.append(data_rate)
#         theta_values.append(np.angle(theta[0, 0], deg=True))  # assuming all diagonal elements have the same phase

#     return theta_values, datarates

# def calculate_theta_and_datarates(U1_Rx, U1_RIS, U1_G, U1_H, U1_D, theta, n, P1, Noise, d_percent):
#     datarates = []
#     theta_values = []

#     for i in range(236):
#         U1_PathLoss = U1_G[0] @ theta @ np.transpose(U1_H[0])
#         U1_RxPower = np.square(np.abs(U1_PathLoss))
#         U1_NoisePower = Noise * np.sum(np.abs(U1_D[0]))
#         U1_SNR = U1_RxPower / U1_NoisePower
#         data_rate = np.log2(1 + U1_SNR)

#         # Add printing for the first 5 iterations
#         if i < 5:
#             print(f"Iteration {i + 1}:")
#             print(f"U1_PathLoss = {U1_PathLoss}")
#             print(f"U1_RxPower = {U1_RxPower}")
#             print(f"U1_NoisePower = {U1_NoisePower}")
#             print(f"U1_SNR = {U1_SNR}")
#             print(f"data_rate = {data_rate}")
#             print("-------------------------")

#         datarates.append(data_rate)
#         theta_values.append(np.angle(theta[0, 0], deg=True))  # assuming all diagonal elements have the same phase

#     return theta_values, datarates


def calculate_datarates(U1_G, U1_H, U1_D, theta, n, P1, Noise, d_percent):
    datarates = []
    
    for G, H in zip(U1_G, U1_H):
        U1_PathLoss = np.abs(np.dot(np.dot(G, theta), np.transpose(H))) + d_percent * np.abs(U1_D)
        U1_SNR = P1 * U1_PathLoss / Noise
        data_rate = np.log2(1 + U1_SNR)
        datarates.append(data_rate)
    
    return datarates



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

def generate_plot(theta_degrees, avg_datarates, position, d_percent):
    max_index = np.argmax(avg_datarates)
    max_theta = theta_degrees[max_index]
    max_dataRate = avg_datarates[max_index]

    # Limit the range of theta_degrees based on the length of avg_datarates
    theta_degrees = theta_degrees[:len(avg_datarates)]

    plt.plot(theta_degrees, avg_datarates, '--gs', linewidth=2, markersize=10, markeredgecolor='b', markerfacecolor=[0.5,0.5,0.5])
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
        print("Length of U1_G: ", len(U1_G))
        print("Length of U1_H: ", len(U1_H))
        print("Length of U1_D: ", len(U1_D))

        n = 64
        Noise = 1e-11
        P = 1
        P1 = 0.5 * P
        theta = np.zeros((n, n), dtype=complex)
        d_percent = 0  # 0 if there is no line of sight, 1 if full line of sight [0:1]

        theta_degrees = np.arange(0, 361, 45)  # Angles from 0 to 360 in steps of 45
        datarates_all = []

        for theta_degree in theta_degrees:
            theta_percent = theta_degree / 360.0
            np.fill_diagonal(theta, np.exp(2 * np.pi * theta_percent * 1j))
            datarates = calculate_datarates(U1_G, U1_H, U1_D, theta, n, P1, Noise, d_percent)
            datarates_all.append(datarates)

        avg_datarates = [np.mean(datarates) for datarates in datarates_all]
        write_output_file(theta_degrees, avg_datarates, position, d_percent)
        generate_plot(theta_degrees, avg_datarates, position, d_percent)


if __name__ == "__main__":
    main()
