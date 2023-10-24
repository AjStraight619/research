import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up some initial data
n = 3
G = np.random.rand(n, 1) + 1j*np.random.rand(n, 1)  # n x 1 matrix
H = np.random.rand(n, 1) + 1j*np.random.rand(n, 1)  # n x 1 matrix
Theta = np.random.rand(n, n) + 1j*np.random.rand(n, n)  # n x n matrix

# Since theta is diagonal
Theta_diag = np.diag(np.diag(Theta))

fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# Set up initial tables
tables = [ax.table(cellText=np.full((n, 1), '', dtype=object),
                   loc='center', cellLoc='center') for ax in axs[:-1]]
result_plot, = axs[-1].plot([], [], 'o-')

# Set up initial titles
titles = ['G.conj().T', 'Theta', 'H', 'Multiplication Result']
for ax, title in zip(axs, titles):
    ax.set_title(title)

# Hide axes
for ax in axs:
    ax.axis('off')


def update(i):
    # Update the theta matrix
    Theta_diag = np.diag(np.exp(1j * 2 * np.pi * i / 200. * np.ones(n)))

    # Multiply matrices
    G_H = np.matmul(np.matmul(G.conj().T, Theta_diag), H)

    # Convert complex numbers to strings in polar form
    matrices = [G.conj().T, Theta_diag, H]
    for table, matrix in zip(tables, matrices):
        cell_text = [
            [f'{np.abs(val):.2f}∠{np.angle(val, deg=True):.2f}°' for val in row] for row in matrix]
        table.properties()['celld'].update(
            {(i, j): [cell_text[i][j]] for i in range(n) for j in range(1)})

    # Update result plot
    result_plot.set_data(range(1, n+1), np.abs(G_H).flatten())
    axs[-1].set_xlim(0, n)
    axs[-1].set_ylim(0, np.abs(G_H).max()+1)


ani = FuncAnimation(fig, update, frames=range(200), repeat=True)

plt.show()
