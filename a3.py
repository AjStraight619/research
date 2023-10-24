import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.table import Table

# Define a function to draw a matrix


def draw_matrix(ax, matrix, title):
    # Clear the previous plot
    ax.clear()
    ax.axis('off')  # Turn off the axis

    # Create the table
    tb = Table(ax, bbox=[0, 0, 1, 1])

    # Add the cells
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            value = matrix[i, j]
            # If the value is complex, format the real and imaginary parts separately
            if np.iscomplex(value):
                cell_text = "{:.3f}+{:.3f}j".format(value.real, value.imag)
            else:
                cell_text = "{:.3f}".format(value)
            tb.add_cell(i, j, text=cell_text, loc='center')

    # Add the table to the plot
    ax.add_table(tb)

    # Set the title
    ax.set_title(title)


# Same process of defining matrices and animation as before
G = np.random.rand(4, 1) + 1j * np.random.rand(4, 1)
H = np.random.rand(4, 1) + 1j * np.random.rand(4, 1)
Theta = np.exp(1j * np.random.rand(4))

# Initialize the figure and the axes
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

# Define the update function


def update(i):
    theta = np.diag(Theta)
    G_H = np.matmul(np.matmul(G.conj().T, theta), H)
    draw_matrix(axs[0], G, "G")
    draw_matrix(axs[1], np.diag(Theta), "Theta")
    draw_matrix(axs[2], H, "H")
    draw_matrix(axs[3], G_H, "Pathloss")


# Create the animation
ani = FuncAnimation(fig, update, frames=range(10), repeat=True)

# Display the animation
plt.show()
