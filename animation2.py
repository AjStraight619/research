import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.table import Table

# Define the matrices
n = 4
G = np.random.rand(n, 1) + 1j * np.random.rand(n, 1)
Theta = np.diag(np.exp(1j * np.random.rand(n)))
H = np.random.rand(n, 1) + 1j * np.random.rand(n, 1)

# Create the result matrix
GH = np.matmul(np.matmul(G.conj().T, Theta), H)

# Define a function to draw a matrix


# Define a function to draw a matrix
def draw_matrix(ax, matrix, title):
    # Clear the previous plot
    ax.clear()

    # Create the table
    tb = Table(ax, bbox=[0, 0, 1, 1])

    # Add the cells
    nrows, ncols = matrix.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    for i in range(nrows):
        for j in range(ncols):
            tb.add_cell(i, j, width, height, text=str(
                matrix[i, j]), loc='center')

    # Add the table to the plot
    ax.add_table(tb)

    # Set the title
    ax.set_title(title)


# Create the plot
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# Draw the initial matrices
draw_matrix(ax[0], G.conj().T, "G")
draw_matrix(ax[1], Theta, "Theta")
draw_matrix(ax[2], H, "H")
draw_matrix(ax[3], GH, "GH")

# Define the animation update function


def update(frame):
    # Calculate the current result
    GH_current = np.matmul(
        np.matmul(G.conj().T[:frame+1], Theta[:frame+1, :frame+1]), H[:frame+1])

    # Redraw the matrices
    draw_matrix(ax[0], G.conj().T, "G")
    draw_matrix(ax[1], Theta, "Theta")
    draw_matrix(ax[2], H, "H")
    draw_matrix(ax[3], GH_current, "GH")

    # Highlight the current row/column
    for i in range(frame+1):
        ax[0].patches[i].set_facecolor('red')
        ax[1].patches[i*n + frame].set_facecolor('red')
        ax[2].patches[frame].set_facecolor('red')


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=n, repeat=False)

# Display the animation
plt.show()
