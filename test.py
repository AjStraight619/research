import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Define a function to draw a matrix


def draw_matrix(ax, matrix, title):
    ax.clear()
    ax.axis('tight')
    ax.axis('off')

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


# Test the draw_matrix function
fig, ax = plt.subplots()

G = np.random.rand(4, 1) + 1j * np.random.rand(4, 1)
draw_matrix(ax, G, "G")

plt.show()
