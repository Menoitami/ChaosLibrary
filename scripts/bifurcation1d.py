import numpy as np
import matplotlib.pyplot as plt
import os

workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_dir = os.path.join(workspace_root, 'workspace/bifurcation/Bifurcation1D')
output_dir = os.path.join(workspace_root, 'results')
input_file1 = os.path.join(input_dir, 'Bifurcation1D_test.csv')

a = np.loadtxt(input_file1, delimiter=',')

# Plot the data
plt.plot(a[:, 0], a[:, 1], 'r.', markersize=2)

# Set the labels and title
plt.xlabel('c', fontsize=24)
plt.ylabel('X', fontsize=24)

# Set the figure size
plt.gcf().set_size_inches(12.8, 6.4)

# Set the font size and tick label interpreter
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Show the plot
plt.show()