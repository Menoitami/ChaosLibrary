import numpy as np
import matplotlib.pyplot as plt
import os

workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_dir = os.path.join(workspace_root, 'workspace/bifurcation')
output_dir = os.path.join(workspace_root, 'results')
input_file1 = os.path.join(input_dir, 'Bifurcation_Graph_chameleon.csv')

# Чтение файла с пропуском первой строки (skiprows=1)
a = np.loadtxt(input_file1, delimiter=',', skiprows=1)

# Plot the data
plt.plot(a[:, 0], a[:, 1], 'r.', markersize=0.05)
# Set the labels and title
plt.xlabel('a', fontsize=14)
plt.ylabel('X', fontsize=14)

# Set the figure size
plt.gcf().set_size_inches(12.8, 6.4)
#plt.ylim(0, 7)  # Set x-axis limits from 10 to 0
#plt.xlim(1, 3.5)

# Set the font size and tick label interpreter
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.show()