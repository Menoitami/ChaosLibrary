#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def load_data(filepath):
    """Load data from CSV files and process it similar to MATLAB's csvread, skipping first two lines"""
    data = []
    with open(filepath, 'r') as f:
        # Skip first two lines
        next(f)
        next(f)
        for line in f:
            row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
            if row:
                data.append(row)
    return np.array(data)

def create_custom_colormap(N_colormap, max_idx):
    """Create a custom colormap similar to MATLAB's slanCM"""
    colors = []
    for i in range(max_idx):
        # Create a color gradient from blue to red
        r = i / max_idx
        g = 0.5
        b = 1 - i / max_idx
        colors.append((r, g, b))
    return LinearSegmentedColormap.from_list('custom', colors, N=N_colormap)

def main():
    # Set font size
    FONTsize = 12
    
    # Set paths
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    input_dir = os.path.join(workspace_root, 'workspace\\basins')
    path1 = os.path.join(input_dir, 'basinsOfAttraction_test.csv')
    path2 = os.path.join(input_dir, 'basinsOfAttraction_test_old.csv')

    # Load data
    idx = load_data(path1)  # basins
    a = load_data(path1 + "_1.csv")  # avgPeaks
    b = load_data(path1 + "_2.csv")  # avgInterval
    c = load_data(path1 + "_3.csv")  # helphulArray

    idx_old = load_data(path2)  # basins
    a_old = load_data(path2 + "_1.csv")  # avgPeaks
    b_old = load_data(path2 + "_2.csv")  # avgInterval
    c_old = load_data(path2 + "_3.csv")  # helphulArray

    # Create result directory if it doesn't exist
    result_dir = os.path.join(workspace_root, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Process data similar to MATLAB code
    x = np.linspace(a[0,0], a[0,1], a.shape[1])
    y = np.linspace(a[1,0], a[1,1], a.shape[0])
    x = x[:-1]
    y = y[:-1]
    
    # Remove first two rows and last column
    a = a[2:, :-1]
    b = b[2:, :-1]
    c = c[2:, :-1]
    idx = idx[2:, :-1]
    
    a_old = a_old[2:, :-1]
    b_old = b_old[2:, :-1]
    c_old = c_old[2:, :-1]
    idx_old = idx_old[2:, :-1]
    
    # Handle NaN values
    a = np.where((a == 999) | (a == 0), np.nan, a)
    b = np.where((b == 999) | (b == 0) | (b == -1), np.nan, b)
    
    # Calculate min/max values
    min_a = np.nanmin(a)
    max_a = np.nanmax(a)
    delt_color_a = 0.005 * (max_a - min_a)
    
    # Reshape data
    A = a.flatten()
    B = b.flatten()
    C = c.flatten()
    labels = idx.flatten()
    X = np.column_stack((A, B))
    
    # Calculate indices
    max_idx = int(np.max(idx))
    min_idx = int(np.min(idx))
    if max_idx > 50:
        max_idx = 50
    
    # Create colormaps
    N_colormap = 168
    cm = create_custom_colormap(N_colormap, max_idx - min_idx + 1)
    cm1 = create_custom_colormap(N_colormap, 256)
    
    # Create first figure
    plt.figure(figsize=(8, 8))
    im = plt.imshow(idx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cm, aspect='auto')
    plt.gca().set_ylabel('$y(0)$', fontsize=22)
    plt.gca().set_xlabel('$x(0)$', fontsize=22)
    plt.colorbar(im)
    im.set_clim(min_idx, max_idx + 0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'basins_1.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(idx_old, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cm, aspect='auto')
    plt.gca().set_ylabel('$y(0)$', fontsize=22)
    plt.gca().set_xlabel('$x(0)$', fontsize=22)
    plt.colorbar(im)
    im.set_clim(min_idx, max_idx + 0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'basins_old.png'), dpi=300)
    plt.close()

    # Визуализация разницы между idx и idx_old с подписями
    diff = idx - idx_old
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)), extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
    for (i, j), val in np.ndenumerate(diff):
        color = 'black' if abs(val) < np.max(np.abs(diff))/2 else 'white'
        ax.text(j, i, int(val), ha='center', va='center', color=color, fontsize=8)
    ax.set_ylabel('$y(0)$', fontsize=22)
    ax.set_xlabel('$x(0)$', fontsize=22)
    plt.title('Различия между idx и idx_old')
    unique_vals = np.unique(diff)
    legend_patches = [mpatches.Patch(color='none', label=f'{v}: {"совпадает" if v==0 else "различие"}') for v in unique_vals]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'basins_diff_values.png'), dpi=300)
    plt.close()

    # Create second figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    
    # Plot 1: Basins
    im = axes[0, 2].imshow(idx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cm)
    axes[0, 2].set_title('$\\geq$1 - stable, $\\leq$-1 - fixed, 0 - unbound', fontsize=FONTsize)
    axes[0, 2].set_xlabel('$x(0)$', fontsize=FONTsize)
    axes[0, 2].set_ylabel('$y(0)$', fontsize=FONTsize)
    plt.colorbar(im, ax=axes[0, 2])
    im.set_clim(min_idx, max_idx + 0.99)
    
    # Plot 2: Mean Peak
    im = axes[0, 0].imshow(a, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cm1)
    axes[0, 0].set_title('mean Peak', fontsize=FONTsize)
    axes[0, 0].set_xlabel('$x$', fontsize=FONTsize)
    axes[0, 0].set_ylabel('$y$', fontsize=FONTsize)
    plt.colorbar(im, ax=axes[0, 0])
    im.set_clim(min_a - delt_color_a, max_a)
    
    # Plot 3: Mean Interval
    im = axes[0, 1].imshow(b, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cm1)
    axes[0, 1].set_title('mean Interval', fontsize=FONTsize)
    axes[0, 1].set_xlabel('$x$', fontsize=FONTsize)
    axes[0, 1].set_ylabel('$y$', fontsize=FONTsize)
    plt.colorbar(im, ax=axes[0, 1])
    im.set_clim(min_a - delt_color_a, max_a)
    
    # Plot 4: Scatter plot
    scatter = axes[1, 0].scatter(X[:, 0], X[:, 1], c=labels, cmap=cm, s=30)
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel('mean Peak', fontsize=FONTsize)
    axes[1, 0].set_ylabel('mean Interval', fontsize=FONTsize)
    minX, maxX = np.min(X[:, 0]), np.max(X[:, 0])
    minY, maxY = np.min(X[:, 1]), np.max(X[:, 1])
    deltX = 0.1 * (maxX - minX)
    deltY = 0.1 * (maxY - minY)
    axes[1, 0].set_xlim([minX - deltX, maxX + deltX])
    axes[1, 0].set_ylim([minY - deltY, maxY + deltY])
    
    # Plot 5: 2D Histogram
    hist = axes[1, 1].hist2d(X[:, 0], X[:, 1], bins=25, cmap=cm1)
    axes[1, 1].set_xlabel('mean Peak', fontsize=FONTsize)
    axes[1, 1].set_ylabel('mean Interval', fontsize=FONTsize)
    plt.colorbar(hist[3], ax=axes[1, 1])
    axes[1, 1].set_xlim([minX - deltX, maxX + deltX])
    axes[1, 1].set_ylim([minY - deltY, maxY + deltY])
    
    # Plot 6: Helpful Array
    im = axes[1, 2].imshow(c, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='turbo')
    axes[1, 2].set_title('1 - stable, -1 - fixed, 0 - unbound', fontsize=FONTsize)
    axes[1, 2].set_xlabel('$x$', fontsize=FONTsize)
    axes[1, 2].set_ylabel('$y$', fontsize=FONTsize)
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_ticks([-0.6667, 0, 0.6667])
    cbar.set_ticklabels([-1, 0, 1])
    im.set_clim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'basins_2.png'), dpi=300)
    plt.close()
    
    print("Processing completed.")

if __name__ == "__main__":
    main() 