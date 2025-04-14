#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LogNorm


def load_data(filepath, replace_negative=True, pad_value=0):
    data = []
    max_cols = 0
    
    with open(filepath, 'r') as f:
        # Skip first two lines
        next(f)
        next(f)
        
        for line in f:
            row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
            if row:
                data.append(row)
                max_cols = max(max_cols, len(row))
    
    padded_data = np.full((len(data), max_cols), pad_value, dtype=np.float32)
    
    for i, row in enumerate(data):
        padded_data[i, :len(row)] = row
    
    if replace_negative:
        padded_data[padded_data < 0] = pad_value
    
    return padded_data


def plot_heatmap(data, output_filepath, title="Bifurcation Diagram Heatmap", 
                 figsize=(14, 12), cmap='hot', vmin=None, vmax=None, show_plot=True):
    plt.figure(figsize=figsize)
    
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    
    if vmax <= vmin:
         vmin = vmax -1 if vmax > 0 else 0
         vmax = vmax + 1 if vmax >=0 else 1

    im = plt.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Value Intensity')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Parameter Value Index', fontsize=12)
    plt.ylabel('Parameter Value Index', fontsize=12)
    
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_data(data1, data2):
    print(f"Data 1 shape: {data1.shape}")
    print(f"Data 2 shape: {data2.shape}")
    
    if data1.size > 0 and data2.size > 0:
        print(f"Data 1 - Min: {np.min(data1)}, Max: {np.max(data1)}, Mean: {np.mean(data1):.2f}")
        print(f"Data 2 - Min: {np.min(data2)}, Max: {np.max(data2)}, Mean: {np.mean(data2):.2f}")
        
        diff = np.abs(data1 - data2)
        mean_error = np.mean(diff)
        max_error = np.max(diff)
        error_order = np.log10(max_error) if max_error > 0 else 0
        
        print(f"\nError Analysis:")
        print(f"Mean absolute error: {mean_error:.2e}")
        print(f"Maximum absolute error: {max_error:.2e}")
        print(f"Error order: 10^{error_order:.2f}")
        
        return diff
    else:
        print("One or both data arrays are empty.")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate and compare heatmaps from bifurcation diagram data.')
    
    parser.add_argument('--input1', type=str, default='bifurcation_test.csv',
                       help='First input CSV filename (expected in workspace/bifurcation/)')
    parser.add_argument('--input2', type=str, default='bifurcation_test_old.csv',
                       help='Second input CSV filename (expected in workspace/bifurcation/)')
    parser.add_argument('--output', type=str, default='bifurcation_comparison.png',
                       help='Output image filename (will be saved in Scripts directory)')
    parser.add_argument('--cmap', type=str, default='hot',
                       help='Matplotlib colormap to use')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the plot window")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..')) 
    
    input_dir = os.path.join(workspace_root, 'workspace/bifurcation')
    output_dir = os.path.join(workspace_root, 'results')

    input_filepath1 = os.path.join(input_dir, args.input1)
    input_filepath2 = os.path.join(input_dir, args.input2)
    output_filepath = os.path.join(output_dir, args.output)
    
    if not os.path.exists(input_filepath1) or not os.path.exists(input_filepath2):
        print(f"Error: Input files not found at {input_filepath1} or {input_filepath2}")
        return
    
    print(f"Loading data from {input_filepath1}...")
    try:
        data1 = load_data(input_filepath1, replace_negative=True)
    except Exception as e:
        print(f"Error loading data1: {e}")
        return

    print(f"Loading data from {input_filepath2}...")
    try:
        data2 = load_data(input_filepath2, replace_negative=True)
    except Exception as e:
        print(f"Error loading data2: {e}")
        return

    print("\nData Analysis:")
    diff = analyze_data(data1, data2)
    
    if diff is not None:
        print("\nGenerating heatmaps...")
        try:
            plt.figure(figsize=(20, 6))
            
            plt.subplot(131)
            plt.imshow(data1, cmap=args.cmap, aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.title(f"Data 1: {args.input1}")
            
            plt.subplot(132)
            plt.imshow(data2, cmap=args.cmap, aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.title(f"Data 2: {args.input2}")
            
            plt.subplot(133)
            plt.imshow(diff, cmap='RdBu', aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.title("Absolute Difference")
            
            plt.tight_layout()
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            print(f"Comparison heatmap saved to {output_filepath}")
            
            if not args.no_show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error plotting heatmaps: {e}")
            return
    
    print("\nDone!")


if __name__ == "__main__":
    main()
