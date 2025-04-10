#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LogNorm


def load_data(filepath, replace_negative=True, pad_value=0):
    ranges = []
    try:
        with open(filepath, 'r') as f:
            for i in range(2):
                line = f.readline().strip()
                if not line: 
                    raise ValueError("Input file has fewer than 2 header lines.")
                ranges.append([float(x) for x in line.split()])
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading header lines from {filepath}: {e}")

    data = []
    max_cols = 0
    
    try:
        with open(filepath, 'r') as f:
            for _ in range(2):
                f.readline()
            
            for line in f:
                row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
                if row:
                    data.append(row)
                    max_cols = max(max_cols, len(row))
    except Exception as e:
         raise ValueError(f"Error reading data lines from {filepath}: {e}")

    if not data:
         print(f"Warning: No data found in {filepath} after header lines.")
         return np.empty((0, 0), dtype=np.float32), ranges # Return empty array if no data

    padded_data = np.full((len(data), max_cols), pad_value, dtype=np.float32)
    
    for i, row in enumerate(data):
        padded_data[i, :len(row)] = row
    
    if replace_negative:
        padded_data[padded_data < 0] = pad_value
    
    return padded_data, ranges


def plot_heatmap(data, output_filepath, ranges, title="LLE Heatmap", 
                 figsize=(14, 12), cmap='viridis', log_scale=False, vmin=None, vmax=None, show_plot=True):
    if data.size == 0:
        print("Cannot plot heatmap: Data array is empty.")
        return
        
    plt.figure(figsize=figsize)
    
    if vmin is None:
        positive_data = data[data > 0]
        vmin = np.min(positive_data) if positive_data.size > 0 and log_scale else np.min(data)
    if vmax is None:
        vmax = np.max(data)

    if vmax <= vmin: # Handle cases with uniform data or single value
         vmin = vmax -1 if vmax > 0 else 0
         vmax = vmax + 1 if vmax >=0 else 1
    
    extent = None
    if len(ranges) == 2 and len(ranges[0]) == 2 and len(ranges[1]) == 2:
         extent = [ranges[0][0], ranges[0][1], ranges[1][1], ranges[1][0]] # Flipped y-axis extent for correct orientation
    else:
         print("Warning: Invalid range format. Plotting without axis scales.")

    if log_scale:
        plot_data = np.copy(data)
        effective_vmin = vmin if vmin > 0 else 1e-9
        plot_data[plot_data <= 0] = effective_vmin
        norm = LogNorm(vmin=effective_vmin, vmax=max(vmax, effective_vmin + 1e-9)) 
        im = plt.imshow(plot_data, cmap=cmap, aspect='auto', norm=norm, extent=extent, origin='lower') # Set origin to lower
    else:
        im = plt.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, extent=extent, origin='lower') # Set origin to lower
    
    cbar = plt.colorbar(im)
    cbar.set_label('LLE Value')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Parameter 1', fontsize=12)
    plt.ylabel('Parameter 2', fontsize=12)
    
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_data(data, ranges):
    print(f"Data shape: {data.shape}")
    if len(ranges) == 2:
         print(f"Parameter 1 range: {ranges[0][0]} to {ranges[0][1]}")
         print(f"Parameter 2 range: {ranges[1][0]} to {ranges[1][1]}")
    else:
         print("Range information incomplete or invalid.")

    if data.size > 0:
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data):.6f}")
        print(f"Median value: {np.median(data)}")
        print(f"Unique values: {len(np.unique(data))}")
        
        neg_count = np.sum(data < 0)
        if neg_count > 0:
            print(f"Negative values: {neg_count} ({neg_count/data.size*100:.2f}%)")
        
        zero_count = np.sum(data == 0)
        print(f"Zero values: {zero_count} ({zero_count/data.size*100:.2f}%)")
    else:
         print("Data array is empty.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a heatmap from LLE data files.')
    
    parser.add_argument('--input', type=str, default='lle_hard_old.csv',
                       help='Input CSV filename (expected in workspace/lle/)')
    parser.add_argument('--output', type=str, default='lle_heatmap.png',
                       help='Output image filename (will be saved in Scripts directory)')
    parser.add_argument('--cmap', type=str, default='viridis',
                       help='Matplotlib colormap to use')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic color scaling')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the plot window")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '../workspace'))
    
    input_dir = os.path.join(workspace_root, 'lle') # Changed input directory
    output_dir = input_dir # Output directory remains the script's directory

    input_filepath = os.path.join(input_dir, args.input)
    output_filepath = os.path.join(output_dir, args.output)
    
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        # Try finding input relative to script dir as fallback or alternative?
        alt_input_filepath = os.path.join(script_dir, args.input)
        if os.path.exists(alt_input_filepath):
             print(f"Found input file relative to script dir: {alt_input_filepath}. Using this instead.")
             input_filepath = alt_input_filepath
        else:
             print(f"Also tried: {alt_input_filepath}")
             return
    
    print(f"Loading data from {input_filepath}...")
    try:
        data, ranges = load_data(input_filepath, replace_negative=True)
    except FileNotFoundError:
         print(f"Error: Input file not found at {input_filepath}")
         return
    except ValueError as e:
         print(f"Error loading data: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred during data loading: {e}")
         return

    print("\nData Analysis:")
    analyze_data(data, ranges)
    
    print("\nGenerating heatmap...")
    try:
        plot_heatmap(
            data, 
            output_filepath,
            ranges,
            title=f"LLE Heatmap - {args.input}", 
            cmap=args.cmap,
            log_scale=args.log,
            show_plot=not args.no_show
        )
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        return
    
    if data.size > 0 and not args.log and np.max(data) > 0: # Check if data is not empty and max > 0
        positive_data = data[data > 0]
        if positive_data.size > 0 and (np.max(data) / max(1e-9, np.min(positive_data))) > 50:
            log_output_base = os.path.splitext(args.output)[0]
            log_output_filepath = os.path.join(output_dir, f"{log_output_base}_log.png")
            print("\nGenerating log-scale heatmap for better visualization of value differences...")
            try:
                plot_heatmap(
                    data, 
                    log_output_filepath,
                    ranges,
                    title=f"LLE Heatmap (Log Scale) - {args.input}", 
                    cmap='plasma', 
                    log_scale=True,
                    show_plot=not args.no_show
                )
            except Exception as e:
                 print(f"Error plotting log-scale heatmap: {e}")

    
    print("\nDone!")


if __name__ == "__main__":
    main() 