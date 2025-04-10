#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from matplotlib.colors import LogNorm


def load_data(filepath, replace_negative=True, pad_value=0):
    data = []
    max_cols = 0
    
    with open(filepath, 'r') as f:
        header1 = f.readline().strip() 
        header2 = f.readline().strip()
        for line in f:
            row = [int(val.strip()) for val in line.strip().split(',') if val.strip()]
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
                 figsize=(14, 12), cmap='hot', log_scale=False, vmin=None, vmax=None, show_plot=True):
    plt.figure(figsize=figsize)
    
    if vmin is None:
        vmin = np.min(data[data > 0]) if log_scale and np.any(data > 0) else np.min(data)
    if vmax is None:
        vmax = np.max(data)
    
    if vmax <= vmin: # Handle cases with uniform data or single value
         vmin = vmax -1 if vmax > 0 else 0
         vmax = vmax + 1 if vmax >=0 else 1


    if log_scale:
        plot_data = np.copy(data)
        effective_vmin = vmin if vmin > 0 else 1e-9 # Ensure vmin is positive for LogNorm
        plot_data[plot_data <= 0] = effective_vmin 
        norm = LogNorm(vmin=effective_vmin, vmax=max(vmax, effective_vmin + 1e-9)) # Ensure vmax > vmin
        im = plt.imshow(plot_data, cmap=cmap, aspect='auto', norm=norm)
    else:
        im = plt.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Value Intensity')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Parameter Value Index', fontsize=12)
    plt.ylabel('Parameter Value Index', fontsize=12) # Changed from Initial Condition Index
    
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_data(data):
    print(f"Data shape: {data.shape}")
    if data.size > 0:
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data):.2f}")
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
    parser = argparse.ArgumentParser(description='Generate a heatmap from bifurcation diagram data.')
    
    parser.add_argument('--input', type=str, default='Bifurcation_medium.csv',
                       help='Input CSV filename (expected in workspace/bifurcation/)')
    parser.add_argument('--output', type=str, default='bifurcation_diagram_heatmap.png',
                       help='Output image filename (will be saved in Scripts directory)')
    parser.add_argument('--cmap', type=str, default='hot',
                       help='Matplotlib colormap to use')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic color scaling')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the plot window")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..')) 
    
    input_dir = os.path.join(workspace_root, 'Bifurcation') # Changed input directory
    output_dir = script_dir # Output directory remains the script's directory

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
        data = load_data(input_filepath, replace_negative=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\nData Analysis:")
    analyze_data(data)
    
    print("\nGenerating heatmap...")
    try:
        plot_heatmap(
            data, 
            output_filepath,
            title=f"Bifurcation Diagram Heatmap - {args.input}", 
            cmap=args.cmap,
            log_scale=args.log,
            vmin=0, 
            show_plot=not args.no_show
        )
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        return

    if data.size > 0 and not args.log and np.max(data) > 0 and np.min(data[data > 0]) > 0 and (np.max(data) / max(1e-9, np.min(data[data > 0]))) > 50:
        log_output_base = os.path.splitext(args.output)[0]
        log_output_filepath = os.path.join(output_dir, f"{log_output_base}_log.png")
        print("\nGenerating log-scale heatmap for better visualization of value differences...")
        try:
            plot_heatmap(
                data, 
                log_output_filepath,
                title=f"Bifurcation Diagram Heatmap (Log Scale) - {args.input}", 
                cmap='inferno', 
                log_scale=True,
                show_plot=not args.no_show
            )
        except Exception as e:
             print(f"Error plotting log-scale heatmap: {e}")

    
    print("\nDone!")


if __name__ == "__main__":
    main()
