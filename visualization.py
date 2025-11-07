"""
visualization.py
---------------------------------------------------------------------------
Provides utility functions for visualizing the structural grid and plotting
the S-parameter results from both the analytical model (ABCD) and the 
full-wave simulation (HFSS).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
# Use the Agg backend to prevent issues when running in headless environments
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

# Configuration and modeling function imports
import config
from config import COLS, CELL_LENGTH
from thz_filter_model import (
    calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
)

# ---------------------------------------------------------------------------
# STRUCTURE / GRID VISUALIZATION
# ---------------------------------------------------------------------------
def visualize_grid(grid, title="Optimized Structure Grid"):
    """
    Visualize the 2D binary grid structure representing the THz filter.

    Args:
        grid (np.ndarray): 2D array representing the structure (0=void, 1=metal).
        title (str): Title for the plot.
    """
    if grid is None:
        print("Cannot visualize: Grid is None.")
        return
        
    n_rows, n_cols = grid.shape
    # Set figure size proportional to the grid dimensions for clear visualization
    fig, ax = plt.subplots(figsize=(n_cols/4, n_rows/4), dpi=100) 
    
    # Create colormap: white for void (0), black for metal (1)
    cmap = mcolors.ListedColormap(['white', 'black'])
    
    # Map the grid to physical dimensions (µm)
    ax.imshow(
        grid, 
        cmap=cmap, 
        origin='upper', 
        extent=[0, n_cols * config.CELL_LENGTH * 1e6, n_rows * config.CELL_LENGTH * 1e6, 0]
    )
    
    ax.set_xlabel("Width (µm)")
    ax.set_ylabel("Height (µm)")
    ax.set_title(title, fontsize=10)
    
    # Generate ticks at cell boundaries
    x_ticks = np.arange(0, (n_cols + 1) * config.CELL_LENGTH * 1e6, config.CELL_LENGTH * 1e6)
    y_ticks = np.arange(0, (n_rows + 1) * config.CELL_LENGTH * 1e6, config.CELL_LENGTH * 1e6)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show() # Keep the standalone show for CLI or manual use.

# ---------------------------------------------------------------------------
# DATA EXPORT UTILITY
# ---------------------------------------------------------------------------
def save_s_parameters_to_csv(base_filename, calculated_data, hfss_data=None, save_dir=".", save_flag=True):
    """
    Save calculated and optional HFSS S-parameter data to CSV files using pandas.

    Args:
        base_filename (str): Base name for the CSV files.
        calculated_data (dict): Dictionary with calculated S-parameter data.
        hfss_data (dict, optional): Dictionary with HFSS S-parameter data.
        save_dir (str): Directory to save the CSV files.
        save_flag (bool): If True, saves files to disk. If False, only returns DataFrames.
        
    Returns:
        tuple: (calc_df, hfss_df) containing the pandas DataFrames.
    """
    calc_df = pd.DataFrame(calculated_data)
    hfss_df = pd.DataFrame(hfss_data) if hfss_data else None
    
    if save_flag:
        os.makedirs(save_dir, exist_ok=True)
        
        calc_path = os.path.join(save_dir, f"{base_filename}_calculated.csv")
        calc_df.to_csv(calc_path, index=False)
        print(f"Calculated S-parameters saved to {calc_path}")

        if hfss_data:
            hfss_path = os.path.join(save_dir, f"{base_filename}_hfss.csv")
            hfss_df.to_csv(hfss_path, index=False)
            print(f"HFSS S-parameters saved to {hfss_path}")
            
    return calc_df, hfss_df # Always return DataFrames

# ---------------------------------------------------------------------------
# S-PARAMETER PLOTTING
# ---------------------------------------------------------------------------
def plot_s_parameters(base_filename='S_parameters_advanced',
                      calculated_data_dict=None,
                      hfss_data_dict=None,
                      best_grid=None, # Unused, kept for original signature compatibility
                      load_dir=None): # Unused, kept for original signature compatibility
    """
    Plots the calculated (ABCD), ideal, and optional HFSS S-parameters from 
    data dictionaries into a multi-panel Matplotlib figure.
    
    Args:
        base_filename (str): Base name for the plot title.
        calculated_data_dict (dict): Dictionary with calculated S-parameter data.
        hfss_data_dict (dict, optional): Dictionary with HFSS S-parameter data.
        
    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """

    if calculated_data_dict is None:
        print("Error: No calculated data available for plotting.")
        return None

    calc_df = pd.DataFrame(calculated_data_dict)
    
    has_hfss = hfss_data_dict is not None
    hfss_df = pd.DataFrame(hfss_data_dict) if has_hfss else None

    # --- Create Figure (2 panels if no HFSS, 3 if HFSS data is present)
    fig, axs = plt.subplots(3 if has_hfss else 2, 1, figsize=(4, 6), dpi=80)
    fig.suptitle(f"S-Parameter Results ({base_filename})", fontsize=12, weight='bold')

    # --- Panel 1: Magnitude (Calculated vs Ideal)
    axs[0].plot(calc_df["Freq_THz"], calc_df["Calc_S21_dB"], label='Calculated $|S_{21}|$', color='blue')
    axs[0].plot(calc_df["Freq_THz"], calc_df["Ideal_S21_dB"], '--', color='blue', label='Ideal $|S_{21}|$', alpha=0.7)
    axs[0].plot(calc_df["Freq_THz"], calc_df["Calc_S11_dB"], label='Calculated $|S_{11}|$', color='red')
    axs[0].plot(calc_df["Freq_THz"], calc_df["Ideal_S11_dB"], '--', color='red', label='Ideal $|S_{11}|$', alpha=0.7)
    axs[0].set_title("Magnitude Response: Calculated (ABCD) vs. Ideal", fontsize=10)
    axs[0].set_ylabel("Magnitude (dB)", fontsize=9)
    axs[0].legend(fontsize=8, loc='best'); 
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # --- Panel 2: Phase (Calculated)
    axs[1].plot(calc_df["Freq_THz"], calc_df["Calc_S21_phase_deg"], label=r'$\angle S_{21}$', color='blue')
    axs[1].plot(calc_df["Freq_THz"], calc_df["Calc_S11_phase_deg"], label=r'$\angle S_{11}$', color='red')
    axs[1].set_title("Phase Response (Calculated - ABCD)", fontsize=10)
    axs[1].set_ylabel("Phase (°)", fontsize=9)
    axs[1].legend(fontsize=8, loc='best'); 
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # --- Panel 3 (Conditional): Magnitude (ABCD vs. HFSS)
    if has_hfss:
        axs[2].plot(calc_df["Freq_THz"], calc_df["Calc_S21_dB"], label='ABCD $|S_{21}|$', color='blue')
        axs[2].plot(hfss_df["HFSS_Freq_THz"], hfss_df["HFSS_S21_dB"], '--', color='blue', label='HFSS $|S_{21}|$', alpha=0.7)
        axs[2].plot(calc_df["Freq_THz"], calc_df["Calc_S11_dB"], label='ABCD $|S_{11}|$', color='red')
        axs[2].plot(hfss_df["HFSS_Freq_THz"], hfss_df["HFSS_S11_dB"], '--', color='red', label='HFSS $|S_{11}|$', alpha=0.7)
        axs[2].set_title("Magnitude Comparison: ABCD vs. HFSS", fontsize=10)
        axs[2].set_xlabel("Frequency (THz)", fontsize=9)
        axs[2].set_ylabel("Magnitude (dB)", fontsize=9)
        axs[2].legend(fontsize=8, loc='best'); 
        axs[2].grid(True, linestyle=':', alpha=0.6)
        
    else:
        # If no HFSS, the last plot is Phase, so set the X-label there
        axs[1].set_xlabel("Frequency (THz)", fontsize=9)

    plt.tight_layout()
    return fig
