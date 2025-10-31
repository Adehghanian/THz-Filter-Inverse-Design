"""
Visualization and data export utilities for the THz inverse-design project.
Includes:
- Grid visualization
- CSV export of calculated and HFSS S-parameters
- Comparison plots between analytical (ABCD) and HFSS results
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from config import COLS, CELL_LENGTH
from thz_filter_model import calculate_S_W_values
import config


# ---------------------------------------------------------------------------
# 1. Structure / Grid Visualization
# ---------------------------------------------------------------------------
def visualize_grid(grid, title="Structure Grid"):
    """
    Visualize the 2D grid structure.

    Args:
        grid (np.ndarray): 2D array representing the structure.
        title (str): Title for the plot.
    """
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(n_cols, n_rows), dpi=100)
    cmap = mcolors.ListedColormap(['white', 'black'])
    ax.imshow(grid, cmap=cmap, origin='upper', extent=[0, n_cols * CELL_LENGTH * 1e6, 0, n_rows * CELL_LENGTH * 1e6])
    ax.set_xlabel("Width (µm)")
    ax.set_ylabel("Height (µm)")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, (n_cols + 1) * CELL_LENGTH * 1e6, CELL_LENGTH * 1e6))
    ax.set_yticks(np.arange(0, (n_rows + 1) * CELL_LENGTH * 1e6, CELL_LENGTH * 1e6))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()



# Save Calculated and HFSS Data to CSV
def save_s_parameters_to_csv(base_filename, calculated_data, hfss_data=None, save_dir="."):
    """
    Save calculated and HFSS S-parameter data to CSV files.

    Args:
        base_filename (str): Base name for the CSV files.
        calculated_data (dict): Dictionary with calculated S-parameter data.
        hfss_data (dict, optional): Dictionary with HFSS S-parameter data.
        save_dir (str): Directory to save the CSV files.
    """
    calc_df = pd.DataFrame(calculated_data)
    calc_path = os.path.join(save_dir, f"{base_filename}_calculated.csv")
    calc_df.to_csv(calc_path, index=False)
    print(f"Calculated S-parameters saved to {calc_path}")

    if hfss_data:
        hfss_df = pd.DataFrame(hfss_data)
        hfss_path = os.path.join(save_dir, f"{base_filename}_hfss.csv")
        hfss_df.to_csv(hfss_path, index=False)
        print(f"HFSS S-parameters saved to {hfss_path}")



def plot_s_parameters_from_csv(base_filename='S_parameters_advanced',
                               load_dir=r"C:\Ali\HFSS\Inverse_design\csvfiles",
                               best_grid=None):

    from thz_filter_model import (
        calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
    )

    calc_path = os.path.join(load_dir, f"{base_filename}_calculated.csv")
    hfss_path = os.path.join(load_dir, f"{base_filename}_hfss.csv")

    has_hfss = os.path.exists(hfss_path)
    has_calc = os.path.exists(calc_path)

    # --- If best_grid provided, recompute and overwrite CSV regardless of file existence ---
    if best_grid is not None:
        print("Computing fresh S-parameters from best_grid (ignoring old CSV)...")

        import importlib, config
        importlib.reload(config)

        S, W = calculate_S_W_values(best_grid)
        Z1 = calculate_Z1(S, W)
        S21_dB_calc, S11_dB_calc, S21_phase_calc, S11_phase_calc, _ = calculate_S21_dB(Z1)

        freqs = config.FREQS
        ideal_S21, ideal_S11, ideal_S21_phase, ideal_S11_phase = ideal_filter(
            filter_type=config.FILTER_TYPE,
            center_frequency=config.FILTER_CENTER_FREQ,
            bandwidth=config.FILTER_BANDWIDTH,
            transition_bw=config.FILTER_TRANSITION_BW,
            depth_dB=config.FILTER_DEPTH_DB,
            freqs=freqs
        )

        calc_df = pd.DataFrame({
            "Freq_THz": freqs / 1e12,
            "Calc_S11_dB": S11_dB_calc,
            "Calc_S21_dB": S21_dB_calc,
            "Calc_S11_phase_deg": S11_phase_calc,
            "Calc_S21_phase_deg": S21_phase_calc,
            "Ideal_S11_dB": ideal_S11,
            "Ideal_S21_dB": ideal_S21
        })

        os.makedirs(load_dir, exist_ok=True)
        calc_path = os.path.join(load_dir, f"{base_filename}_calculated.csv")
        calc_df.to_csv(calc_path, index=False)
        print(f"Updated S-parameters saved to {calc_path}")

    elif has_calc:
        print("Loading existing calculated CSV...")
        calc_df = pd.read_csv(calc_path)

    else:
        print("Error: no calculated data available.")
        return None

    # --- Load HFSS results if available ---
    if has_hfss:
        hfss_df = pd.read_csv(hfss_path)
    else:
        hfss_df = None

    # (plotting code continues here...)


    # --- Create Figure
    fig, axs = plt.subplots(3 if has_hfss else 2, 1, figsize=(6, 8), dpi=60)

    # --- Magnitude (Calculated vs Ideal)
    axs[0].plot(calc_df["Freq_THz"], calc_df["Calc_S21_dB"], label='Calculated |$S_{21}$|', color='blue')
    axs[0].plot(calc_df["Freq_THz"], calc_df["Ideal_S21_dB"], '--', color='blue', label='Ideal |$S_{21}$|')
    axs[0].plot(calc_df["Freq_THz"], calc_df["Calc_S11_dB"], label='Calculated |$S_{11}$|', color='red')
    axs[0].plot(calc_df["Freq_THz"], calc_df["Ideal_S11_dB"], '--', color='red', label='Ideal |$S_{11}$|')
    axs[0].set_title("Magnitude Response: Calculated vs. Ideal")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].legend(); axs[0].grid(True)

    # --- Phase (Calculated)
    axs[1].plot(calc_df["Freq_THz"], calc_df["Calc_S21_phase_deg"], label=r'$\angle S_{21}$', color='blue')
    axs[1].plot(calc_df["Freq_THz"], calc_df["Calc_S11_phase_deg"], label=r'$\angle S_{11}$', color='red')
    axs[1].set_title("Phase Response (Calculated)")
    axs[1].set_ylabel("Phase (°)")
    axs[1].legend(); axs[1].grid(True)

    # --- Include HFSS results if available
    if has_hfss:
        axs[2].plot(calc_df["Freq_THz"], calc_df["Calc_S21_dB"], label='ABCD |$S_{21}$|', color='blue')
        axs[2].plot(hfss_df["HFSS_Freq_THz"], hfss_df["HFSS_S21_dB"], '--', color='blue', label='HFSS |$S_{21}$|')
        axs[2].plot(calc_df["Freq_THz"], calc_df["Calc_S11_dB"], label='ABCD |$S_{11}$|', color='red')
        axs[2].plot(hfss_df["HFSS_Freq_THz"], hfss_df["HFSS_S11_dB"], '--', color='red', label='HFSS |$S_{11}$|')
        axs[2].set_title("Magnitude: ABCD vs. HFSS")
        axs[2].set_xlabel("Frequency (THz)")
        axs[2].set_ylabel("Magnitude (dB)")
        axs[2].legend(); axs[2].grid(True)

    plt.tight_layout()
    return fig
