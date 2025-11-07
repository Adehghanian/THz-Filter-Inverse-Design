"""
thz_filter_model.py
---------------------------------------------------------------------------
Contains the analytical model (Conformal Mapping + ABCD Matrix) used to 
calculate the S-parameters of the THz filter structure, and functions 
to define the ideal target filter response.
"""

import numpy as np
from scipy.special import ellipk
import importlib

# Explicitly import only what is necessary from config (but keep redundant imports for consistency/safety)
from config import (
    E_EFF, C, COLS, CELL_LENGTH, RECT_WIDTH, W_CPS,
    FILTER_GROUP_DELAY
)


# ---------------------------------------------------------------------------
# GEOMETRY → ELECTRICAL PARAMETER CONVERSION
# ---------------------------------------------------------------------------
def calculate_S_W_values(grid):
    """
    Calculates the conductor width (W) and slot width (S) for each column
    of the structure grid based on the binary pixel configuration.

    Args:
        grid (np.ndarray): The 2D binary structure grid (ROWS x COLS).

    Returns:
        tuple: (S_values [m], W_values [m])
    """

    # Total conductor width per column = (number of metal pixels * rect_width) + base CPS width
    W_values = np.sum(grid, axis=0) * RECT_WIDTH + W_CPS

    # Slot width between conductors = total spacing (100 µm) - conductor width
    # Note: Total spacing is assumed to be 100e-6 m based on the model geometry.
    S_values = 100e-6 - W_values
    return S_values, W_values


def calculate_Z1(S_values, W_values):
    """
    Calculates the Characteristic Impedance (Z1) for each column based on 
    slot (S) and conductor (W) widths using a quasi-static model involving
    elliptic integrals (Conformal Mapping).

    Args:
        S_values (np.ndarray): Array of slot widths in meters.
        W_values (np.ndarray): Array of conductor widths in meters.

    Returns:
        np.ndarray: Array of characteristic impedances Z1 (Ohms).
    """

    h = 1e-6  # Substrate height (1 µm) — typical for thin-film membrane

    # Elliptic modulus terms for conformal mapping (K and K')
    K_values = S_values / (S_values + 2 * W_values)
    KP_values = np.sqrt(1 - K_values**2)

    # Adjusted terms considering finite substrate thickness (K1 and K1')
    # NOTE: K1 and K1p are calculated but not used in the final Z1 equation,
    # suggesting this implementation relies on the infinite substrate approximation
    # provided by K/KP, which is often sufficient for membrane-supported structures.
    K1_values = np.sinh(np.pi * S_values / (4*h)) / np.sinh(np.pi * (S_values + 2*W_values) / (4*h))
    K1p_values = np.sqrt(1 - K1_values**2)

    # Characteristic impedance equation for Coplanar Stripline (CPS)
    Z1_values = 120 * np.pi * ellipk(K_values) / (np.sqrt(E_EFF) * ellipk(KP_values))
    return Z1_values


# ---------------------------------------------------------------------------
# ABCD MATRIX CALCULATIONS
# ---------------------------------------------------------------------------
def calculate_matrix(beta, L, Z1):
    """
    Calculates the ABCD matrix for a single transmission line section.

    Args:
        beta (float): Propagation constant (rad/m).
        L (float): Length of the section (CELL_LENGTH).
        Z1 (float): Characteristic impedance of the section.

    Returns:
        np.ndarray: 2x2 complex ABCD matrix.
    """

    # Transmission line equations
    A = np.cos(beta * L)
    B = 1j * Z1 * np.sin(beta * L)
    C = 1j * np.sin(beta * L) / Z1 if Z1 != 0 else 0j
    D = np.cos(beta * L)
    return np.array([[A, B], [C, D]])


def calculate_scattering_parameters(TL, Z1_reference):
    """
    Converts a cascaded ABCD matrix (TL) to S-parameters (S11, S21).

    Args:
        TL (np.ndarray): The final cascaded 2x2 ABCD matrix.
        Z1_reference (float): Reference impedance (typically Z0 or the impedance 
                              of the first section).

    Returns:
        tuple: (S11_complex, S21_complex)
    """

    # Standard ABCD -> S conversion equations (assuming Z_source = Z_load = Z1_reference)
    denom = TL[0,0] + TL[0,1]/Z1_reference + TL[1,0]*Z1_reference + TL[1,1]
    S11 = (TL[0,0] + TL[0,1]/Z1_reference - TL[1,0]*Z1_reference - TL[1,1]) / denom
    S21 = 2 / denom
    return S11, S21


def calculate_S21_dB(Z1_values, freqs):
    """
    Calculates the full S-parameter response (S21, S11 magnitude and phase) 
    by cascading the ABCD matrix for all structural columns across the 
    defined frequency sweep.

    Args:
        Z1_values (np.ndarray): Array of characteristic impedances for each column.
        freqs (np.ndarray): Array of frequencies (Hz) for the sweep.

    Returns:
        tuple: (S21_dB, S11_dB, S21_phase, S11_phase, S21_complex)
    """

    
    # Initialize storage arrays
    S21_dB, S11_dB, S21_phase, S11_phase, S21_complex = [], [], [], [], []
    
    # Sweep over all frequencies
    for F in freqs:
        # Propagation constant β = 2πf√ε_eff / c
        beta = (2 * np.pi * F * np.sqrt(E_EFF)) / C
        TL = np.eye(2, dtype=complex)  # Identity matrix for initial ABCD

        # Cascade each section (multiply ABCD matrices)
        for i in range(COLS):
            TL = np.dot(TL, calculate_matrix(beta, CELL_LENGTH, Z1_values[i]))

        # Convert final ABCD matrix to S-parameters
        # Reference impedance is set to the impedance of the first section
        Z_ref = Z1_values[0] if Z1_values[0] != 0 else 1 
        S11, S21 = calculate_scattering_parameters(TL, Z_ref)

        # Store magnitude, phase, and complex values
        S21_dB.append(20 * np.log10(abs(S21)))
        S11_dB.append(20 * np.log10(abs(S11)))
        S21_phase.append(np.angle(S21, deg=True))
        S11_phase.append(np.angle(S11, deg=True))
        S21_complex.append(S21)

    # Return all quantities as arrays
    return (
        np.array(S21_dB),
        np.array(S11_dB),
        np.array(S21_phase),
        np.array(S11_phase),
        np.array(S21_complex)
    )


# ---------------------------------------------------------------------------
# IDEAL FILTER TARGET DEFINITION
# ---------------------------------------------------------------------------
def ideal_filter(
    filter_type=None,
    center_frequency=None,
    bandwidth=None,
    transition_bw=None,
    depth_dB=None,
    freqs=None,
):
    """
    Generates the ideal magnitude and phase response for common filter types 
    (bandstop, bandpass, lowpass, highpass) using simple linear slopes.

    Args:
        filter_type (str, optional): 'bandstop', 'bandpass', 'lowpass', or 'highpass'.
        center_frequency (float, optional): Center or cutoff frequency (Hz).
        bandwidth (float, optional): Filter bandwidth (Hz).
        transition_bw (float, optional): Width of the transition region (Hz).
        depth_dB (float, optional): Stopband attenuation level (negative dB).
        freqs (np.ndarray, optional): Frequency sweep array (Hz).

    Returns:
        tuple: (ideal_S21_dB, ideal_S11_dB, ideal_S21_phase_deg, ideal_S11_phase_deg)
    """

    # --- SENSITIVE: Reload config and fallback to global values if arguments are None ---
    # This block is essential for the GUI's dynamic configuration update.
    import config
    importlib.reload(config)

    if filter_type is None:
        filter_type = config.FILTER_TYPE
    if center_frequency is None:
        center_frequency = config.FILTER_CENTER_FREQ
    if bandwidth is None:
        bandwidth = config.FILTER_BANDWIDTH
    if transition_bw is None:
        transition_bw = config.FILTER_TRANSITION_BW
    if depth_dB is None:
        depth_dB = config.FILTER_DEPTH_DB
    if freqs is None:
        freqs = config.FREQS

    # --- Response Generation ---

    # Initialize arrays for frequency sweep
    ideal_S21 = np.zeros_like(freqs)
    ideal_S21_phase = np.zeros_like(freqs)
    ideal_S11_phase = np.zeros_like(freqs)

    # Define key frequency markers for Bandpass/Bandstop
    f_low = center_frequency - bandwidth / 2
    f_high = center_frequency + bandwidth / 2
    slope_low_start = f_low - transition_bw
    slope_high_end = f_high + transition_bw

    # ---------------- Magnitude Response ----------------
    for i, f in enumerate(freqs):

        # ---------- BANDSTOP (Notch Filter) ----------
        if filter_type.lower() == "bandstop":
            if f_low <= f <= f_high:
                ideal_S21[i] = depth_dB
            elif slope_low_start <= f < f_low:
                ideal_S21[i] = depth_dB * (f - slope_low_start) / transition_bw
            elif f_high < f <= slope_high_end:
                ideal_S21[i] = depth_dB * (slope_high_end - f) / transition_bw
            else:
                ideal_S21[i] = 0  # Passband (no attenuation)

        # ---------- BANDPASS ----------
        elif filter_type.lower() == "bandpass":
            if f_low <= f <= f_high:
                ideal_S21[i] = 0  # Passband (0 dB)
            elif slope_low_start <= f < f_low:
                ideal_S21[i] = depth_dB * (1 - (f - slope_low_start) / transition_bw)
            elif f_high < f <= slope_high_end:
                ideal_S21[i] = depth_dB * (1 - (slope_high_end - f) / transition_bw)
            else:
                ideal_S21[i] = depth_dB  # Stopband attenuation

        # ---------- LOWPASS ----------
        elif filter_type.lower() == "lowpass":
            f_cutoff = center_frequency
            if f <= f_cutoff:
                ideal_S21[i] = 0  # Passband
            elif f_cutoff < f <= f_cutoff + transition_bw:
                ideal_S21[i] = depth_dB * (f - f_cutoff) / transition_bw
            else:
                ideal_S21[i] = depth_dB  # Stopband

        # ---------- HIGHPASS ----------
        elif filter_type.lower() == "highpass":
            f_cutoff = center_frequency
            if f >= f_cutoff:
                ideal_S21[i] = 0  # Passband
            elif f_cutoff - transition_bw <= f < f_cutoff:
                ideal_S21[i] = depth_dB * (1 - (f - (f_cutoff - transition_bw)) / transition_bw)
            else:
                ideal_S21[i] = depth_dB  # Stopband

        else:
            raise ValueError(f"Unknown filter_type '{filter_type}'. Valid options: bandstop, bandpass, lowpass, highpass.")

    # ---------------- Phase Response ----------------
    # Linear phase (constant group delay) is typically desired
    ideal_S21_phase = -2 * np.pi * freqs * FILTER_GROUP_DELAY
    ideal_S11_phase = ideal_S21_phase + np.pi  # Reflection phase shifted by π

    # Convert from radians to degrees and normalize to (-180, 180)
    ideal_S21_phase = np.rad2deg(ideal_S21_phase)
    ideal_S11_phase = np.rad2deg(ideal_S11_phase)
    ideal_S21_phase = (ideal_S21_phase + 180) % 360 - 180
    ideal_S11_phase = (ideal_S11_phase + 180) % 360 - 180

    # ---------------- Reflection Magnitude ----------------
    # Approximate reflection mirror: deep reflection where transmission dips
    ideal_S11 = -ideal_S21 + depth_dB

    return ideal_S21, ideal_S11, ideal_S21_phase, ideal_S11_phase
