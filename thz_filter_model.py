
import numpy as np
from scipy.special import ellipk
from config import E_EFF, C, COLS, CELL_LENGTH, RECT_WIDTH, W_CPS
from config import (
    FREQS, FILTER_TYPE, FILTER_CENTER_FREQ,
    FILTER_BANDWIDTH, FILTER_TRANSITION_BW,
    FILTER_DEPTH_DB, FILTER_GROUP_DELAY
)

# ---------------------------------------------------------------------------
# Geometry → Electrical Parameter Conversion
# ---------------------------------------------------------------------------
def calculate_S_W_values(grid):

    # Total conductor width per column = number of metal pixels × rect_width + base CPS width
    W_values = np.sum(grid, axis=0) * RECT_WIDTH + W_CPS

    # Slot width between conductors = total spacing (100 µm) - conductor width
    S_values = 100e-6 - W_values
    return S_values, W_values


def calculate_Z1(S_values, W_values):

    h = 1e-6  # Substrate height (1 µm) — typical for thin-film membrane
    # Elliptic modulus terms for conformal mapping
    K_values = S_values / (S_values + 2 * W_values)
    KP_values = np.sqrt(1 - K_values**2)

    # Adjusted terms considering finite substrate thickness
    K1_values = np.sinh(np.pi * S_values / (4*h)) / np.sinh(np.pi * (S_values + 2*W_values) / (4*h))
    K1p_values = np.sqrt(1 - K1_values**2)

    # Characteristic impedance equation for CPS
    Z1_values = 120 * np.pi * ellipk(K_values) / (np.sqrt(E_EFF) * ellipk(KP_values))
    return Z1_values


# ---------------------------------------------------------------------------
# ABCD Matrix Calculations (per-section and cascaded)
# ---------------------------------------------------------------------------
def calculate_matrix(beta, L, Z1):

    # Transmission line equations
    A = np.cos(beta * L)
    B = 1j * Z1 * np.sin(beta * L)
    C = 1j * np.sin(beta * L) / Z1 if Z1 != 0 else 0j
    D = np.cos(beta * L)
    return np.array([[A, B], [C, D]])


def calculate_scattering_parameters(TL, Z1):

    # Standard ABCD → S conversion equations
    denom = TL[0,0] + TL[0,1]/Z1 + TL[1,0]*Z1 + TL[1,1]
    S11 = (TL[0,0] + TL[0,1]/Z1 - TL[1,0]*Z1 - TL[1,1]) / denom
    S21 = 2 / denom
    return S11, S21


def calculate_S21_dB(Z1_values, freqs=FREQS):

    
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
        S11, S21 = calculate_scattering_parameters(TL, Z1_values[0] if Z1_values[0] != 0 else 1)

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


def ideal_filter(
    filter_type=None,
    center_frequency=None,
    bandwidth=None,
    transition_bw=None,
    depth_dB=None,
    freqs=None,
):

    import importlib, config
    importlib.reload(config)

    if filter_type is None:
        filter_type = FILTER_TYPE
    if center_frequency is None:
        center_frequency = FILTER_CENTER_FREQ
    if bandwidth is None:
        bandwidth = FILTER_BANDWIDTH
    if transition_bw is None:
        transition_bw = FILTER_TRANSITION_BW
    if depth_dB is None:
        depth_dB = FILTER_DEPTH_DB
    if freqs is None:
        freqs = config.FREQS



    # Initialize arrays for frequency sweep
    ideal_S21 = np.zeros_like(freqs)
    ideal_S21_phase = np.zeros_like(freqs)
    ideal_S11_phase = np.zeros_like(freqs)

    # Define key frequency markers
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
    # Linear phase → constant group delay
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
