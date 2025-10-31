"""
config.py
----------
Centralized configuration file for the THz inverse design project.

Contains:
    • Geometric and physical constants
    • Frequency sweep parameters
    • Simulation settings (HFSS version, headless mode)
    • Ideal filter target specifications (type, bandwidth, etc.)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Geometry Parameters (in SI units)
# ---------------------------------------------------------------------------
ROWS = 18               # Number of vertical pixels in the structure
COLS = 200              # Number of horizontal cells (length-wise)
W_CPS = 10e-6           # Base CPS conductor width [m]
RECT_WIDTH = 5e-6       # Width per binary "pixel" [m]
CELL_LENGTH = 10e-6     # Length of each structural cell [m]

# ---------------------------------------------------------------------------
# Material / Physical Constants
# ---------------------------------------------------------------------------
E_EFF = 1.45            # Effective dielectric constant (membrane-supported)
C = 3e8                 # Speed of light [m/s]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# HFSS / PyAEDT Configuration
# ---------------------------------------------------------------------------
HFSS_VERSION = "2024.1"
HFSS_NON_GRAPHICAL = False

HFSS_PROJECT_NAME = "Inverse_design_ABCD.aedt"
HFSS_SAVE_PATH = r"C:\Ali\HFSS\Inverse_design\pygad_ABCD.aedt"
HFSS_TEMP_SUFFIX = ".ansys"       # Suffix for temporary AEDT sessions

# ---------------------------------------------------------------------------
# HFSS Setup Parameters
# ---------------------------------------------------------------------------
HFSS_SETUP_NAME = "Setup1"
HFSS_SETUP_FREQ = 1.0

# Adaptive setup refinement parameters
HFSS_MAX_PASSES = 10
HFSS_MAX_DELTA_S = 0.6

# ---------------------------------------------------------------------------
# HFSS Sweep Parameters
# ---------------------------------------------------------------------------
HFSS_LOW_FREQ = 0.4
HFSS_HIGH_FREQ = 1.2
HFSS_STEP_SIZE = 0.05

HFSS_SWEEP_TYPE = "Discrete"
HFSS_SAVE_FIELDS = False
HFSS_SAVE_RAD_FIELDS = False

# ---------------------------------------------------------------------------
# HFSS Export / Reuse Configuration
# ---------------------------------------------------------------------------
HFSS_EXPORT_DIR = r"C:\Ali\HFSS\Inverse_design\csvfiles"  # Where to save S-parameter CSV
HFSS_EXPORT_SPARAMS = True        # Export S-parameters to CSV
HFSS_REUSE_PROJECT = True         # Reuse existing AEDT project if available
HFSS_AUTO_CLOSE = True            # Close HFSS after simulation

# ---------------------------------------------------------------------------
# Derived Frequency Array (for analytical modeling)
# ---------------------------------------------------------------------------

FREQS = np.arange(HFSS_LOW_FREQ * 1e12,
                  HFSS_HIGH_FREQ * 1e12 + HFSS_STEP_SIZE * 1e12,
                  HFSS_STEP_SIZE * 1e12)



# Export / Result options
HFSS_EXPORT_DIR = r"C:\Ali\HFSS\Inverse_design\csvfiles"
HFSS_EXPORT_SPARAMS = True        # Whether to export S-parameters to CSV
HFSS_REUSE_PROJECT = True         # Reuse project instead of creating new each run


# ---------------------------------------------------------------------------
# Ideal Filter Target Configuration
# ---------------------------------------------------------------------------
FILTER_TYPE = "bandstop"
FILTER_CENTER_FREQ = 0.8e12
FILTER_BANDWIDTH = 0.14e12
FILTER_TRANSITION_BW = 0.03e12
FILTER_DEPTH_DB = -40.0
FILTER_GROUP_DELAY = 4.0e-12

# ==========================================================
# OPTIMIZATION DEFAULT PARAMETERS
# ==========================================================

# ---- Genetic Algorithm ----
GA_GENERATIONS = 20
GA_POPULATION = 40
GA_PARENTS = 10
GA_ELITISM = 10
GA_MUTATION_PROB = 0.1

# ---- Particle Swarm Optimization ----
PSO_PARTICLES = 30
PSO_ITERATIONS = 50
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

# ---- Adjoint Method ----
ADJ_LEARNING_RATE = 0.05
ADJ_ITERATIONS = 100
