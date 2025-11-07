"""
structure_generator.py
---------------------------------------------------------------------------
Generates valid structural column patterns for THz inverse design filters.

Each column is represented as a binary array (0 = void / air region, 1 = metal region).

These valid columns are later combined horizontally by the genetic algorithm
to form the full 2D structure of the THz filter.
"""

import numpy as np
from itertools import product
from config import ROWS


def generate_valid_columns(rows: int = ROWS):
    """
    Generate all valid column configurations with vertical mirror symmetry.

    Validity is determined by the constraint that there must be fewer than two
    switches between '1' (metal) and '0' (void) in the vertical direction
    of the half-column, ensuring stable geometric features.

    Args:
        rows (int): Total number of rows (vertical pixels) in the structure grid.

    Returns:
        list[np.ndarray]: A list of 1D NumPy arrays, each representing one
                          valid mirrored column configuration.
    """

    # Calculate possible configurations for the top half of the column
    half_rows = (rows - 1) // 2
    possible_columns = list(product([0, 1], repeat=half_rows))

    # Filter possible halves based on the stability/connectivity rule
    valid_half = [
        np.array(col)
        for col in possible_columns
        # The stability rule ensures smooth transitions (fewest possible 0->1 or 1->0 changes).
        # XOR with shifted array checks for transitions (1-0 or 0-1 switch).
        # We append 1 to the end to force a check on the last element transition.
        if np.sum(np.array(col) ^ np.append(np.array(col)[1:], 1)) < 2
    ]

    # Convert the list of valid half-patterns into a matrix
    valid_half = np.array(valid_half).T

    # ---------------------------------------------------------------------------
    # ENFORCE VERTICAL SYMMETRY AND CONNECTIVITY
    # ---------------------------------------------------------------------------
    # Add a center row of 1s (ensures continuous metallic center line for connectivity)
    middle_row = np.ones((1, valid_half.shape[1]))

    # Create a mirrored (bottom) version of the top half
    mirrored_half = np.flipud(valid_half)

    # Stack top half + middle row + mirrored bottom to form the full column
    combined = np.vstack((valid_half, middle_row, mirrored_half))

    # ---------------------------------------------------------------------------
    # FINAL COLUMN OUTPUT
    # ---------------------------------------------------------------------------
    # Convert each column of the combined matrix into its own 1D array
    valid_columns = [combined[:, i] for i in range(combined.shape[1])]

    return valid_columns


# ---------------------------------------------------------------------------
# STANDALONE TEST MODE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run a quick check when executed directly
    cols = generate_valid_columns()
    print(f"Generated {len(cols)} valid columns of length {len(cols[0])}.")
