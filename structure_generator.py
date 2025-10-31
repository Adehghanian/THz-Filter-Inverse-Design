"""
structure_generator.py
----------------------
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


    Args:
        rows (int): Total number of rows (vertical pixels) in the structure grid.

    Returns:
        list[np.ndarray]: A list of 1D NumPy arrays, each representing one
                          valid mirrored column configuration.
    """

    possible_columns = list(product([0, 1], repeat=(rows - 1) // 2))

    valid_half = [
        np.array(col)
        for col in possible_columns
        if np.sum(np.array(col) ^ np.append(np.array(col)[1:], 1)) < 2
    ]

    # Convert the list into a matrix where each column is one pattern
    valid_half = np.array(valid_half).T

    # -----------------------------------------------------------------------
    # Mirror the valid half to enforce vertical symmetry
    # -----------------------------------------------------------------------
    # Add a center row of 1s (continuous metallic center line--ensure connectivity)
    middle_row = np.ones((1, valid_half.shape[1]))

    # Create a mirrored (bottom) version of the top half
    mirrored_half = np.flipud(valid_half)

    # Stack top half + middle row + mirrored bottom
    combined = np.vstack((valid_half, middle_row, mirrored_half))

    # -----------------------------------------------------------------------
    # Convert each column of the combined matrix into a 1D array
    # -----------------------------------------------------------------------
    valid_columns = [combined[:, i] for i in range(combined.shape[1])]

    return valid_columns


# ---------------------------------------------------------------------------
# Standalone test mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run a quick check when executed directly
    cols = generate_valid_columns()
    print(f"Generated {len(cols)} valid columns of length {len(cols[0])}.")