"""
fitness_functions.py
--------------------
Defines fitness evaluation and helper metrics for the GA optimization.
"""

import numpy as np
import multiprocessing as mp
from thz_filter_model import calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
from structure_generator import generate_valid_columns
from config import FREQS


# ---------------------------------------------------------------------------
# Error Metrics
# ---------------------------------------------------------------------------
def calculate_rmse(simulated, ideal):
    """Compute RMSE between simulated and ideal responses."""
    simulated = np.array(simulated)
    ideal = np.array(ideal)
    return np.sqrt(np.mean((simulated - ideal) ** 2))


def check_phase_linearity(S21_complex, freqs):
    """Quantify phase linearity using RMSE from a linear fit."""
    phase_unwrapped = np.unwrap(np.angle(S21_complex))
    coeffs = np.polyfit(freqs, phase_unwrapped, deg=1)
    phase_fit = np.polyval(coeffs, freqs)
    return np.sqrt(np.mean((phase_unwrapped - phase_fit) ** 2))


# ---------------------------------------------------------------------------
# GA Fitness Function
# ---------------------------------------------------------------------------
def fitness_func(ga_instance, solution, solution_idx):
    """Fitness function for the Genetic Algorithm."""
    try:
        valid_columns = generate_valid_columns()
        grid = np.column_stack([valid_columns[int(g)] for g in solution])
        S, W = calculate_S_W_values(grid)
        Z1 = calculate_Z1(S, W)
        S21_dB, _, _, _, S21_complex = calculate_S21_dB(Z1)
        ideal_S21, _, _, _ = ideal_filter(**ga_instance.filter_params)


        magnitude_rmse = calculate_rmse(S21_dB, ideal_S21)
        phase_rmse = check_phase_linearity(S21_complex, FREQS)
        fitness = - (magnitude_rmse + phase_rmse)

        if not np.isfinite(fitness):
            return -1e12
        return fitness

    except Exception:
        import traceback
        with open(f"worker_error_{mp.current_process().pid}.log", "a", encoding="utf-8") as f:
            f.write("".join(traceback.format_exc()) + "\n")
        return -1e12


# ---------------------------------------------------------------------------
# GA Callback: On Generation
# ---------------------------------------------------------------------------
def on_generation(ga_instance):
    """Callback executed after each GA generation."""
    solution, fitness, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    try:
        ga_instance.logger.info(
            f"Generation {ga_instance.generations_completed:02d} | Fitness = {fitness:.4f}"
        )
    except Exception:
        print(f"Generation {ga_instance.generations_completed} | Fitness = {fitness:.4f}")
    
    if hasattr(ga_instance, "gui_update_fn"):
        ga_instance.gui_update_fn(fitness)
