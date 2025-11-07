"""
fitness_functions.py
---------------------------------------------------------------------------
Defines fitness evaluation and helper metrics for the GA optimization.
"""

import numpy as np
import multiprocessing as mp
from thz_filter_model import calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
from structure_generator import generate_valid_columns
from config import FREQS


# ---------------------------------------------------------------------------
# ERROR METRICS
# ---------------------------------------------------------------------------
def calculate_rmse(simulated, ideal):
    """Compute Root Mean Square Error (RMSE) between simulated and ideal responses."""
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
# GA FITNESS FUNCTION
# ---------------------------------------------------------------------------
def fitness_func(ga_instance, solution, solution_idx):
    import importlib, config
    importlib.reload(config)
    """
    Fitness function for the Genetic Algorithm.

    Calculates the fitness based on the negative sum of magnitude RMSE 
    and phase linearity RMSE relative to the ideal filter.
    """
    try:
        valid_columns = generate_valid_columns()
        grid = np.column_stack([valid_columns[int(g)] for g in solution])
        
        # 1. Calculate analytical response
        S, W = calculate_S_W_values(grid)
        Z1 = calculate_Z1(S, W)
        S21_dB, _, _, _, S21_complex = calculate_S21_dB(Z1, config.FREQS)
        
        # 2. Get ideal response (uses filter_params attached to ga_instance)
        ideal_S21, _, _, _ = ideal_filter(**ga_instance.filter_params)

        # 3. Calculate error metrics
        magnitude_rmse = calculate_rmse(S21_dB, ideal_S21)
        phase_rmse = check_phase_linearity(S21_complex, config.FREQS)
        
        # 4. Compute final fitness (maximization problem)
        fitness = - (magnitude_rmse + phase_rmse)

        if not np.isfinite(fitness):
            return -1e12
        return fitness

    except Exception:
        # Log error for multiprocessing workers
        import traceback
        with open(f"worker_error_{mp.current_process().pid}.log", "a", encoding="utf-8") as f:
            f.write("".join(traceback.format_exc()) + "\n")
        return -1e12


# ---------------------------------------------------------------------------
# GA CALLBACK: ON GENERATION
# ---------------------------------------------------------------------------
def on_generation(ga_instance):
    """
    Callback executed after each GA generation. 
    Logs the best fitness and calls the optional GUI update function.
    """
    solution, fitness, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    
    try:
        # Log to PyGAD's internal logger
        ga_instance.logger.info(
            f"Generation {ga_instance.generations_completed:02d} | Fitness = {fitness:.4f}"
        )
    except Exception:
        # Fallback print statement
        print(f"Generation {ga_instance.generations_completed} | Fitness = {fitness:.4f}")
    
    # Call GUI update function if provided
    if hasattr(ga_instance, "gui_update_fn"):
        ga_instance.gui_update_fn(fitness)
