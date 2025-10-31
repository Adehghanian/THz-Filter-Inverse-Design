"""
optimization_methods.py
-----------------------
Modular optimization backends for THz inverse design.

Contains:
    • run_genetic_algorithm() – PyGAD-based optimization
    • run_particle_swarm() – placeholder for PSO
    • run_adjoint_method() – placeholder for adjoint inverse design
"""

import numpy as np
import pygad
import time
from fitness_functions import fitness_func
from structure_generator import generate_valid_columns
from pyswarms.single.global_best import GlobalBestPSO
import config
# ---------------------------------------------------------------------------
# Genetic Algorithm Optimization
# ---------------------------------------------------------------------------
def run_genetic_algorithm(params, update_callback=None, console_callback=None, filter_params=None):
    """
    Runs the Genetic Algorithm with optional live filter parameters from GUI.
    """

    valid_columns = generate_valid_columns()

    # --- GUI callback integration ---
    def on_generation_gui(ga_instance):
        """Send generation info to GUI instead of terminal."""
        solution, fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )
        msg = f"Generation {ga_instance.generations_completed:03d} | Best Fitness = {fitness:.6f}\n"
        if console_callback:
            console_callback(msg)        # Send to GUI
        else:
            print(msg, flush=True)       # Fallback
        if update_callback:
            update_callback(fitness)

    # --- Create GA instance ---
    ga_instance = pygad.GA(
        num_generations=params.get("num_generations", 30),
        num_parents_mating=params.get("num_parents_mating", 10),
        keep_elitism=params.get("keep_elitism", 10),
        fitness_func=fitness_func,
        sol_per_pop=params.get("sol_per_pop", 40),
        num_genes=params.get("num_genes", 200),
        gene_space=list(range(len(valid_columns))),
        parent_selection_type=params.get("selection_type", "tournament"),
        K_tournament=params.get("K_tournament", 10),
        crossover_type=params.get("crossover_type", "two_points"),
        mutation_type=params.get("mutation_type", "inversion"),
        mutation_percent_genes=params.get("mutation_percent_genes", 20),
        random_seed=10,
        stop_criteria="saturate_40",
        on_generation=on_generation_gui,
    )

    # 🔥 Attach live GUI filter parameters for fitness_func()
    if filter_params:
        ga_instance.filter_params = filter_params
    else:
        from config import FILTER_TYPE, FILTER_CENTER_FREQ, FILTER_BANDWIDTH, FILTER_TRANSITION_BW, FILTER_DEPTH_DB, FREQS
        ga_instance.filter_params = {
            "filter_type": FILTER_TYPE,
            "center_frequency": FILTER_CENTER_FREQ,
            "bandwidth": FILTER_BANDWIDTH,
            "transition_bw": FILTER_TRANSITION_BW,
            "depth_dB": FILTER_DEPTH_DB,
            "freqs": FREQS
        }

    # --- Run optimization ---
    ga_instance.run()

    # --- Retrieve results ---
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_grid = np.column_stack([valid_columns[int(g)] for g in best_solution])
    fitness_history = ga_instance.best_solutions_fitness

    if console_callback:
        console_callback(f"\nOptimization complete. Best fitness: {best_fitness:.6f}\n")

    return best_grid, fitness_history


# ---------------------------------------------------------------------------
# Particle Swarm Optimization (placeholder)
# ---------------------------------------------------------------------------
def run_particle_swarm(params, update_callback=None, console_callback=None, filter_params=None):
    """
    Runs Particle Swarm Optimization (PSO) using PySwarms to minimize the custom fitness function.
    Integrates live updates to the GUI via callbacks.
    """

    valid_columns = generate_valid_columns()
    num_valid_cols = len(valid_columns)
    num_genes = config.COLS  # same number of structural columns as GA

    # ---- PSO Parameters ----
    n_particles = params.get("pso_particles", config.PSO_PARTICLES)
    iterations = params.get("pso_iterations", config.PSO_ITERATIONS)
    w = params.get("pso_w", config.PSO_W)
    c1 = params.get("pso_c1", config.PSO_C1)
    c2 = params.get("pso_c2", config.PSO_C2)

    if console_callback:
        console_callback(f"Initializing PSO with {n_particles} particles and {iterations} iterations...\n")

    # ---- Define Fitness Wrapper ----
    def pso_fitness(x):
        # x is (n_particles, num_genes)
        # Convert continuous particle positions → discrete valid column indices
        idx = np.clip(np.round(x).astype(int), 0, num_valid_cols - 1)
        fitness_vals = []
        for particle in idx:
            # Reconstruct full grid from column indices
            grid = np.column_stack([valid_columns[i] for i in particle])
            score = fitness_func(solution=particle, solution_idx=0, filter_params=filter_params)
            fitness_vals.append(score)
        return np.array(fitness_vals)

    # ---- PSO Boundaries ----
    bounds = (np.zeros(num_genes), np.ones(num_genes) * (num_valid_cols - 1))

    # ---- Initialize Optimizer ----
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=num_genes,
        options={'c1': c1, 'c2': c2, 'w': w},
        bounds=bounds
    )

    # ---- Optimization Loop ----
    best_cost_history = []
    for i in range(iterations):
        cost, pos = optimizer.optimize(pso_fitness, iters=1, verbose=False)
        best_cost_history.append(cost)

        msg = f"Iteration {i+1:03d}/{iterations} | Best Fitness = {cost:.6f}\n"
        if console_callback:
            console_callback(msg)
        else:
            print(msg)
        if update_callback:
            update_callback(cost)

    # ---- Best Result ----
    best_position = np.clip(np.round(optimizer.swarm.best_pos).astype(int), 0, num_valid_cols - 1)
    best_grid = np.column_stack([valid_columns[i] for i in best_position])

    if console_callback:
        console_callback(f"\nPSO complete. Best fitness: {cost:.6f}\n")

    return best_grid, np.array(best_cost_history)


# ---------------------------------------------------------------------------
# Adjoint Method Optimization (placeholder)
# ---------------------------------------------------------------------------
def run_adjoint_method(params, update_callback=None, filter_params=None):
    """
    Placeholder for Adjoint Method optimization.
    """
    time.sleep(2)
    print("Running Adjoint Method (placeholder)...")
    dummy_grid = np.ones((18, 200))
    dummy_fitness = np.linspace(-0.5, 0, 20)
    return dummy_grid, dummy_fitness
