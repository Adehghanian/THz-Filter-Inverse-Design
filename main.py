import os
import time
import tempfile
import traceback
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import ansys.aedt.core
import config
import logging
from fitness_functions import fitness_func, on_generation
from thz_filter_model import calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
from visualization import visualize_grid, save_s_parameters_to_csv, plot_s_parameters # <--- Changed import
from optimization_methods import run_genetic_algorithm, run_particle_swarm, run_adjoint_method
import pickle # <--- NEW
import shutil # <--- NEW


# ---------------------------------------------------------------------------
# Multiprocessing config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use 'spawn' on Windows for COM safety
    mp.set_start_method("spawn", force=True)

def setup_logger(name="console_logger", level=logging.DEBUG):
    """Configure and return a formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# Optimization runner
# ---------------------------------------------------------------------------
def run_optimization(selected_method="Genetic Algorithm",
                     update_callback=None,
                     console_callback=None,
                     filter_params=None,
                     params=None):
    """
    Dispatches to the selected optimization method (GA, PSO, Adjoint).
    Expects `params` dict from GUI or config, and live filter_params.
    """
    import importlib, config
    importlib.reload(config)

    logger = setup_logger()
    logger.info(f"Launching optimization method: {selected_method}")

    # Fallback: if params not passed (e.g. run from CLI), use config defaults
    if params is None:
        params = dict(
            num_generations=config.GA_GENERATIONS,
            sol_per_pop=config.GA_POPULATION,
            num_parents_mating=config.GA_PARENTS,
            keep_elitism=config.GA_ELITISM,
            mutation_prob=config.GA_MUTATION_PROB,
            pso_particles=config.PSO_PARTICLES,
            pso_iterations=config.PSO_ITERATIONS,
            pso_w=config.PSO_W,
            pso_c1=config.PSO_C1,
            pso_c2=config.PSO_C2,
            adj_lr=config.ADJ_LEARNING_RATE,
            adj_iterations=config.ADJ_ITERATIONS,
        )

    method_map = {
        "Genetic Algorithm": run_genetic_algorithm,
        "Particle Swarm Optimization": run_particle_swarm,
        "Adjoint Method": run_adjoint_method,
    }

    if selected_method not in method_map:
        raise ValueError(f"Unknown optimization method: {selected_method}")

    # 🔹 Run chosen method
    best_grid, fitness_history = method_map[selected_method](
        params=params,
        update_callback=update_callback,
        console_callback=console_callback,
        filter_params=filter_params,
    )

    # Optional: visualize best result
    visualize_grid(best_grid)
    return best_grid, fitness_history





# ---------------------------------------------------------------------------
# HFSS process launcher (used when running in a separate process)
# ---------------------------------------------------------------------------
# NOTE: This function is not used in the GUI flow, but kept for completeness
def hfss_process_entry(best_grid, console_pipe, stop_flag):
    """Separate process entry for running HFSS safely with COM."""
    from main import run_hfss_simulation

    def console_callback(msg):
        try:
            console_pipe.send(msg)
        except Exception:
            pass

    try:
        # Run HFSS and get data back
        calc_data, hfss_data = run_hfss_simulation(best_grid, console_callback=console_callback, stop_flag=stop_flag)
        # In a real multiprocessing scenario, you'd send data back here
        # For simplicity and aligning with GUI change, we'll just log
        if hfss_data:
            console_callback("HFSS simulation completed and data retrieved.")
            # For this context, we will not implement the complex pipe data transfer
            # that is needed for non-serializable objects like numpy arrays.
        
    except Exception:
        console_callback("HFSS process crashed:\n" + traceback.format_exc())
    finally:
        console_pipe.send("__HFSS_DONE__")
        console_pipe.close()


# ---------------------------------------------------------------------------
# HFSS Simulation (MODIFIED)
# ---------------------------------------------------------------------------
def run_hfss_simulation(best_grid, logger=None, console_callback=None, stop_flag=None):
    """
    Runs HFSS using PyAEDT for the optimized structure.
    
    Returns:
        tuple[dict, dict]: (calculated_data, hfss_data)
    """
    import importlib, config
    importlib.reload(config)

    # --- Setup (same as original) ---
    # ... (setup code remains the same) ...

    def log(msg):
        if console_callback:
            console_callback(msg + "\n")
        elif logger:
            logger.info(msg)
        else:
            print(msg)

    AEDT_VERSION = config.HFSS_VERSION
    NG_MODE = config.HFSS_NON_GRAPHICAL  # note: this means NON-GRAPHICAL=True (headless). Set False if you want the UI.
    # Use TemporaryDirectory only if not using an existing project path, or clean up after.
    # Given the complexity with COM, let's stick to the temp folder logic from original:
    temp_folder = tempfile.TemporaryDirectory(suffix=".ansys") 
    
    calculated_data = None
    hfss_data = None
    
    try:
        project_name = config.HFSS_PROJECT_NAME
        log(f"🚀 Starting HFSS {AEDT_VERSION} simulation...")
        
        # Determine if a project path is specified to reuse. If not, use the temporary folder path.
        if config.HFSS_REUSE_PROJECT and os.path.exists(config.HFSS_SAVE_PATH):
            project_dir = os.path.dirname(config.HFSS_SAVE_PATH)
        else:
            # Copy a base project or simply use the temp dir for the new project
            # For simplicity, we'll let PyAEDT create a new one in the temp dir if not reusing.
            project_dir = temp_folder.name 

        d = ansys.aedt.core.launch_desktop(
            AEDT_VERSION,
            non_graphical=NG_MODE,
            new_desktop=False,
        )

        # Create HFSS design
        hfss = ansys.aedt.core.Hfss(version=AEDT_VERSION, 
                                    project=project_name, # Use name from config
                                    solution_type="Terminal")
        Modeler = hfss.modeler
        hfss["patch_dim"] = "10um"
        length_units = "um"
        freq_units = "THz"
        Modeler.model_units = length_units

        buffer = 80
        CPS_w = 40

        Z_bound = config.COLS * config.CELL_LENGTH * 1e6 + 2 * buffer
        X_bound = 800
        Y_bound = 500

        VAC = Modeler.create_box(origin=[-X_bound/2, -Y_bound/2, 0], sizes=[X_bound, Y_bound, Z_bound], name="Vac")
        hfss.assign_radiation_boundary_to_objects(["Vac"])
        Vac_faces = Modeler.get_object_faces("Vac")



        left_face = 8
        right_face = 7

        CPS1, CPS2, CPS3, CPS4 = "CPS1", "CPS2", "CPS3", "CPS4"
        Modeler.create_rectangle(orientation=ansys.aedt.core.constants.PLANE.ZX, origin=[30, 0, 0],                sizes=[buffer, CPS_w], name=CPS1)
        Modeler.create_rectangle(orientation=ansys.aedt.core.constants.PLANE.ZX, origin=[-70, 0, 0],               sizes=[buffer, CPS_w], name=CPS2)
        Modeler.create_rectangle(orientation=ansys.aedt.core.constants.PLANE.ZX, origin=[30, 0, Z_bound - buffer], sizes=[buffer, CPS_w], name=CPS3)
        Modeler.create_rectangle(orientation=ansys.aedt.core.constants.PLANE.ZX, origin=[-70, 0, Z_bound - buffer],sizes=[buffer, CPS_w], name=CPS4)

        # Build from GA best_grid
        S, W = calculate_S_W_values(best_grid)

        drawn_rects = []
        W_um = W * 1e6
        for col_idx, val in enumerate(W_um):
            if val:
                z_pos = buffer + col_idx * config.CELL_LENGTH * 1e6
                x_pos = val / 2.0

                name_cps1 = f"pix_cps1_{col_idx}"
                Modeler.create_rectangle(
                    orientation=ansys.aedt.core.constants.PLANE.ZX,
                    origin=[30 + CPS_w/2 - x_pos, 0, z_pos],
                    sizes=[config.CELL_LENGTH * 1e6, val],
                    name=name_cps1
                )

                name_cps2 = f"pix_cps2_{col_idx}"
                Modeler.create_rectangle(
                    orientation=ansys.aedt.core.constants.PLANE.ZX,
                    origin=[-70 + CPS_w/2 - x_pos, 0, z_pos],
                    sizes=[config.CELL_LENGTH * 1e6, val],
                    name=name_cps2
                )
                drawn_rects.extend([name_cps1, name_cps2])

        all_sheets = [CPS1, CPS2, CPS3, CPS4] + drawn_rects
        PEC = hfss.assign_perfecte_to_sheets(all_sheets)

        # Ports
        hfss.wave_port(assignment=left_face,  reference="CPS1", integration_line="ZPos", port_on_plane=True, renormalize=False, is_microstrip=False, name="T1")
        hfss.wave_port(assignment=right_face, reference="CPS4", integration_line="ZPos", port_on_plane=True, renormalize=False, is_microstrip=False, name="T2")

        # --- Setup and Sweep ---
        setup = hfss.create_setup(
            name=config.HFSS_SETUP_NAME,
            Frequency=f"{config.HFSS_SETUP_FREQ}THz"
        )
        setup.enable_adaptive_setup_broadband(
            low_frequency=f"{config.HFSS_LOW_FREQ}THz",
            high_frquency=f"{config.HFSS_HIGH_FREQ}THz",
            max_passes=config.HFSS_MAX_PASSES,
            max_delta_s=config.HFSS_MAX_DELTA_S
        )

        hfss.create_linear_step_sweep(
            setup=config.HFSS_SETUP_NAME,
            unit="THz",
            start_frequency=config.HFSS_LOW_FREQ,
            stop_frequency=config.HFSS_HIGH_FREQ,
            step_size=config.HFSS_STEP_SIZE,
            name="Sweep1",
            save_fields=config.HFSS_SAVE_FIELDS,
            save_rad_fields=config.HFSS_SAVE_RAD_FIELDS,
            sweep_type=config.HFSS_SWEEP_TYPE
        )


        
        setup.props['UseABCOnPort'] = True

        # --- Run analysis ---

        hfss.validate_full_design()
        hfss.analyze()
        
        # Save project to the configured path after analysis
        hfss.save_project(config.HFSS_SAVE_PATH)

        # --- Retrieve S-parameters ---
        trace_names = hfss.get_traces_for_plot(category="S")
        S11 = hfss.post.get_solution_data(expressions=trace_names[1])
        S21 = hfss.post.get_solution_data(expressions=trace_names[2])

        # Extract data from HFSS
        freq_hfss = np.array(S11.primary_sweep_values, dtype=float)
        hfss_S11_dB = np.array(S11.data_db20(), dtype=float)
        hfss_S21_dB = np.array(S21.data_db20(), dtype=float)

        # Recalculate analytical (ABCD) data for comparison
        S, W = calculate_S_W_values(best_grid)
        Z1 = calculate_Z1(S, W)
        S21_dB_calc, S11_dB_calc, S21_phase_calc, S11_phase_calc, _ = calculate_S21_dB(Z1, config.FREQS)
        # Use ideal_filter with current config (or passed filter_params)
        ideal_S21, ideal_S11, ideal_S21_phase, ideal_S11_phase = ideal_filter() 

        # --- Prepare data for saving ---
        calculated_data = {
            "Freq_THz": config.FREQS / 1e12,
            "Calc_S11_dB": S11_dB_calc,
            "Calc_S21_dB": S21_dB_calc,
            "Calc_S11_phase_deg": S11_phase_calc,
            "Calc_S21_phase_deg": S21_phase_calc,
            "Ideal_S11_dB": ideal_S11,
            "Ideal_S21_dB": ideal_S21
        }

        hfss_data = {
            "HFSS_Freq_THz": freq_hfss / 1e3, # Convert Hz to THz
            "HFSS_S11_dB": hfss_S11_dB,
            "HFSS_S21_dB": hfss_S21_dB
        }

        log("✅ Data retrieved from HFSS.")
        
        # --- RETURN DATA INSTEAD OF SAVING TO CSV ---
        return calculated_data, hfss_data


    except Exception:
        log("HFSS simulation failed:\n" + traceback.format_exc())
        return calculated_data, None # Return whatever data was calculated or None

    # In main.py, inside run_hfss_simulation:
    finally:
        try:
            if hfss is not None:
                # Use the design object's release_desktop method first
                hfss.release_desktop()
        except Exception:
            pass
        
        # New/modified cleanup logic for the desktop object 'd'
        if 'd' in locals() and d is not None:
            try:
                # Check if it's the desktop object and close it
                if hasattr(d, 'port'):
                    # This branch is for the older PyAEDT API
                    d.force_close()
                else:
                    # This handles the newer PyAEDT Desktop object which should have a close method
                    # Assuming newer PyAEDT versions support .close_desktop()
                    d.close_desktop() 
            except Exception as e_close:
                log(f"Warning: Failed to close desktop object 'd' gracefully: {e_close}")
                
        # Clean up the temporary directory
        if os.path.isdir(temp_folder.name):
            try:
                shutil.rmtree(temp_folder.name)
            except OSError:
                log(f"Warning: Could not remove temporary directory {temp_folder.name}")
        
        log("HFSS session closed and cleaned up.") # Move log outside the loop
        
        # Must return the data in all paths
        return calculated_data, hfss_data


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger = setup_logger()
    grid, history = run_optimization("Genetic Algorithm")
    calc_data, hfss_data = run_hfss_simulation(grid, logger) # <--- Get data back
    
    if calc_data is not None:
        fig = plot_s_parameters(
            base_filename="CLI Run Results",
            calculated_data_dict=calc_data,
            hfss_data_dict=hfss_data,
            best_grid=grid
        )
        if fig:
            plt.show()
