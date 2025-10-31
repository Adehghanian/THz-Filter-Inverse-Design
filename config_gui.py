import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re
import subprocess
import threading
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from config import FREQS
from thz_filter_model import ideal_filter
from visualization import visualize_grid
from main import run_optimization, run_hfss_simulation

# ---------------------------------------------------------------------------
# Helper: Save updated values to config.py
# ---------------------------------------------------------------------------
import importlib

def update_config_file(new_values):
    """Overwrite config.py constants with updated GUI values and reload the module."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for key, val in new_values.items():
        pattern = re.compile(rf"^{key}\s*=\s*.*$", re.MULTILINE)
        replacement = f"{key} = {val}\n"
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                lines[i] = replacement
                break
        else:
            # key not found: append
            lines.append(replacement)

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 🔄 Reload config immediately
    import config
    importlib.reload(config)

    messagebox.showinfo("Success", "Configuration saved and reloaded successfully!")



# ---------------------------------------------------------------------------
# GUI Setup
# ---------------------------------------------------------------------------
def launch_config_gui():
    # --- Root window FIRST (fixes earlier scope issues) ---
    root = tk.Tk()
    root.title("THz Inverse Design Configuration & Launcher")
    root.geometry("900x760")
    root.resizable(False, False)

    # ---- Styling ----
    style = ttk.Style(root)
    style.configure("TLabel", font=("Segoe UI", 10))
    style.configure("TEntry", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10, "bold"))

    # Notebook for sections
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # ================= HFSS TAB =================
    hfss_tab = ttk.Frame(notebook)
    notebook.add(hfss_tab, text="HFSS Settings")

    ttk.Label(hfss_tab, text="HFSS Version:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    hfss_version = tk.StringVar(value=config.HFSS_VERSION)
    ttk.Entry(hfss_tab, textvariable=hfss_version, width=25).grid(row=0, column=1, sticky="w")

    hfss_headless = tk.BooleanVar(value=config.HFSS_NON_GRAPHICAL)
    ttk.Checkbutton(hfss_tab, text="Run in Headless (Non-Graphical) Mode", variable=hfss_headless)\
        .grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    ttk.Label(hfss_tab, text="Project Name:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    hfss_project = tk.StringVar(value=config.HFSS_PROJECT_NAME)
    ttk.Entry(hfss_tab, textvariable=hfss_project, width=40).grid(row=2, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Save Path:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    hfss_save_path = tk.StringVar(value=config.HFSS_SAVE_PATH)

    def browse_hfss_path():
        path = filedialog.askdirectory(title="Select HFSS Save Directory")
        if path:
            hfss_save_path.set(os.path.join(path, os.path.basename(config.HFSS_SAVE_PATH)))

    ttk.Entry(hfss_tab, textvariable=hfss_save_path, width=50).grid(row=3, column=1, sticky="w")
    ttk.Button(hfss_tab, text="Browse", command=browse_hfss_path).grid(row=3, column=2, padx=5, sticky="w")

        # ================= HFSS SETUP PARAMETERS =================
    ttk.Separator(hfss_tab, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 5))
    ttk.Label(hfss_tab, text="HFSS Setup Parameters", font=("Segoe UI Semibold", 10)).grid(
        row=5, column=0, columnspan=3, padx=10, sticky="w"
    )

    ttk.Label(hfss_tab, text="Setup Name:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
    hfss_setup_name = tk.StringVar(value=getattr(config, "HFSS_SETUP_NAME", "new_setup"))
    ttk.Entry(hfss_tab, textvariable=hfss_setup_name, width=25).grid(row=6, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Setup Frequency (THz):").grid(row=7, column=0, padx=10, pady=5, sticky="w")
    hfss_setup_freq = tk.DoubleVar(value=getattr(config, "HFSS_SETUP_FREQ", 1.0))
    ttk.Entry(hfss_tab, textvariable=hfss_setup_freq, width=10).grid(row=7, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Low/Start Frequency (THz):").grid(row=8, column=0, padx=10, pady=5, sticky="w")
    hfss_low_freq = tk.DoubleVar(value=getattr(config, "HFSS_LOW_FREQ", 0.1))
    ttk.Entry(hfss_tab, textvariable=hfss_low_freq, width=10).grid(row=8, column=1, sticky="w")

    ttk.Label(hfss_tab, text="High/Stop Frequency (THz):").grid(row=9, column=0, padx=10, pady=5, sticky="w")
    hfss_high_freq = tk.DoubleVar(value=getattr(config, "HFSS_HIGH_FREQ", 1.2))
    ttk.Entry(hfss_tab, textvariable=hfss_high_freq, width=10).grid(row=9, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Max Passes:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
    hfss_max_passes = tk.IntVar(value=getattr(config, "HFSS_MAX_PASSES", 20))
    ttk.Entry(hfss_tab, textvariable=hfss_max_passes, width=10).grid(row=10, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Max ΔS:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
    hfss_max_delta_s = tk.DoubleVar(value=getattr(config, "HFSS_MAX_DELTA_S", 0.5))
    ttk.Entry(hfss_tab, textvariable=hfss_max_delta_s, width=10).grid(row=11, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Step Size (THz):").grid(row=12, column=0, padx=10, pady=5, sticky="w")
    hfss_step_size = tk.DoubleVar(value=getattr(config, "HFSS_STEP_SIZE", 0.02))
    ttk.Entry(hfss_tab, textvariable=hfss_step_size, width=10).grid(row=12, column=1, sticky="w")

    ttk.Label(hfss_tab, text="Sweep Type:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
    hfss_sweep_type = tk.StringVar(value=getattr(config, "HFSS_SWEEP_TYPE", "Discrete"))
    ttk.Combobox(
        hfss_tab, textvariable=hfss_sweep_type,
        values=["Discrete", "Interpolating"], width=15, state="readonly"
    ).grid(row=13, column=1, sticky="w")

    hfss_save_fields = tk.BooleanVar(value=getattr(config, "HFSS_SAVE_FIELDS", False))
    ttk.Checkbutton(hfss_tab, text="Save Fields", variable=hfss_save_fields).grid(
        row=14, column=0, columnspan=2, padx=10, pady=3, sticky="w"
    )

    hfss_save_rad_fields = tk.BooleanVar(value=getattr(config, "HFSS_SAVE_RAD_FIELDS", False))
    ttk.Checkbutton(hfss_tab, text="Save Radiation Fields", variable=hfss_save_rad_fields).grid(
        row=15, column=0, columnspan=2, padx=10, pady=3, sticky="w"
    )



    # ================= FILTER TAB =================
    filter_tab = ttk.Frame(notebook)
    notebook.add(filter_tab, text="Filter Settings")

    ttk.Label(filter_tab, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    filter_type = tk.StringVar(value=config.FILTER_TYPE)
    ttk.Combobox(
        filter_tab,
        textvariable=filter_type,
        values=["bandstop", "bandpass", "lowpass", "highpass"],
        width=18, state="readonly"
    ).grid(row=0, column=1, sticky="w", padx=5)

    ttk.Label(filter_tab, text="Center Frequency (THz):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    filter_cf = tk.DoubleVar(value=config.FILTER_CENTER_FREQ / 1e12)
    ttk.Entry(filter_tab, textvariable=filter_cf, width=10).grid(row=1, column=1, sticky="w")

    ttk.Label(filter_tab, text="Bandwidth (THz):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    filter_bw = tk.DoubleVar(value=config.FILTER_BANDWIDTH / 1e12)
    ttk.Entry(filter_tab, textvariable=filter_bw, width=10).grid(row=2, column=1, sticky="w")

    ttk.Label(filter_tab, text="Transition Bandwidth (THz):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    filter_tbw = tk.DoubleVar(value=config.FILTER_TRANSITION_BW / 1e12)
    ttk.Entry(filter_tab, textvariable=filter_tbw, width=10).grid(row=3, column=1, sticky="w")

    ttk.Label(filter_tab, text="Stopband Depth (dB):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
    filter_depth = tk.DoubleVar(value=config.FILTER_DEPTH_DB)
    ttk.Entry(filter_tab, textvariable=filter_depth, width=10).grid(row=4, column=1, sticky="w")


    # ================ Embedded Ideal Filter Plot =================
    plot_frame = ttk.Frame(filter_tab)
    plot_frame.grid(row=7, column=0, columnspan=3, pady=(5, 5), sticky="ew")

    fig = Figure(figsize=(8, 4), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.4)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(padx=5, pady=5, fill="both", expand=False)

    def preview_ideal_filter():
        # 🔹 1. Save the current GUI settings first (writes to config.py)
        save_config()

        # 🔹 2. Reload the updated config module
        import importlib, config
        importlib.reload(config)

        # 🔹 3. Get current GUI values (from Tkinter variables)
        f_type = filter_type.get()
        cf = filter_cf.get() * 1e12
        bw = filter_bw.get() * 1e12
        tbw = filter_tbw.get() * 1e12
        depth = filter_depth.get()

        # 🔹 4. Recalculate frequency array based on the new HFSS settings
        freqs = config.FREQS

        # 🔹 5. Compute the ideal response
        ideal_S21, ideal_S11, ideal_S21_phase, ideal_S11_phase = ideal_filter(
            filter_type=f_type,
            center_frequency=cf,
            bandwidth=bw,
            transition_bw=tbw,
            depth_dB=depth,
            freqs=freqs
        )

        # 🔹 6. Plot magnitude and phase responses
        ax1.clear(); ax2.clear()
        for ax in (ax1, ax2):
            ax.grid(True, linewidth=0.4)

        ax1.plot(freqs / 1e12, ideal_S21, label="|S21| (dB)", linewidth=0.9)
        ax1.plot(freqs / 1e12, ideal_S11, label="|S11| (dB)", linewidth=0.9)
        ax1.set_title(f"Ideal {f_type.capitalize()} Filter", fontsize=9, weight="bold")
        ax1.set_ylabel("Mag (dB)", fontsize=8)
        ax1.legend(fontsize=7, loc="best")

        ax2.plot(freqs / 1e12, ideal_S21_phase, label="∠S21", linewidth=0.9)
        ax2.plot(freqs / 1e12, ideal_S11_phase, label="∠S11", linewidth=0.9)
        ax2.set_xlabel("Frequency (THz)", fontsize=8)
        ax2.set_ylabel("Phase (°)", fontsize=8)
        ax2.legend(fontsize=7, loc="best")

        fig.tight_layout(pad=1)
        canvas.draw()


    # Right-side preview button
    btn_container = tk.Frame(filter_tab, bg="#533E03", borderwidth=1, relief="solid")
    btn_container.grid(row=0, column=2, rowspan=10, padx=10, pady=10, sticky="n")

    preview_btn = tk.Button(
        btn_container,
        text="📊 Preview Ideal Filter Response",
        font=("Segoe UI Semibold", 10),
        bg="#866501", fg="white",
        activebackground="#106EBE", activeforeground="white",
        relief="flat", bd=0, cursor="hand2", width=30, height=4,
        command=preview_ideal_filter
    )
    preview_btn.pack(padx=4, pady=4)

    def on_enter(_): preview_btn.config(bg="#1890F1")
    def on_leave(_): preview_btn.config(bg="#866501")
    preview_btn.bind("<Enter>", on_enter)
    preview_btn.bind("<Leave>", on_leave)
    filter_tab.grid_columnconfigure(2, weight=1)

    # ================= OPTIMIZATION TAB =================
    opt_tab = ttk.Frame(notebook)
    notebook.add(opt_tab, text="Optimization Settings")
    # ================= LOG / RUN TAB =================
    run_tab = ttk.Frame(notebook)
    notebook.add(run_tab, text="Run Simulation")

    ttk.Label(run_tab, text="Console Output:").pack(anchor="w", padx=10, pady=5)
    console_box = scrolledtext.ScrolledText(run_tab, wrap="word", height=20, width=100, state="disabled")
    console_box.pack(padx=10, pady=5, fill="both", expand=True)



    # ================= CONVERGENCE TAB =================
    convergence_tab = ttk.Frame(notebook)
    notebook.add(convergence_tab, text="Convergence")

    fig_conv = Figure(figsize=(6, 4), dpi=100)
    ax_conv = fig_conv.add_subplot(111)
    ax_conv.set_title("GA Convergence")
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Best Fitness")
    canvas_conv = FigureCanvasTkAgg(fig_conv, master=convergence_tab)
    canvas_conv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    fitness_values = []



    # ---------- Algorithm selector (Row 0-2) ----------
    opt_method_frame = ttk.Frame(opt_tab)
    opt_method_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Label(opt_method_frame, text="Select Optimization Algorithm:").pack(side="left", padx=(0, 10))
    opt_method = tk.StringVar(value="Genetic Algorithm")
    opt_algorithms = ["Genetic Algorithm", "Particle Swarm Optimization", "Adjoint Method"]
    opt_combo = ttk.Combobox(
        opt_method_frame, textvariable=opt_method,
        values=opt_algorithms,
        width=30, state="readonly"
    )
    opt_combo.pack(side="left")


    # ================== OPTIMIZATION PARAMETERS (Container) ==================
    ttk.Separator(opt_tab, orient="horizontal").pack(fill="x", padx=10, pady=5)
    ttk.Label(opt_tab, text="Algorithm Parameters", font=("Segoe UI Semibold", 10)).pack(anchor="w", padx=10)
    
    # 💥 NEW: Frame to hold the parameters for the currently selected algorithm
    parameters_frame = ttk.Frame(opt_tab)
    parameters_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # ---------- Genetic Algorithm parameters FRAME ----------
    ga_frame = ttk.Frame(parameters_frame)
    row = 0
    
    ttk.Label(ga_frame, text="Generations:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    ga_generations = tk.IntVar(value=getattr(config, "GA_GENERATIONS", 30))
    ttk.Entry(ga_frame, textvariable=ga_generations, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(ga_frame, text="Population Size:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    ga_population = tk.IntVar(value=getattr(config, "GA_POPULATION", 40))
    ttk.Entry(ga_frame, textvariable=ga_population, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(ga_frame, text="Parents Mating:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    ga_parents = tk.IntVar(value=getattr(config, "GA_PARENTS", 10))
    ttk.Entry(ga_frame, textvariable=ga_parents, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(ga_frame, text="Elitism:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    ga_elitism = tk.IntVar(value=getattr(config, "GA_ELITISM", 10))
    ttk.Entry(ga_frame, textvariable=ga_elitism, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(ga_frame, text="Mutation Probability:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    ga_mutation_prob = tk.DoubleVar(value=getattr(config, "GA_MUTATION_PROB", 0.1))
    ttk.Entry(ga_frame, textvariable=ga_mutation_prob, width=10).grid(row=row-1, column=1, sticky="w")


    # ---------- PSO parameters FRAME ----------
    pso_frame = ttk.Frame(parameters_frame)
    row = 0
    
    ttk.Label(pso_frame, text="Particles:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    pso_particles = tk.IntVar(value=getattr(config, "PSO_PARTICLES", 30))
    ttk.Entry(pso_frame, textvariable=pso_particles, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(pso_frame, text="Iterations:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    pso_iterations = tk.IntVar(value=getattr(config, "PSO_ITERATIONS", 50))
    ttk.Entry(pso_frame, textvariable=pso_iterations, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(pso_frame, text="Inertia Weight (w):").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    pso_w = tk.DoubleVar(value=getattr(config, "PSO_W", 0.7))
    ttk.Entry(pso_frame, textvariable=pso_w, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(pso_frame, text="Cognitive Coeff (c1):").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    pso_c1 = tk.DoubleVar(value=getattr(config, "PSO_C1", 1.5))
    ttk.Entry(pso_frame, textvariable=pso_c1, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(pso_frame, text="Social Coeff (c2):").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    pso_c2 = tk.DoubleVar(value=getattr(config, "PSO_C2", 1.5))
    ttk.Entry(pso_frame, textvariable=pso_c2, width=10).grid(row=row-1, column=1, sticky="w")


    # ---------- Adjoint Method parameters FRAME ----------
    adj_frame = ttk.Frame(parameters_frame)
    row = 0
    
    ttk.Label(adj_frame, text="Learning Rate:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    adj_lr = tk.DoubleVar(value=getattr(config, "ADJ_LEARNING_RATE", 0.05))
    ttk.Entry(adj_frame, textvariable=adj_lr, width=10).grid(row=row-1, column=1, sticky="w")

    ttk.Label(adj_frame, text="Iterations:").grid(row=row, column=0, padx=10, pady=5, sticky="w"); row+=1
    adj_iterations = tk.IntVar(value=getattr(config, "ADJ_ITERATIONS", 100))
    ttk.Entry(adj_frame, textvariable=adj_iterations, width=10).grid(row=row-1, column=1, sticky="w")


    # ---------- Enable/disable logic (Now showing/hiding frames) ----------
    def update_opt_fields(*_):
        method = opt_method.get()
        
        # 1. Unpack (hide) all parameter frames
        for frame in [ga_frame, pso_frame, adj_frame]:
            frame.pack_forget()
        
        # 2. Pack (show) the selected frame
        if method == "Genetic Algorithm":
            ga_frame.pack(fill="both", expand=True)
        elif method == "Particle Swarm Optimization":
            pso_frame.pack(fill="both", expand=True)
        elif method == "Adjoint Method":
            adj_frame.pack(fill="both", expand=True)

    opt_method.trace_add("write", update_opt_fields)
    
    # 3. Initial call to set the default visible frame
    update_opt_fields() 


    # ===== RESULTS TAB =====
    results_tab = ttk.Frame(notebook)
    notebook.add(results_tab, text="Results")
    results_plot_frame = ttk.Frame(results_tab)
    results_plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    ttk.Label(results_tab, text="Optimized Structure Visualization").pack(anchor="w", padx=10, pady=5)

    def show_results_figure(fig):
        """Embed a matplotlib figure into the Results tab."""
        # Clear old figure
        for widget in results_plot_frame.winfo_children():
            widget.destroy()
        if fig is None:
            return
        canvas = FigureCanvasTkAgg(fig, master=results_plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
    # -------------------------------------------------------------------
    # Helper functions (now safely in scope)
    # -------------------------------------------------------------------
    def write_to_console(msg):
        def _append():
            console_box.configure(state="normal")
            console_box.insert("end", msg)
            console_box.see("end")
            console_box.configure(state="disabled")
        root.after(0, _append)


    def update_convergence_plot(new_fitness):
        fitness_values.append(new_fitness)
        ax_conv.clear()
        ax_conv.plot(fitness_values, linewidth=1.2)
        ax_conv.set_title("Optimization Convergence", fontsize=10, weight="bold")
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Best Fitness")
        ax_conv.grid(True)
        canvas_conv.draw()

    # Store optimized grid so HFSS button can use it
    root.best_grid = None

    def run_selected_optimization():
        selected_method = opt_method.get()
        write_to_console(f"\nRunning {selected_method}...\n")

        # 🔥 NEW: automatically save the current GUI values before running
        save_config()  # <---- add this line here
        # clear old convergence
        fitness_values.clear()
        ax_conv.clear()
        ax_conv.set_title("GA Convergence")
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Best Fitness")
        canvas_conv.draw()

        def update_plot_safe(new_fitness):
            # Schedule GUI-safe update via root.after()
            root.after(0, lambda: update_convergence_plot(new_fitness))
        

        filter_params = {
        "filter_type": filter_type.get(),
        "center_frequency": filter_cf.get() * 1e12,
        "bandwidth": filter_bw.get() * 1e12,
        "transition_bw": filter_tbw.get() * 1e12,
        "depth_dB": filter_depth.get(),
        }



        def worker():
            best_grid, fitness_history = run_optimization(
                selected_method=selected_method,
                update_callback=update_plot_safe,   # ✅ use the defined function
                console_callback=write_to_console,  # ✅ use your GUI console writer
                filter_params=filter_params
            )

            write_to_console("\nOptimization finished successfully.\n")
            root.best_grid = best_grid
            run_hfss_btn.configure(state="normal")
            write_to_console("\nYou can now run the HFSS simulation.\n")
            # --- 🔥 NEW: Save ABCD + Ideal results to CSV BEFORE HFSS ---
            import importlib, config
            importlib.reload(config)
            from thz_filter_model import (
                calculate_S_W_values, calculate_Z1, calculate_S21_dB, ideal_filter
            )
            from visualization import save_s_parameters_to_csv, plot_s_parameters_from_csv

            S, W = calculate_S_W_values(best_grid)
            Z1 = calculate_Z1(S, W)
            S21_dB_calc, S11_dB_calc, S21_phase_calc, S11_phase_calc, _ = calculate_S21_dB(Z1)

            freqs = config.FREQS
            ideal_S21, ideal_S11, ideal_S21_phase, ideal_S11_phase = ideal_filter(
                filter_type=filter_params["filter_type"],
                center_frequency=filter_params["center_frequency"],
                bandwidth=filter_params["bandwidth"],
                transition_bw=filter_params["transition_bw"],
                depth_dB=filter_params["depth_dB"],
                freqs=freqs
            )

            calculated_data = {
                "Freq_THz": freqs / 1e12,
                "Calc_S11_dB": S11_dB_calc,
                "Calc_S21_dB": S21_dB_calc,
                "Calc_S11_phase_deg": S11_phase_calc,
                "Calc_S21_phase_deg": S21_phase_calc,
                "Ideal_S11_dB": ideal_S11,
                "Ideal_S21_dB": ideal_S21
            }

            save_s_parameters_to_csv(
                base_filename="S_parameters_advanced",
                calculated_data=calculated_data,
                save_dir=config.HFSS_EXPORT_DIR
            )
            write_to_console("✅ Pre-HFSS S-parameters saved to CSV.\n")
            import importlib, visualization
            importlib.reload(visualization)
            fig_pre = visualization.plot_s_parameters_from_csv("S_parameters_advanced", best_grid=root.best_grid)

            root.after(0, lambda: show_results_figure(fig_pre))


        threading.Thread(target=worker, daemon=True).start()


    def run_hfss_after_gui():
        if root.best_grid is None or root.best_grid.size == 0:
            messagebox.showwarning("No optimized grid", "Please run optimization first.")
            return

        if not messagebox.askyesno(
            "Confirm HFSS Run",
            "This will close the GUI and open HFSS simulation.\nDo you want to continue?"
        ):
            return

        best_grid = root.best_grid

        def hfss_runner():
            try:
                write_to_console("⚙️ HFSS simulation started...\n")
                run_hfss_simulation(best_grid)
                write_to_console("✅ HFSS simulation completed successfully.\n")
            except Exception as e:
                write_to_console(f"❌ HFSS simulation failed: {e}\n")
            finally:
                # reopen result viewer *after* HFSS finishes
                subprocess.Popen(
                    [sys.executable, os.path.abspath(sys.argv[0]), "--gui_results"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )



        # 🔹 make it non-daemon, so it survives after GUI close
        t = threading.Thread(target=hfss_runner)
        t.start()

        # 🔹 give HFSS thread time to start before closing GUI
        root.after(2000, root.destroy)








    # -------------------------------------------------------------------
    # Buttons (one clean block)
    # -------------------------------------------------------------------
    button_frame = ttk.Frame(root)
    button_frame.pack(fill="x", pady=10)

    def save_config():
        new_values = {
            "HFSS_VERSION": f'"{hfss_version.get()}"',
            "HFSS_NON_GRAPHICAL": hfss_headless.get(),
            "HFSS_PROJECT_NAME": f'"{hfss_project.get()}"',
            "HFSS_SAVE_PATH": f'r"{hfss_save_path.get()}"',
            "FILTER_TYPE": f'"{filter_type.get()}"',
            "FILTER_CENTER_FREQ": f"{filter_cf.get()}e12",
            "FILTER_BANDWIDTH": f"{filter_bw.get()}e12",
            "FILTER_TRANSITION_BW": f"{filter_tbw.get()}e12",
            "FILTER_DEPTH_DB": f"{filter_depth.get()}",
            "HFSS_SETUP_NAME": f'"{hfss_setup_name.get()}"',
            "HFSS_SETUP_FREQ": f"{hfss_setup_freq.get()}",
            "HFSS_LOW_FREQ": f"{hfss_low_freq.get()}",
            "HFSS_HIGH_FREQ": f"{hfss_high_freq.get()}",
            "HFSS_MAX_PASSES": f"{hfss_max_passes.get()}",
            "HFSS_MAX_DELTA_S": f"{hfss_max_delta_s.get()}",
            "HFSS_STEP_SIZE": f"{hfss_step_size.get()}",
            "HFSS_SWEEP_TYPE": f'"{hfss_sweep_type.get()}"',
            "HFSS_SAVE_FIELDS": hfss_save_fields.get(),
            "HFSS_SAVE_RAD_FIELDS": hfss_save_rad_fields.get(),
            "GA_GENERATIONS": f"{ga_generations.get()}",
            "GA_POPULATION": f"{ga_population.get()}",
            "GA_PARENTS": f"{ga_parents.get()}",
            "GA_ELITISM": f"{ga_elitism.get()}",
            "GA_MUTATION_PROB": f"{ga_mutation_prob.get()}",
            "PSO_PARTICLES": f"{pso_particles.get()}",
            "PSO_ITERATIONS": f"{pso_iterations.get()}",
            "PSO_W": f"{pso_w.get()}",
            "PSO_C1": f"{pso_c1.get()}",
            "PSO_C2": f"{pso_c2.get()}",
            "ADJ_LEARNING_RATE": f"{adj_lr.get()}",
            "ADJ_ITERATIONS": f"{adj_iterations.get()}",
        }
        update_config_file(new_values)

    ttk.Button(button_frame, text="💾 Save Configuration",
                command=save_config).pack(side="left", padx=20)

    ttk.Button(button_frame, text="▶ Run Optimization",
                command=run_selected_optimization).pack(side="right", padx=20)

    run_hfss_btn = ttk.Button(button_frame, text="🧠 Run HFSS Simulation",
                                command=run_hfss_after_gui, state="disabled")
    run_hfss_btn.pack(side="right", padx=10)

    
    
    
        # ----------------------------------------------------------------------
    # Auto-load results if GUI reopened after HFSS run
    # ----------------------------------------------------------------------

    from visualization import plot_s_parameters_from_csv

    if "--show_results" in sys.argv:
        csv_path = os.path.join(config.HFSS_EXPORT_DIR, "S_parameters_advanced.csv")
        if os.path.exists(csv_path):
            write_to_console("\nDetected completed HFSS run. Loading results...\n")
            try:
                plot_s_parameters_from_csv("S_parameters_advanced")
                visualize_grid(None)  # visualize last grid
                write_to_console("Results loaded successfully.\n")
            except Exception as e:
                write_to_console(f"Error loading results: {e}\n")
        else:
            write_to_console("No results file found to display.\n")





    # ----------------------------------------------------------------------
    # Auto-open Results tab if launched with flag
    # ----------------------------------------------------------------------
    if "--gui_results" in sys.argv:
        try:
            notebook.select(results_tab)                    # jump to Results tab
            write_to_console("\nLoading HFSS results...\n")
            from visualization import plot_s_parameters_from_csv
            fig = plot_s_parameters_from_csv("S_parameters_advanced")
            show_results_figure(fig)
            write_to_console("✅ Results loaded and displayed.\n")
        except Exception as e:
            write_to_console(f"❌ Could not load results: {e}\n")

    root.mainloop()


if __name__ == "__main__":
    if "--gui_results" in sys.argv:
        launch_config_gui()             # reuse same GUI logic
    else:
        launch_config_gui()             # normal start