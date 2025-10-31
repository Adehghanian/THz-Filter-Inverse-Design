# THz Filter Inverse Design: Multi-Method Optimization

## 🚀 Overview

This project implements an inverse design workflow for optimizing Terahertz (THz) frequency-selective filters. The core idea is to automatically find the optimal binary pixel pattern for a THz Coplanar Stripline (CPS) structure that matches a desired filter response (e.g., bandstop, bandpass).

The workflow supports a modular approach to optimization:
1.  **Fast Analytical Modeling:** A rapid **ABCD matrix model** is used to quickly calculate the filter's frequency response.
2.  **Modular Optimization:** The design space is explored using a **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**.
3.  **High-Fidelity Validation:** The final optimized structure is validated using **Ansys HFSS** (via PyAEDT) for accurate electromagnetic simulation.

## ✨ Features

* **Modular Optimization:** Supports Genetic Algorithm and Particle Swarm Optimization backends.
* **Analytical Speed:** Uses an analytical ABCD matrix model for fast fitness evaluation.
* **Ideal Filter Target:** Defines and plots target ideal responses (bandstop, bandpass, lowpass, highpass) for comparison.
* **HFSS Integration:** Seamlessly exports the optimized geometry to Ansys HFSS for full wave simulation and comparison plotting.
* **GUI Launcher:** Includes a Tkinter-based GUI (`config_gui.py`) for easy parameter adjustment, ideal filter preview, and pipeline control.

## 🗃️ Project Structure

| File/Directory | Description |
| :--- | :--- |
| `config.py` | Centralized configuration for geometry, constants, frequency sweeps, ideal filter targets, and all optimization parameters (GA/PSO/Adjoint). |
| `main.py` | The main execution entry point. Handles running the chosen optimization method and launching the subsequent HFSS validation. |
| `optimization_methods.py` | Implements the **Genetic Algorithm (GA)** using PyGAD and **Particle Swarm Optimization (PSO)** using PySwarms. |
| `fitness_functions.py` | Defines the fitness metric, primarily based on the **Root Mean Square Error (RMSE)** between the simulated and ideal S-parameters. |
| `thz_filter_model.py` | Implements the **ABCD matrix model** for fast, analytical calculation of S-parameters. |
| `structure_generator.py` | Logic for generating all valid, vertically symmetric structural columns used as genes/particles. |
| `config_gui.py` | The Graphical User Interface (GUI) for parameter configuration and workflow execution. |
| `visualization.py` | Utilities for plotting S-parameters and visualizing the optimized 2D grid structure. |

## ⚙️ Requirements and Setup

### 1. Software
* **Python 3.8+**
* **Ansys AEDT (HFSS)** (Tested with 2024 R1, 2025 R1)

### 2. Python Packages
Install the required libraries, including those for the two implemented optimization methods:

```bash
pip install numpy scipy pandas matplotlib pygad pyswarms ansys-aedt-core
```

## ▶️ How to Run the Project

The project is designed to be run via the GUI for the best experience.

### Option 1: Using the GUI (Recommended)

Run the GUI to visually set all parameters, preview the ideal filter, and control the simulation pipeline:

```bash
python config_gui.py
```
Then:
1. Navigate to the Filter Settings tab to define your ideal target (e.g., "bandstop" at 0.6 THz).

2. Navigate to the Optimization Settings tab and select either Genetic Algorithm or Particle Swarm Optimization.

3. Click "▶ Run Optimization" to start the inverse design process.

4. After optimization, click "🧠 Run HFSS Simulation" to launch the high-fidelity validation.

## 📊 Results and Visualization

Optimization results, including the **final optimized grid**, calculated **ABCD S-parameters**, and **HFSS-simulated S-parameters**, are saved to the directory specified by `HFSS_EXPORT_DIR` in `config.py`.

The visualization utility generates a multi-plot figure comparing:

* The fast **ABCD Model** response against the **Ideal Target**.
* The final **ABCD Model** response against the **HFSS Simulation** response.

## 🙏 Acknowledgements

This project was conducted at the **THz Lab** under the supervision of **Dr. Levi Smith and Prof. Thomas Darcie**.

Dehghanian, Ali, Thomas Darcie, and Levi Smith. "Genetic Algorithm-Based Inverse Design of Guided Wave Planar Terahertz Filters." arXiv preprint arXiv:2506.03372 (2025).
