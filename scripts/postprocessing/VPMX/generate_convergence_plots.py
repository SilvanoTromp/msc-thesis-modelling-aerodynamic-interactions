
"""
Plot VPMchi load convergence plots for all simulation cases.

The script:
1. Searches all simulation folders in results/VPMx.
2. Reads the loading_history.csv file from each simulation folder.
3. Extracts total-aircraft force and moment histories.
4. Uses the simulation time directly from the CSV file.
5. Calculates mean loads and standard deviations over the last 2000 samples.
6. Plots Fx, Fy, Fz, Mx, My, and Mz versus simulation time.
7. Displays sampling-phase mean loads as horizontal lines.
8. Applies common y-axis limits to forces and moments separately.
9. Saves convergence plots to results/convergence_plots/VPMx.

All forces are in N and moments in N m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================
# Settings
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

VPMX_DIR = PROJECT_ROOT / "results" / "VPMx"

OUTPUT_DIR = PROJECT_ROOT / "plots" / "convergence_plots" / "VPMx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of final samples used for averaging
N_LAST = 2000

# Figure settings
FIGURE_SIZE = (15, 8)
FIGURE_DPI = 300

# ==============================================================
# Load definitions
# ==============================================================

LOAD_VARIABLES = [
    "total_Fx_N",
    "total_Fy_N",
    "total_Fz_N",
    "total_Mx_Nm",
    "total_My_Nm",
    "total_Mz_Nm",
]

LOAD_NAMES = [
    "FX", "FY", "FZ", "MX", "MY", "MZ"
]

LOAD_LABELS = [
    r"$F_x$ [N]",
    r"$F_y$ [N]",
    r"$F_z$ [N]",
    r"$M_x$ [N m]",
    r"$M_y$ [N m]",
    r"$M_z$ [N m]",
]

# ==============================================================
# Process individual simulation
# ==============================================================

def process_simulation(case_dir):
    """
    Read VPMchi loading history and save total-aircraft convergence plot.
    """

    # ----------------------------------------------------------
    # Locate loading history file
    # ----------------------------------------------------------

    loads_file = (
        case_dir
        / "out_cuda_run"
        / f"{case_dir.name}_loading_history.csv"
    )

    if not loads_file.is_file():
        print(f"Skipping {case_dir.name}: loading history not found.")
        return

    print(f"\nProcessing: {case_dir.name}")

    # ----------------------------------------------------------
    # Read loading history file
    # ----------------------------------------------------------

    data = pd.read_csv(loads_file)

    # Check required columns
    required_columns = ["time_s"] + LOAD_VARIABLES

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in loading history: {missing_columns}"
        )

    # ----------------------------------------------------------
    # Simulation information
    # ----------------------------------------------------------

    # Extract simulation time [s]
    time = data["time_s"].to_numpy(dtype=float)

    # Number of stored samples
    nsteps_total = len(time)

    if nsteps_total == 0:
        raise ValueError("Loading history contains no samples.")

    # Number of samples used for averaging
    nsteps_sample = min(N_LAST, nsteps_total)

    if nsteps_total < N_LAST:
        print(
            f"WARNING: Only {nsteps_total} samples available. "
            f"Using all samples for averaging."
        )

    # Sampling indices
    sample_start = nsteps_total - nsteps_sample
    sample_slice = slice(sample_start, nsteps_total)

    # Simulation time boundaries
    t_start = time[0]
    t_sample_start = time[sample_start]
    t_total = time[-1]

    print("Simulation information:")
    print("-" * 50)
    print(f"Total samples          : {nsteps_total}")
    print(f"Averaging samples      : {nsteps_sample}")
    print(f"Total simulated time   : {t_total:.4f} s")
    print(f"Averaging start time   : {t_sample_start:.4f} s")

    # ----------------------------------------------------------
    # Extract total-aircraft load histories
    # ----------------------------------------------------------

    load_histories = {}
    mean_loads = {}
    std_loads = {}

    for variable, name in zip(LOAD_VARIABLES, LOAD_NAMES):

        # Extract load history
        history = data[variable].to_numpy(dtype=float)

        load_histories[name] = history

        # Calculate sampling-phase statistics
        sample_history = history[sample_slice]

        mean_loads[name] = np.mean(sample_history)
        std_loads[name] = np.std(sample_history)

    # ----------------------------------------------------------
    # Print mean total-aircraft loads
    # ----------------------------------------------------------

    print("Mean total-aircraft loads:")
    print("-" * 50)

    for name in LOAD_NAMES:

        mean = mean_loads[name]
        std = std_loads[name]

        unit = "N" if name.startswith("F") else "N m"

        print(
            f"{name:2s} = {mean:11.6f} "
            f"+/- {std:9.6f} {unit}"
        )

    # ----------------------------------------------------------
    # Plot total-aircraft load histories
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        2, 3,
        figsize=FIGURE_SIZE,
        sharex=True,
    )

    axes = axes.flatten()

    for i, (axis, name, ylabel) in enumerate(
        zip(axes, LOAD_NAMES, LOAD_LABELS)
    ):

        # ------------------------------------------------------
        # Load history
        # ------------------------------------------------------

        history = load_histories[name]

        axis.plot(
            time,
            history,
            color="black",
            linewidth=1.2,
        )

        # ------------------------------------------------------
        # Sampling-phase boundary
        # ------------------------------------------------------

        axis.axvline(
            t_sample_start,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        # ------------------------------------------------------
        # Sampling-phase mean load
        # ------------------------------------------------------

        mean_value = mean_loads[name]

        axis.hlines(
            mean_value,
            xmin=t_sample_start,
            xmax=t_total,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:.1f}",
        )

        # ------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------

        axis.set_ylabel(ylabel)
        axis.set_xlim(t_start, t_total)

        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="best")

        if i >= 3:
            axis.set_xlabel("Simulation time [s]")

    # ----------------------------------------------------------
    # Apply common y-axis limits for forces and moments
    # ----------------------------------------------------------

    for row_axes in [axes[:3], axes[3:]]:

        # Determine minimum and maximum across all three plots
        ymin = min(axis.get_ylim()[0] for axis in row_axes)
        ymax = max(axis.get_ylim()[1] for axis in row_axes)

        # Apply identical limits to all three plots
        for axis in row_axes:
            axis.set_ylim(ymin, ymax)

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{case_dir.name}.png"

    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ==============================================================
# Main execution
# ==============================================================

def main():

    if not VPMX_DIR.is_dir():
        raise FileNotFoundError(
            f"VPMchi results directory not found: {VPMX_DIR}"
        )

    # Find all simulation folders
    case_dirs = sorted(
        folder
        for folder in VPMX_DIR.glob("V-*_alpha-*_RPMs-*")
        if folder.is_dir()
    )

    print(f"Found {len(case_dirs)} VPMchi simulation folders.")

    # Process each simulation
    for case_dir in case_dirs:

        try:
            process_simulation(case_dir)

        except (ValueError, KeyError, OSError) as error:
            print(f"ERROR processing {case_dir.name}: {error}")

    print("\nFinished processing VPMchi simulations.")


if __name__ == "__main__":
    main()