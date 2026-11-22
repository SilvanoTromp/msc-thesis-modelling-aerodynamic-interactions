
"""
Plot FLIGHTLAB load convergence plots for all simulation cases.

The script:
1. Searches all simulation folders in results/flightlab.
2. Reads the loads.tab file from each simulation folder.
3. Extracts total-aircraft force and moment histories.
4. Identifies initialization, convergence, and sampling phases.
5. Calculates sampling-phase mean loads and standard deviations.
6. Plots Fx, Fy, Fz, Mx, My, and Mz versus simulation time.
7. Displays sampling-phase mean loads as horizontal lines.
8. Saves convergence plots to results/convergence_plots/flightlab.

All forces are in N and moments in N m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# Settings
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

FLIGHTLAB_DIR = PROJECT_ROOT / "results" / "FLIGHTLAB"

OUTPUT_DIR = PROJECT_ROOT / "plots" / "convergence_plots" / "FLIGHTLAB"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
FIGURE_SIZE = (15, 8)
FIGURE_DPI = 300

# ==============================================================
# Load definitions
# ==============================================================

LOAD_VARIABLES = [
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

N_COMPONENTS = 7
TOTAL_INDEX = 6

# ==============================================================
# FLIGHTLAB TAB reader
# ==============================================================

def read_tab_file(file_path):
    """
    Read numerical blocks from a FLIGHTLAB .tab file.

    Each block starts with:

        !B VARIABLE NROWS NCOLUMNS

    Returns
    -------
    dict
        Numerical arrays indexed by variable name.
    """

    data = {}
    current_variable = None
    current_values = []

    with open(file_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("!B"):

                if current_variable is not None:
                    data[current_variable] = np.asarray(
                        current_values, dtype=float
                    )

                parts = line.split()
                current_variable = parts[1].upper()
                current_values = []
                continue

            # Ignore other FLIGHTLAB metadata lines
            if line.startswith("!"):
                continue

            if current_variable is not None:
                current_values.extend(
                    float(value) for value in line.split()
                )

    if current_variable is not None:
        data[current_variable] = np.asarray(
            current_values, dtype=float
        )

    return data

def read_scalar(data, name):
    """Extract a scalar parameter from the TAB data."""

    if name not in data:
        raise KeyError(
            f"Missing variable '{name}' in loads.tab. "
            f"Available variables: {list(data)}"
        )

    values = data[name]

    if values.size != 1:
        raise ValueError(
            f"Expected scalar '{name}', got {values.size} values."
        )

    return float(values[0])

# ==============================================================
# Process individual simulation
# ==============================================================

def process_simulation(case_dir):
    """
    Read FLIGHTLAB loads and save total-aircraft convergence plot.
    """

    loads_file = case_dir / "loads.tab"

    if not loads_file.is_file():
        print(f"Skipping {case_dir.name}: loads.tab not found.")
        return

    print(f"\nProcessing: {case_dir.name}")

    # ----------------------------------------------------------
    # Read loads file
    # ----------------------------------------------------------

    data = read_tab_file(loads_file)

    # ----------------------------------------------------------
    # Simulation information
    # ----------------------------------------------------------

    # Simulation durations [s]
    tinit = read_scalar(data, "TINIT")
    tconv = read_scalar(data, "TCONV")
    tsample = read_scalar(data, "TSAMPLE")

    # Simulation time step [s]
    dt = read_scalar(data, "TS")

    # Reconstruct step counts using FLIGHTLAB's ceil()
    nsteps_init = int(np.ceil(tinit / dt))
    nsteps_conv = int(np.ceil(tconv / dt))
    nsteps_sample = int(np.ceil(tsample / dt))

    nsteps_expected = (
        nsteps_init + nsteps_conv + nsteps_sample
    )

    # Number of steps actually stored
    nsteps_total = data["FX"].size // N_COMPONENTS

    if nsteps_total != nsteps_expected:
        raise ValueError(
            f"Step-count mismatch: loads.tab contains "
            f"{nsteps_total} steps, but {nsteps_expected} "
            f"were expected."
        )

    # Reconstruct simulation time [s]
    time = np.arange(1, nsteps_total + 1) * dt

    # Simulation phase boundaries
    t_init_end = nsteps_init * dt
    t_conv_end = t_init_end + nsteps_conv * dt
    t_total = nsteps_total * dt

    # Sampling indices
    sample_start = nsteps_init + nsteps_conv
    sample_slice = slice(sample_start, nsteps_total)

    # ----------------------------------------------------------
    # Extract total-aircraft load histories
    # ----------------------------------------------------------

    load_histories = {}
    mean_loads = {}
    std_loads = {}

    for variable in LOAD_VARIABLES:

        if variable not in data:
            raise KeyError(f"Missing load variable: {variable}")

        values = data[variable]

        expected_size = N_COMPONENTS * nsteps_total

        if values.size != expected_size:
            raise ValueError(
                f"{variable}: expected {expected_size} values, "
                f"found {values.size}."
            )

        # FLIGHTLAB stores component histories as blocks
        histories = values.reshape(
            N_COMPONENTS, nsteps_total
        )

        # Extract total aircraft only
        history = histories[TOTAL_INDEX].copy()

        load_histories[variable] = history

        # Calculate sampling-phase statistics
        sample_history = history[sample_slice]

        mean_loads[variable] = np.mean(sample_history)
        std_loads[variable] = np.std(sample_history)

    # ----------------------------------------------------------
    # Print mean total-aircraft loads
    # ----------------------------------------------------------

    print("Mean total-aircraft loads:")
    print("-" * 50)

    for variable in LOAD_VARIABLES:

        mean = mean_loads[variable]
        std = std_loads[variable]

        unit = "N" if variable.startswith("F") else "N m"

        print(
            f"{variable:2s} = {mean:11.6f} "
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

    for i, (axis, variable, ylabel) in enumerate(
        zip(axes, LOAD_VARIABLES, LOAD_LABELS)
    ):

        # ------------------------------------------------------
        # Load history
        # ------------------------------------------------------

        history = load_histories[variable]

        axis.plot(
            time,
            history,
            color="black",
            linewidth=1.2,
        )

        # ------------------------------------------------------
        # Simulation phase boundaries
        # ------------------------------------------------------

        axis.axvline(
            t_init_end,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        axis.axvline(
            t_conv_end,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        # ------------------------------------------------------
        # Sampling-phase mean load
        # ------------------------------------------------------

        mean_value = mean_loads[variable]

        axis.hlines(
            mean_value,
            xmin=t_conv_end,
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
        axis.set_xlim(0, t_total)

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

    if not FLIGHTLAB_DIR.is_dir():
        raise FileNotFoundError(
            f"FLIGHTLAB results directory not found: {FLIGHTLAB_DIR}"
        )

    # Find all simulation folders
    case_dirs = sorted(
        folder
        for folder in FLIGHTLAB_DIR.glob("V-*_alpha-*_RPMs-*")
        if folder.is_dir()
    )

    print(f"Found {len(case_dirs)} FLIGHTLAB simulation folders.")

    # Process each simulation
    for case_dir in case_dirs:

        try:
            process_simulation(case_dir)

        except (ValueError, KeyError, OSError) as error:
            print(f"ERROR processing {case_dir.name}: {error}")

    print("\nFinished processing FLIGHTLAB simulations.")

if __name__ == "__main__":
    main()