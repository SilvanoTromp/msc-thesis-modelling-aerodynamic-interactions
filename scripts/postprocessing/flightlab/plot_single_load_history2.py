
"""
Plot FLIGHTLAB aircraft load histories and calculate mean loads.

The script:
1. Reads a FLIGHTLAB loads.tab output file.
2. Extracts force and moment histories for six aircraft components.
3. Identifies initialization, convergence, and sampling phases.
4. Plots Fx, Fy, Fz, Mx, My, and Mz versus simulation time.
5. Calculates mean and standard deviation during sampling.
6. Displays sampling-phase mean loads as horizontal lines.
7. Optionally plots individual component load histories.

All forces are in N and moments in N m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# Settings
# ==============================================================

LOADS_FILE = Path(
    "S:/EVO/WT/results/final/"
    "V-5.10_alpha-1.70_RPMs-3000-6169-2189/loads.tab"
)

# ==============================================================
# Load definitions
# ==============================================================

LOAD_NAMES = [
    "Forward hover rotor",
    "Aft hover rotor",
    "Pusher propeller",
    "Wing",
    "Body",
    "Fuselage",
    "Total aircraft",
]

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

N_COMPONENTS = len(LOAD_NAMES)
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
# Read loads file
# ==============================================================

if not LOADS_FILE.is_file():
    raise FileNotFoundError(
        f"FLIGHTLAB loads file not found: {LOADS_FILE}"
    )

data = read_tab_file(LOADS_FILE)

# ==============================================================
# Simulation information
# ==============================================================

# Simulation durations [s]
tinit = read_scalar(data, "TINIT")
tconv = read_scalar(data, "TCONV")
tsample = read_scalar(data, "TSAMPLE")

# Simulation time step [s]
dt = read_scalar(data, "TS")

# Reconstruct the step counts using FLIGHTLAB's ceil()
nsteps_init = int(np.ceil(tinit / dt))
nsteps_conv = int(np.ceil(tconv / dt))
nsteps_sample = int(np.ceil(tsample / dt))

nsteps_expected = nsteps_init + nsteps_conv + nsteps_sample

# Number of steps actually stored
nsteps_total = data["FX"].size // N_COMPONENTS

if nsteps_total != nsteps_expected:
    raise ValueError(
        f"Step-count mismatch: loads.tab contains "
        f"{nsteps_total} steps, but {nsteps_expected} "
        f"were expected from TINIT, TCONV, TSAMPLE and TS."
    )

# Reconstruct simulation time [s]
time = np.arange(1, nsteps_total + 1) * dt

# Phase boundaries using actual step counts
t_init_end = nsteps_init * dt
t_conv_end = t_init_end + nsteps_conv * dt
t_total = nsteps_total * dt

# Sampling indices
sample_start = nsteps_init + nsteps_conv
sample_slice = slice(sample_start, nsteps_total)

print("\nSimulation information")
print("-" * 50)
print(f"Initialization duration : {tinit:.6f} s")
print(f"Converging duration     : {tconv:.6} s")
print(f"Sampling duration       : {tsample:.6f} s")
print(f"Time step               : {dt:.8f} s")
print(f"Initialization steps    : {nsteps_init}")
print(f"Converging steps        : {nsteps_conv}")
print(f"Sampling steps          : {nsteps_sample}")
print(f"Total steps             : {nsteps_total}")
print(f"Total simulated time    : {t_total:.6f} s")

# ==============================================================
# Extract component load histories
# ==============================================================

load_histories = {}

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

    # FLIGHTLAB stores each component history as a block
    load_histories[variable] = values.reshape(
        N_COMPONENTS, nsteps_total
    ).copy()

# ==============================================================
# Calculate sampling-phase statistics
# ==============================================================

mean_loads = {}
std_loads = {}

for variable in LOAD_VARIABLES:

    # Statistics for all six aircraft components
    histories = load_histories[variable][:, sample_slice]

    mean_loads[variable] = np.mean(
        histories, axis=1
    )

    std_loads[variable] = np.std(
        histories, axis=1
    )

# ==============================================================
# Print mean total-aircraft loads
# ==============================================================

print(
    f"\nMean total-aircraft loads over sampling interval "
    f"[{t_conv_end:.6f}, {t_total:.6f}] s"
)
print("-" * 50)

for variable in LOAD_VARIABLES:

    mean = mean_loads[variable][TOTAL_INDEX]
    std = std_loads[variable][TOTAL_INDEX]

    unit = "N" if variable.startswith("F") else "N m"

    print(
        f"{variable:2s} = {mean: 11.6f} "
        f"+/- {std:9.6f} {unit}"
    )

# ==============================================================
# Plot load histories for all components
# ==============================================================

# Plot total aircraft first, followed by five components
plot_order = [TOTAL_INDEX, 0, 1, 2, 3, 4, 5]

for component_index in plot_order:

    component_name = LOAD_NAMES[component_index]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8),
        sharex=True,
    )

    axes = axes.flatten()

    for i, (axis, variable, ylabel) in enumerate(
        zip(axes, LOAD_VARIABLES, LOAD_LABELS)
    ):

        # ------------------------------------------------------
        # Extract component load history
        # ------------------------------------------------------

        history = load_histories[variable][component_index]

        axis.plot(
            time,
            history,
            color="black",
            linewidth=1.2,
            label=component_name,
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

        mean_value = mean_loads[variable][component_index]
        std_value = std_loads[variable][component_index]

        axis.hlines(
            mean_value,
            xmin=t_conv_end,
            xmax=t_total,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:.3f}",
        )

        # ------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------

        axis.set_ylabel(ylabel)
        axis.set_xlim(0, t_total)
        axis.set_ylim(-20,50)

        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="best")

        if i >= 3:
            axis.set_xlabel("Simulation time [s]")

    # ----------------------------------------------------------
    # Figure formatting
    # ----------------------------------------------------------

    fig.suptitle(
        f"FLIGHTLAB Load Histories - {component_name}",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

# ==============================================================
# Display figures
# ==============================================================

plt.show()
