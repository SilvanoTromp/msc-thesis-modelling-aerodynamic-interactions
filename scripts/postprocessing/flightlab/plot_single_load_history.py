
"""
Plot FLIGHTLAB aircraft load histories and calculate mean loads.

The script:
1. Reads a FLIGHTLAB loads.tab output file.
2. Extracts the force and moment histories.
3. Separates the histories into the individual aircraft components:
   - Forward hover rotor
   - Aft hover rotor
   - Pusher propeller
   - Wing
   - Body
   - Total aircraft
4. Plots Fx, Fy, Fz, Mx, My, and Mz as functions of simulation time.
5. Calculates the mean total-aircraft loads over the sampling interval.
6. Displays the mean total-aircraft load as a horizontal line.

The script is intended for checking load convergence and extracting
time-averaged loads from FLIGHTLAB simulations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================
# Settings
# ==============================================================

# FLIGHTLAB loads file
LOADS_FILE = Path(
    "S:/EVO/WT/results/V-10.30_alpha-6_RPMs-6700-5754-3705/loads.tab"
)


# ==============================================================
# Load components
# ==============================================================

LOAD_NAMES = [
    "Forward hover rotor",
    "Aft hover rotor",
    "Pusher propeller",
    "Wing",
    "Body",
    "Total aircraft",
]

LOAD_VARIABLES = [
    "FX",
    "FY",
    "FZ",
    "MX",
    "MY",
    "MZ",
]


# ==============================================================
# FLIGHTLAB TAB reader
# ==============================================================

def read_tab_file(file_path):
    """
    Read all data blocks from a FLIGHTLAB .tab file.

    Each block starts with a line of the form:

        !B VARIABLE NROWS NCOLUMNS

    Parameters
    ----------
    file_path : Path
        Path to the FLIGHTLAB .tab file.

    Returns
    -------
    dict
        Dictionary containing the numerical data for each variable.
    """

    data = {}
    current_variable = None
    current_values = []

    with open(file_path, "r") as file:

        for line in file:

            line = line.strip()

            # Ignore empty lines and header comments
            if not line or line.startswith("#"):
                continue

            # Start of a new data block
            if line.startswith("!B"):

                if current_variable is not None:
                    data[current_variable] = np.asarray(
                        current_values,
                        dtype=float,
                    )

                parts = line.split()

                current_variable = parts[1]
                current_values = []

                continue

            # Read numerical values
            if current_variable is not None:

                for value in line.split():
                    current_values.append(float(value))

    # Store final block
    if current_variable is not None:
        data[current_variable] = np.asarray(
            current_values,
            dtype=float,
        )

    return data


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

# Simulation durations and time step [s]
tinit = float(data["TINIT"][0])
tsample = float(data["TSAMPLE"][0])
dt = float(data["DT"][0])

if dt <= 0 or tinit < 0 or tsample <= 0:
    raise ValueError(
        "Invalid simulation settings: DT must be positive, "
        "TINIT must be nonnegative, and TSAMPLE must be positive."
    )

# Determine actual number of stored time steps
nsteps_total = len(data["FX"]) // len(LOAD_NAMES)

# Reconstruct initialization and sampling step counts
nsteps_init = int(np.ceil(tinit / dt))
nsteps_sample = nsteps_total - nsteps_init

if nsteps_sample <= 0:
    raise ValueError(
        "No sampling steps are available in the load history."
    )

# Simulation time corresponding to each stored load
time = np.arange(1, nsteps_total + 1) * dt

# Start of sampling interval
sampling_start_time = nsteps_init * dt

print("\nSimulation information")
print("----------------------")
print(f"Initialization duration : {tinit:.6f} s")
print(f"Sampling duration       : {tsample:.6f} s")
print(f"Time step               : {dt:.8f} s")
print(f"Initialization steps    : {nsteps_init}")
print(f"Sampling steps          : {nsteps_sample}")
print(f"Total steps             : {nsteps_total}")
print(f"Total simulated time    : {time[-1]:.6f} s")


# ==============================================================
# Separate component load histories
# ==============================================================

load_histories = {}

for variable in LOAD_VARIABLES:

    values = data[variable]

    # Load columns:
    #   1 = Forward hover rotor
    #   2 = Aft hover rotor
    #   3 = Pusher propeller
    #   4 = Wing
    #   5 = Body
    #   6 = Total aircraft

    expected_size = len(LOAD_NAMES) * nsteps_total

    if len(values) != expected_size:
        raise ValueError(
            f"{variable} contains {len(values)} values, while "
            f"{expected_size} values were expected."
        )

    # FLIGHTLAB stores each component history as a block
    load_histories[variable] = np.reshape(
        values,
        (len(LOAD_NAMES), nsteps_total),
    )


# ==============================================================
# Define averaging interval
# ==============================================================

# Average over the complete sampling phase
average_start_step = nsteps_init

if average_start_step >= nsteps_total:
    raise ValueError(
        "The averaging interval contains no simulation steps."
    )


# ==============================================================
# Calculate mean total-aircraft loads
# ==============================================================

mean_total_loads = {}

for variable in LOAD_VARIABLES:

    # Index 5 corresponds to total aircraft
    total_history = load_histories[variable][5]

    mean_total_loads[variable] = np.mean(
        total_history[average_start_step:]
    )


# ==============================================================
# Print mean loads
# ==============================================================

print(
    f"\nMean total-aircraft loads over final "
    f"{nsteps_sample * dt:.6f} s"
)
print("----------------------------------------------")

print(f"Fx = {mean_total_loads['FX']: .6f} N")
print(f"Fy = {mean_total_loads['FY']: .6f} N")
print(f"Fz = {mean_total_loads['FZ']: .6f} N")

print(f"Mx = {mean_total_loads['MX']: .6f} N m")
print(f"My = {mean_total_loads['MY']: .6f} N m")
print(f"Mz = {mean_total_loads['MZ']: .6f} N m")



# ==============================================================
# Plot total-aircraft load histories
# ==============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8),
    sharex=True,
)

axes = axes.flatten()

ylabels = [
    r"$F_x$ [N]",
    r"$F_y$ [N]",
    r"$F_z$[N]",
    r"$M_x$ [N m]",
    r"$M_y$ [N m]",
    r"$M_z$ [N m]",
]

# Common axis limits
FORCE_LIMITS = (-30, 50)
MOMENT_LIMITS = (-20, 40)

for i, (axis, variable, ylabel) in enumerate(
    zip(axes, LOAD_VARIABLES, ylabels)
):

    # Extract total-aircraft loads (column 6)
    total_history = load_histories[variable][5]

    # Plot total-aircraft load history
    axis.plot(
        time,
        total_history,
        linewidth=1.2,
        color="black",
    )

    # Apply common axis limits
    if i < 3:
        axis.set_ylim(FORCE_LIMITS)
    else:
        axis.set_ylim(MOMENT_LIMITS)

    axis.set_ylabel(ylabel)
    axis.grid(True)

    # X-axis labels for bottom row
    if i >= 3:
        axis.set_xlabel("Simulation time [s]")


# ==============================================================
# Figure formatting
# ==============================================================

fig.tight_layout()

plt.show()