
"""
Compare VPMchi, FLIGHTLAB, and experimental aircraft loads.

The script:
1. Searches all VPMchi simulation folders in results/VPMx.
2. Searches all FLIGHTLAB simulation folders in results/FLIGHTLAB.
3. Reads experimental aircraft loads from Loads.xlsx.
4. Calculates mean VPMchi loads over the last N_LAST samples.
5. Calculates mean FLIGHTLAB loads over the sampling phase.
6. Matches simulation and experimental operating conditions.
7. Generates five RPM-sweep and six angle-of-attack-sweep figures.
8. Applies common y-axis limits to forces and moments separately.
9. Saves comparison plots to results/load_comparison_plots.

All forces are in N and moments in N m.
"""

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================
# Settings
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VPM_DIR = PROJECT_ROOT / "results" / "VPMx"

FLIGHTLAB_DIR = PROJECT_ROOT / "results" / "FLIGHTLAB"

EXPERIMENT_FILE = PROJECT_ROOT / "results" / "experimental" / "Loads.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "plots" / "load_comparison_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of final VPMchi samples used for averaging
N_LAST = 2000

# Experimental workbook sheet
EXPERIMENT_SHEET = 0

# Figure settings
FIGURE_SIZE = (15, 8)
FIGURE_DPI = 220

# ==============================================================
# Load definitions
# ==============================================================

LOADS = [
    "Fx", "Fy", "Fz", "Mx", "My", "Mz"
]

FL_VARIABLES = [
    "FX", "FY", "FZ", "MX", "MY", "MZ"
]

CSV_COLS = [
    "total_Fx_N",
    "total_Fy_N",
    "total_Fz_N",
    "total_Mx_Nm",
    "total_My_Nm",
    "total_Mz_Nm",
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
# Load sign conventions
# ==============================================================

# Order: Fx, Fy, Fz, Mx, My, Mz
# Experimental and VPMchi Z force and moment are reversed.

EXP_SIGNS = np.array([1, 1, -1, 1, 1, -1])

FLIGHTLAB_SIGNS = np.ones(6)

VPMX_SIGNS = np.array([1, 1, -1, 1, 1, -1])

# ==============================================================
# Matching settings
# ==============================================================

MATCH_ATOL_V = 0.11
MATCH_ATOL_ALPHA = 0.11
MATCH_ATOL_RPM = 1.1

# Missing results are represented by NaN and not plotted
MISSING_VALUE = np.nan

# ==============================================================
# Simulation folder naming convention
# ==============================================================

PATTERN = re.compile(
    r"^V-(?P<v>-?\d+(?:\.\d+)?)"
    r"_alpha-(?P<a>-?\d+(?:\.\d+)?)"
    r"_RPMs-(?P<r1>\d+)-(?P<r4>\d+)-(?P<r5>\d+)$"
)

# ==============================================================
# Operating conditions: RPM sweeps
# ==============================================================

RPM_SWEEPS = [
    dict(
        name="Pusher only",
        v=0.0,
        alpha=0.0,
        rpms=[(0, 0, r) for r in (3000, 4000, 5000)],
        xrotor="r5",
    ),
    dict(
        name="Hover",
        v=0.0,
        alpha=0.0,
        rpms=[(r, r, 0) for r in (3000, 4000, 5000, 6000)],
        xrotor="r1",
    ),
    dict(
        name="Transition 1",
        v=5.1,
        alpha=1.7,
        rpms=[(r, 6169, 2189) for r in (3000, 4000, 5000, 6000)],
        xrotor="r1",
    ),
    dict(
        name="Transition 2",
        v=10.3,
        alpha=3.3,
        rpms=[(r, 5754, 3705) for r in (3000, 4000, 5000, 6000)],
        xrotor="r1",
    ),
    dict(
        name="Cruise",
        v=16.5,
        alpha=5.3,
        rpms=[(0, 0, r) for r in (5500, 6000, 6500, 7000)],
        xrotor="r5",
    ),
]

# ==============================================================
# Operating conditions: angle-of-attack sweeps
# ==============================================================

AOA_SWEEPS = [
    dict(name="Transition 1", v=5.1, rpms=(6700, 6169, 2189)),
    dict(name="Transition 1 (airframe)", v=5.1, rpms=(0, 0, 0)),
    dict(name="Transition 2", v=10.3, rpms=(6700, 5754, 3705)),
    dict(name="Transition 2 (airframe)", v=10.3, rpms=(0, 0, 0)),
    dict(name="Cruise", v=16.5, rpms=(0, 0, 5500)),
    dict(name="Cruise (airframe)", v=16.5, rpms=(0, 0, 0)),
]

ANGLES = [-6.0, 0.0, 6.0]

# ==============================================================
# Plot styles
# ==============================================================

STYLES = {
    "exp": dict(
        label="Experiment",
        color="black",
        marker="s",
        linestyle="-",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="black",
        markeredgecolor="black",
    ),
    "flightlab": dict(
        label="FLIGHTLAB",
        color="black",
        marker="^",
        linestyle=":",
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
    ),
    "vpmx": dict(
        label="VPMχ",
        color="black",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
    ),
}

# ==============================================================
# Extract operating conditions from case folder names
# ==============================================================

def case_fields(folder):
    """Extract velocity, angle of attack, and rotor RPMs."""

    match = PATTERN.fullmatch(folder.name)

    if match is None:
        return None

    values = match.groupdict()

    return dict(
        case=folder.name,
        v=float(values["v"]),
        alpha=float(values["a"]),
        r1=int(values["r1"]),
        r4=int(values["r4"]),
        r5=int(values["r5"]),
    )


# ==============================================================
# Read and average VPMchi simulation results
# ==============================================================

def read_simulations():
    """
    Read VPMchi loading histories and average the final N_LAST samples.
    """

    rows = []

    if not VPM_DIR.is_dir():
        warnings.warn(f"VPMchi directory missing: {VPM_DIR}")
        return pd.DataFrame(rows)

    for folder in sorted(VPM_DIR.iterdir()):

        if not folder.is_dir():
            continue

        fields = case_fields(folder)

        if fields is None:
            continue

        # ------------------------------------------------------
        # Locate loading history file
        # ------------------------------------------------------

        csv_file = (
            folder
            / "out_cuda_run"
            / f"{folder.name}_loading_history.csv"
        )

        if not csv_file.is_file():
            warnings.warn(f"Missing VPMchi CSV: {csv_file}")
            continue

        try:

            # --------------------------------------------------
            # Read loading history
            # --------------------------------------------------

            data = pd.read_csv(csv_file)

            missing = set(CSV_COLS) - set(data.columns)

            if missing:
                raise ValueError(
                    f"Missing columns: {sorted(missing)}"
                )

            # --------------------------------------------------
            # Calculate sampling-phase mean loads
            # --------------------------------------------------

            last = (
                data[CSV_COLS]
                .apply(pd.to_numeric, errors="coerce")
                .tail(N_LAST)
            )

            if len(last) < N_LAST:
                warnings.warn(
                    f"{folder.name}: only {len(last)} VPMchi samples"
                )

            if last.isna().any().any():
                warnings.warn(
                    f"{folder.name}: NaNs in final VPMchi samples"
                )

            mean_loads = last.mean().to_numpy() * VPMX_SIGNS

            rows.append(
                dict(
                    **fields,
                    n_samples=len(last),
                    **dict(zip(LOADS, mean_loads)),
                )
            )

        except Exception as error:
            warnings.warn(
                f"Could not process {csv_file}: {error}"
            )

    print(f"VPMchi: {len(rows)} cases")

    return pd.DataFrame(rows)


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

                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid !B header: {line}"
                    )

                current_variable = parts[1].upper()
                current_values = []
                continue

            # Ignore other FLIGHTLAB metadata lines
            if line.startswith("!"):
                continue

            if current_variable is not None:
                current_values.extend(
                    float(value.replace("D", "E").replace("d", "e"))
                    for value in line.split()
                )

    if current_variable is not None:
        data[current_variable] = np.asarray(
            current_values, dtype=float
        )

    return data


def read_scalar(data, name):
    """Extract a scalar parameter from the TAB data."""

    if name not in data:
        raise KeyError(f"Missing FLIGHTLAB variable: {name}")

    values = data[name]

    if values.size != 1:
        raise ValueError(
            f"Expected scalar '{name}', got {values.size} values."
        )

    return float(values[0])


# ==============================================================
# Calculate FLIGHTLAB sampling-phase mean loads
# ==============================================================

def flightlab_total_means(file_path):
    """
    Calculate mean total-aircraft loads over the sampling phase.
    """

    data = read_tab_file(file_path)

    # ----------------------------------------------------------
    # Simulation information
    # ----------------------------------------------------------

    dt = read_scalar(data, "TS")

    if dt <= 0:
        raise ValueError(f"Invalid time step: {dt}")

    tinit = read_scalar(data, "TINIT")
    tconv = read_scalar(data, "TCONV")
    tsample = read_scalar(data, "TSAMPLE")

    # Reconstruct step counts using FLIGHTLAB's ceil()
    nsteps_init = int(np.ceil(tinit / dt))
    nsteps_conv = int(np.ceil(tconv / dt))
    nsteps_sample = int(np.ceil(tsample / dt))

    nsteps_total = (
        nsteps_init + nsteps_conv + nsteps_sample
    )

    if nsteps_sample <= 0:
        raise ValueError("Sampling interval contains no steps.")

    # Sampling indices
    sample_start = nsteps_init + nsteps_conv

    # ----------------------------------------------------------
    # Extract total-aircraft load histories
    # ----------------------------------------------------------

    mean_loads = []

    for variable in FL_VARIABLES:

        values = data[variable]

        expected_size = 7 * nsteps_total

        if values.size != expected_size:
            raise ValueError(
                f"{variable}: expected {expected_size} values, "
                f"found {values.size}."
            )

        # FLIGHTLAB stores component histories as blocks
        histories = values.reshape(7, nsteps_total)

        # Extract total aircraft and sampling phase
        sample_history = histories[
            6, sample_start:nsteps_total
        ]

        mean_loads.append(np.mean(sample_history))

    return np.asarray(mean_loads, dtype=float), nsteps_sample


# ==============================================================
# Read FLIGHTLAB simulation results
# ==============================================================

def read_flightlab():
    """
    Read all FLIGHTLAB cases directly from results/FLIGHTLAB.
    """

    rows = []

    if not FLIGHTLAB_DIR.is_dir():
        warnings.warn(
            f"FLIGHTLAB directory missing: {FLIGHTLAB_DIR}"
        )
        return pd.DataFrame(rows)

    for folder in sorted(FLIGHTLAB_DIR.iterdir()):

        if not folder.is_dir():
            continue

        fields = case_fields(folder)

        if fields is None:
            continue

        # ------------------------------------------------------
        # Locate loads file
        # ------------------------------------------------------

        loads_file = folder / "loads.tab"

        if not loads_file.is_file():
            warnings.warn(
                f"Missing FLIGHTLAB loads.tab: {loads_file}"
            )
            continue

        try:

            # --------------------------------------------------
            # Calculate sampling-phase mean loads
            # --------------------------------------------------

            means, nsteps_sample = flightlab_total_means(
                loads_file
            )

            means = means * FLIGHTLAB_SIGNS

            rows.append(
                dict(
                    **fields,
                    n_samples=nsteps_sample,
                    **dict(zip(LOADS, means)),
                )
            )

        except Exception as error:
            warnings.warn(
                f"Could not process {loads_file}: {error}"
            )

    print(f"FLIGHTLAB: {len(rows)} cases")

    return pd.DataFrame(rows)


# ==============================================================
# Read experimental load measurements
# ==============================================================

def read_experiments():
    """
    Read experimental aircraft loads from Loads.xlsx.

    Columns:
        C   = Freestream velocity
        D   = RPM1
        E   = RPM4
        F   = RPM5
        G   = Angle of attack
        K:P = Fx, Fy, Fz, Mx, My, Mz
    """

    # ----------------------------------------------------------
    # Read experimental workbook
    # ----------------------------------------------------------

    data = pd.read_excel(
        EXPERIMENT_FILE,
        sheet_name=EXPERIMENT_SHEET,
        header=None,
        usecols="C:G,K:P",
        engine="openpyxl",
    )

    data.columns = [
        "v", "r1", "r4", "r5", "alpha"
    ] + LOADS

    # ----------------------------------------------------------
    # Convert columns to numerical values
    # ----------------------------------------------------------

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Remove rows without valid operating conditions
    data = data.dropna(
        subset=["v", "r1", "r4", "r5", "alpha"]
    ).copy()

    # Apply experimental sign conventions
    data.loc[:, LOADS] = (
        data[LOADS].to_numpy(dtype=float) * EXP_SIGNS
    )

    return data.reset_index(drop=True)


# ==============================================================
# Match operating conditions to simulation and experimental data
# ==============================================================

def lookup(data, v, alpha, rpms):
    """
    Find mean loads for a specified operating condition.
    """

    if data.empty:
        return None

    # ----------------------------------------------------------
    # Match velocity and angle of attack
    # ----------------------------------------------------------

    mask = (
        np.isclose(data["v"], v, atol=MATCH_ATOL_V)
        & np.isclose(
            data["alpha"], alpha, atol=MATCH_ATOL_ALPHA
        )
    )

    # ----------------------------------------------------------
    # Match rotor RPMs
    # ----------------------------------------------------------

    for column, rpm in zip(
        ("r1", "r4", "r5"), rpms
    ):
        mask &= np.isclose(
            data[column], rpm, atol=MATCH_ATOL_RPM
        )

    matches = data.loc[mask]

    if matches.empty:
        return None

    if len(matches) > 1:
        warnings.warn(
            f"Multiple matches for V={v}, "
            f"alpha={alpha}, RPMs={rpms}; averaging."
        )

    return matches[LOADS].mean().to_numpy(dtype=float)


# ==============================================================
# Build RPM and angle-of-attack sweep datasets
# ==============================================================

def collect_series(sources):
    """
    Collect load data for all predefined RPM and AoA sweeps.
    """

    series = []

    # ----------------------------------------------------------
    # RPM sweeps
    # ----------------------------------------------------------

    for spec in RPM_SWEEPS:

        # Horizontal axis: selected rotor RPM
        x = [
            dict(zip(("r1", "r4", "r5"), rpm))[spec["xrotor"]]
            for rpm in spec["rpms"]
        ]

        xlabel = (
            "RPM (R1 & R4)"
            if spec["name"] == "Hover"
            else f"RPM ({spec['xrotor'].upper()})"
        )

        values = {
            key: [
                lookup(
                    data,
                    spec["v"],
                    spec["alpha"],
                    rpm,
                )
                for rpm in spec["rpms"]
            ]
            for key, data in sources.items()
        }

        series.append(
            dict(
                kind="RPM",
                name=spec["name"],
                x=np.asarray(x),
                xlabel=xlabel,
                values=values,
                v=spec["v"],
                alpha=spec["alpha"],
                rpms=spec["rpms"],
            )
        )

    # ----------------------------------------------------------
    # Angle-of-attack sweeps
    # ----------------------------------------------------------

    for spec in AOA_SWEEPS:

        values = {
            key: [
                lookup(
                    data,
                    spec["v"],
                    alpha,
                    spec["rpms"],
                )
                for alpha in ANGLES
            ]
            for key, data in sources.items()
        }

        series.append(
            dict(
                kind="AoA",
                name=spec["name"],
                x=np.asarray(ANGLES),
                xlabel="Angle of attack [deg]",
                values=values,
                v=spec["v"],
                rpms=[spec["rpms"]] * len(ANGLES),
            )
        )

    return series


# ==============================================================
# Convert missing data points into plotting arrays
# ==============================================================

def array_for(values):
    """
    Replace missing load vectors with NaN values.
    """

    return np.asarray(
        [
            np.full(6, MISSING_VALUE)
            if value is None else value
            for value in values
        ],
        dtype=float,
    )


# ==============================================================
# Calculate consistent vertical limits for all figures
# ==============================================================

def global_limits(series):
    """
    Use one common force range and one common moment range
    across all RPM and angle-of-attack sweep figures.
    """

    limits = []

    # ----------------------------------------------------------
    # Calculate force and moment ranges separately
    # ----------------------------------------------------------

    for component_indices in (
        range(3),
        range(3, 6),
    ):

        values = np.concatenate(
            [
                array_for(load_values)[:, j]
                for sweep in series
                for load_values in sweep["values"].values()
                for j in component_indices
            ]
        )

        # Exclude missing and non-finite values
        values = values[np.isfinite(values)]

        if len(values) == 0:
            common_limit = (-1, 1)

        else:

            ymin = float(values.min())
            ymax = float(values.max())

            # Add padding around the data
            margin = max(
                0.08 * (ymax - ymin),
                0.05 * max(abs(ymin), abs(ymax)),
                0.001,
            )

            common_limit = (
                ymin - margin,
                ymax + margin,
            )

        # Apply identical limits to all three components
        limits.extend([common_limit] * 3)

    return limits


# ==============================================================
# Generate output filename
# ==============================================================

def output_filename(sweep):
    """
    Generate a filename containing operating conditions.

    RPM sweep:
        RPM_Transition_1_V-5.10_alpha-1.70.png

    AoA sweep:
        AoA_Transition_1_V-5.10_RPMs-6700-6169-2189.png
    """

    # Replace spaces and special characters in phase name
    name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        sweep["name"],
    ).strip("_")

    # ----------------------------------------------------------
    # RPM sweep filename
    # ----------------------------------------------------------

    if sweep["kind"] == "RPM":

        return (
            f"RPM_{name}"
            f"_V-{sweep['v']:.2f}"
            f"_alpha-{sweep['alpha']:.2f}.png"
        )

    # ----------------------------------------------------------
    # Angle-of-attack sweep filename
    # ----------------------------------------------------------

    rpms = sweep["rpms"][0]

    return (
        f"AoA_{name}"
        f"_V-{sweep['v']:.2f}"
        f"_RPMs-{rpms[0]}-{rpms[1]}-{rpms[2]}.png"
    )


# ==============================================================
# Plot one sweep with six load-component subplots
# ==============================================================

def plot_one(sweep, limits, destination):
    """
    Plot forces and moments for one RPM or angle-of-attack sweep.
    """

    # ----------------------------------------------------------
    # Create figure
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        2, 3,
        figsize=FIGURE_SIZE,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # Plot individual load components
    # ----------------------------------------------------------

    for j, axis in enumerate(axes.flat):

        for source, style in STYLES.items():

            values = array_for(
                sweep["values"][source]
            )

            history = values[:, j]

            # Skip if all data points are zero or NaN
            if np.all(
                np.isnan(history) | (history == 0)
            ):
                continue

            axis.plot(
                sweep["x"],
                history,
                **style,
            )

        # ------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------

        axis.set_ylabel(LOAD_LABELS[j])
        axis.set_xlabel(sweep["xlabel"])
        axis.set_ylim(limits[j])
        axis.set_xticks(sweep["x"])

        axis.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # Figure legend
    # ----------------------------------------------------------

    # Collect legend entries from all subplots so that a source
    # is included even if it is absent from the first subplot.
    handles = {}

    for axis in axes.flat:

        axis_handles, axis_labels = (
            axis.get_legend_handles_labels()
        )

        for handle, label in zip(
            axis_handles, axis_labels
        ):
            if label not in handles:
                handles[label] = handle

    if handles:

        fig.legend(
            list(handles.values()),
            list(handles.keys()),
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------

    fig.savefig(
        destination,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================
# Main execution
# ==============================================================

def main():

    if not EXPERIMENT_FILE.is_file():
        raise FileNotFoundError(
            f"Experimental workbook not found: {EXPERIMENT_FILE}"
        )

    # ----------------------------------------------------------
    # Read all datasets
    # ----------------------------------------------------------

    sources = {
        "exp": read_experiments(),
        "flightlab": read_flightlab(),
        "vpmx": read_simulations(),
    }

    for source, data in sources.items():
        print(
            f"{STYLES[source]['label']}: {len(data)} cases/rows"
        )

    # ----------------------------------------------------------
    # Collect all sweep datasets
    # ----------------------------------------------------------

    series = collect_series(sources)

    # Calculate common y-axis limits
    limits = global_limits(series)

    # ----------------------------------------------------------
    # Generate and save comparison plots
    # ----------------------------------------------------------

    missing = []

    for i, sweep in enumerate(series, 1):

        filename = output_filename(sweep)

        destination = OUTPUT_DIR / filename

        plot_one(
            sweep,
            limits,
            destination,
        )

        print(
            f"[{i}/{len(series)}] Saved: {filename}"
        )

        # ------------------------------------------------------
        # Report missing simulation cases
        # ------------------------------------------------------

        for k, x in enumerate(sweep["x"]):

            for source, values in sweep["values"].items():

                if values[k] is None:

                    missing.append(
                        dict(
                            sweep=sweep["name"],
                            source=STYLES[source]["label"],
                            x=x,
                        )
                    )

                    print(
                        f"  Missing {STYLES[source]['label']}: "
                        f"{sweep['name']}, x={x:g}"
                    )

    # ----------------------------------------------------------
    # Print final summary
    # ----------------------------------------------------------

    print("\nFinished processing load comparisons.")
    print(f"Figures saved : {len(series)}")
    print(f"Missing cases : {len(missing)}")
    print(f"Output folder : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()