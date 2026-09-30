"""
Plot propeller thrust and torque validation data.

Two figures are created:
    1. Hover validation at V = 0 m/s
    2. Cruise validation at approximately V = 16.5 m/s

Each figure contains:
    - Thrust versus RPM
    - Torque versus RPM

The propeller names are intentionally not included in the figure titles so
that they can be added separately in the LaTeX/Overleaf document.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Plot settings
# =============================================================================

OUTPUT_DIRECTORY = Path("validation_plots")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "png"
FIGURE_DPI = 300
SHOW_FIGURES = True

LINE_WIDTH = 1.5
MARKER_SIZE = 5


# =============================================================================
# Validation data
# =============================================================================

# Hover condition: V = 0 m/s
hover_rpm = np.array([3000, 4000, 5000, 6000])

hover_thrust = {
    "APC Performance Data": np.array([2.773, 4.943, 7.745, 11.189]),
    "BEMT": np.array([2.593, 4.726, 7.622, 11.299]),
    "LOPNOR_BEMT": np.array([2.848, 4.969, 7.850, 11.509]),
}

hover_torque = {
    "APC Performance Data": np.array([0.050, 0.088, 0.135, 0.193]),
    "BEMT": np.array([0.046, 0.082, 0.128, 0.186]),
    "LOPNOR_BEMT": np.array([0.045, 0.081, 0.128, 0.186]),
}


# Cruise condition: V ≈ 16.5 m/s
cruise_rpm = np.array([5000, 6000, 7000, 8000])
cruise_velocity = np.array([16.102, 16.107, 16.290, 17.180])

cruise_thrust = {
    "APC Performance Data": np.array([9.666, 17.288, 26.267, 36.115]),
    "BEMT": np.array([9.799, 17.995, 27.629, 38.247]),
    "LOPNOR_BEMT": np.array([9.815, 17.882, 27.386, 37.867]),
}

cruise_torque = {
    "APC Performance Data": np.array([0.402, 0.638, 0.899, 1.188]),
    "BEMT": np.array([0.399, 0.649, 0.921, 1.226]),
    "LOPNOR_BEMT": np.array([0.392, 0.643, 0.917, 1.223]),
}


# =============================================================================
# Line formatting
# =============================================================================

line_styles = {
    "APC Performance Data": {
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "black",
    },
    "BEMT": {
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": "white",
    },
    "LOPNOR_BEMT": {
        "linestyle": ":",
        "marker": "^",
        "markerfacecolor": "white",
    },
}


# =============================================================================
# Plotting function
# =============================================================================

def plot_validation(
    rpm,
    thrust_data,
    torque_data,
    output_filename,
):
    """
    Create a side-by-side thrust and torque validation figure.

    Parameters
    ----------
    rpm : array-like
        Propeller rotational speeds [RPM].
    thrust_data : dict
        Thrust datasets in newtons.
    torque_data : dict
        Torque datasets in newton-metres.
    output_filename : str
        Name of the saved figure without the file extension.
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9, 4),
    )

    ax_thrust, ax_torque = axes

    # ---------------------------------------------------------------------
    # Thrust
    # ---------------------------------------------------------------------

    for label, values in thrust_data.items():
        style = line_styles[label]

        ax_thrust.plot(
            rpm,
            values,
            color="black",
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor="black",
            label=label,
        )

    ax_thrust.set_xlabel("Rotational speed [RPM]")
    ax_thrust.set_ylabel("Thrust [N]")
    ax_thrust.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax_thrust.tick_params(direction="in", top=True, right=True)

    # ---------------------------------------------------------------------
    # Torque
    # ---------------------------------------------------------------------

    for label, values in torque_data.items():
        style = line_styles[label]

        ax_torque.plot(
            rpm,
            values,
            color="black",
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor="black",
            label=label,
        )

    ax_torque.set_xlabel("Rotational speed [RPM]")
    ax_torque.set_ylabel(r"Torque [N$\cdot$m]")
    ax_torque.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax_torque.tick_params(direction="in", top=True, right=True)

    # Shared legend
    handles, labels = ax_thrust.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        frameon=False,
    )

    # Reserve space at the bottom for the shared legend
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.97,
        bottom=0.22,
        wspace=0.18,
    )

    output_path = OUTPUT_DIRECTORY / f"{output_filename}.{FIGURE_FORMAT}"

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    print(f"Saved figure to: {output_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Create the hover and cruise validation figures."""

    plot_validation(
        rpm=hover_rpm,
        thrust_data=hover_thrust,
        torque_data=hover_torque,
        output_filename="validation_hover",
    )

    plot_validation(
        rpm=cruise_rpm,
        thrust_data=cruise_thrust,
        torque_data=cruise_torque,
        output_filename="validation_cruise",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()