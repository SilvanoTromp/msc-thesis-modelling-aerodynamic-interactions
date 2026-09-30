"""
plot_extrapolated_airfoil_polars.py

Interactively plot Viterna-extrapolated airfoil polar files.

The script:
    1. Lets the user select a propeller with extrapolated airfoil polars.
    2. Reads all Airfoil_rR*_Re*.txt files from the airfoil_polars_extrapolated folder.
    3. Sorts the available data by blade station r/R and Reynolds number.
    4. Opens an interactive Matplotlib figure with three subplots.
    5. Uses sliders to move through blade stations and Reynolds numbers.
    6. Shows Cl, Cd, and Cm as functions of angle of attack.
    7. Splits each polar into an estimated original-data region and an
       extrapolated region based on detected stall points.

Reads:
    <propeller>/airfoil_polars_extrapolated/Airfoil_rR*_Re*.txt

Shows:
    - Cl(alpha)
    - Cd(alpha)
    - Cm(alpha)

Use the sliders to move through:
    - blade stations r/R
    - Reynolds numbers

The estimated original XFOIL region is shown with solid lines.
The Viterna-extrapolated region is shown with dashed lines.
"""

import sys
from pathlib import Path
utilities_dir = Path(__file__).resolve().parent.parent.parent / "src" / "propeller_data"
sys.path.insert(0, str(utilities_dir))
from utilities import *


# =============================================================================
# Input settings
# =============================================================================

# Directory containing the APC <propeller>-PERF.PE0 geometry files
APC_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "external"
    / "apc"
    / "geometry"
)

# Directory containing the propeller subfolders
PROP_DIRECTORY = Path(
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed_propellers"
)


# =============================================================================
# Plot utilities
# =============================================================================

def split_original_and_extrapolated(alpha, cl, cd, cm, stall):
    """
    Split one polar into estimated original and extrapolated regions.

    Input arrays:
        alpha               : angle of attack values [deg]
        cl                  : lift coefficients [-]
        cd                  : drag coefficients [-]
        cm                  : pitching moment coefficients [-]

    Input dictionary:
        stall               : detected positive and negative stall points

    Required stall entries:
        alpha_neg           : detected negative stall angle [deg]
        alpha_pos           : detected positive stall angle [deg]

    Returned dictionaries:
        original            : polar data between negative and positive stall
        extrapolated        : polar data outside the stall limits

    The region between alpha_neg and alpha_pos is treated as the estimated
    original XFOIL-data region. The regions below negative stall and above
    positive stall are treated as Viterna-extrapolated regions.

    Values outside each region are replaced with NaN. This keeps all arrays the
    same length and allows Matplotlib to draw separated solid and dashed parts
    without changing the alpha grid.
    """

    # Select the estimated original XFOIL-data region
    idx_neg = np.searchsorted(alpha, stall["alpha_neg"], side="left")
    idx_pos = np.searchsorted(alpha, stall["alpha_pos"], side="right") - 1
    original_mask = np.zeros_like(alpha, dtype=bool)
    original_mask[idx_neg:idx_pos + 1] = True

    # Everything outside the estimated original region is treated as extrapolated
    extrapolated_mask = ~original_mask

    # Include one overlapping point
    if idx_neg > 0:
        extrapolated_mask[idx_neg] = True

    if idx_pos < len(alpha) - 1:
        extrapolated_mask[idx_pos] = True

    # Keep only the estimated original region and mask the rest with NaN
    original = {
        "alpha": np.where(original_mask, alpha, np.nan),
        "Cl": np.where(original_mask, cl, np.nan),
        "Cd": np.where(original_mask, cd, np.nan),
        "Cm": np.where(original_mask, cm, np.nan),
    }

    # Keep only the extrapolated region and mask the rest with NaN
    extrapolated = {
        "alpha": np.where(extrapolated_mask, alpha, np.nan),
        "Cl": np.where(extrapolated_mask, cl, np.nan),
        "Cd": np.where(extrapolated_mask, cd, np.nan),
        "Cm": np.where(extrapolated_mask, cm, np.nan),
    }

    return original, extrapolated


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the interactive extrapolated-polar plotting workflow.
    """

    # Ask the user to select a propeller that has extrapolated airfoil polars
    propeller_name, polar_dir = ask_for_propeller_with_extrapolated_polars(
        APC_DIRECTORY,
        PROP_DIRECTORY,
    )

    # Find all extrapolated polar files for the selected propeller
    polar_files = get_polar_files(polar_dir)

    if len(polar_files) == 0:
        raise ValueError(f"No polar files found in {polar_dir.resolve()}")

    # Store all polar data before opening the figure
    # Key format:
    #     (r/R, Reynolds number)
    polar_database = {}

    for path in polar_files:
        rR = extract_rR(path)
        Re = extract_Re(path)

        polar_database[(rR, Re)] = parse_polar_file(path)

    # Create sorted lists of available blade stations and Reynolds numbers
    rR_values = sorted(set(key[0] for key in polar_database.keys()))
    Re_values = sorted(set(key[1] for key in polar_database.keys()))

    # Create figure with three vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Reserve space below the subplots for the sliders
    plt.subplots_adjust(bottom=0.24, hspace=0.25)

    # Solid = estimated original XFOIL region
    cl_line_original, = axes[0].plot(
        [], [],
        color="black",
        linewidth=1.4,
        linestyle="-",
        label="Estimated XFOIL polar",
    )

    cd_line_original, = axes[1].plot(
        [], [],
        color="black",
        linewidth=1.4,
        linestyle="-",
    )

    cm_line_original, = axes[2].plot(
        [], [],
        color="black",
        linewidth=1.4,
        linestyle="-",
    )

    # Dashed = Viterna extrapolation
    cl_line_extrapolated, = axes[0].plot(
        [], [],
        color="0.6",
        linewidth=1.4,
        linestyle="--",
        label="Viterna extrapolation",
    )

    cd_line_extrapolated, = axes[1].plot(
        [], [],
        color="0.6",
        linewidth=1.4,
        linestyle="--",
    )

    cm_line_extrapolated, = axes[2].plot(
        [], [],
        color="0.6",
        linewidth=1.4,
        linestyle="--",
    )

    axes[0].legend(loc="upper left", frameon=False)

    axes[0].set_ylabel("$C_l$")
    axes[1].set_ylabel("$C_d$")
    axes[2].set_ylabel("$C_m$")
    axes[2].set_xlabel(r"$\alpha$ [deg]")

    # Use fixed x-limits for all polar plots
    for ax in axes:
        ax.grid(True)
        ax.set_xlim(-90, 90)

    # Keep fixed y-limits so curves can be compared while moving the sliders
    axes[0].set_ylim(-1.0, 2.0)
    axes[1].set_ylim(-0.5, 2.0)
    axes[2].set_ylim(-0.4, 0.4)

    # Create blade-station and Reynolds-number sliders
    rR_slider_ax = plt.axes([0.15, 0.105, 0.70, 0.035])
    Re_slider_ax = plt.axes([0.15, 0.045, 0.70, 0.035])

    # The sliders select indices in rR_values and Re_values
    rR_slider = Slider(
        rR_slider_ax,
        "Blade station",
        0,
        len(rR_values) - 1,
        valinit=0,
        valstep=1,
    )

    Re_slider = Slider(
        Re_slider_ax,
        "Re index",
        0,
        len(Re_values) - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        """
        Update the polar curves for the selected blade station and Reynolds number.

        The slider values are numbers, but they represent indices in the sorted
        r/R and Reynolds-number lists. Therefore they are converted to integers
        before indexing.
        """

        rR_index = int(rR_slider.val)
        Re_index = int(Re_slider.val)

        rR = rR_values[rR_index]
        Re = Re_values[Re_index]

        key = (rR, Re)

        # Some r/R-Re combinations may be missing
        if key not in polar_database:
            fig.suptitle(
                f"No polar available for r/R = {rR:.3f}, Re = {Re:.3e}",
                fontsize=12,
            )

            cl_line_original.set_data([], [])
            cd_line_original.set_data([], [])
            cm_line_original.set_data([], [])

            cl_line_extrapolated.set_data([], [])
            cd_line_extrapolated.set_data([], [])
            cm_line_extrapolated.set_data([], [])

            fig.canvas.draw_idle()
            return

        polar = polar_database[key]

        alpha = polar["alpha"].to_numpy()
        Cl = polar["Cl"].to_numpy()
        Cd = polar["Cd"].to_numpy()
        Cm = polar["Cm"].to_numpy()

        # Detect stall points used to split solid and dashed line regions
        stall = find_stall_points(alpha, Cl, Cd, Cm)

        original, extrapolated = split_original_and_extrapolated(
            alpha,
            Cl,
            Cd,
            Cm,
            stall,
        )

        # Update the solid original-region curves
        cl_line_original.set_data(original["alpha"], original["Cl"])
        cd_line_original.set_data(original["alpha"], original["Cd"])
        cm_line_original.set_data(original["alpha"], original["Cm"])

        # Update the dashed extrapolated-region curves
        cl_line_extrapolated.set_data(
            extrapolated["alpha"],
            extrapolated["Cl"],
        )

        cd_line_extrapolated.set_data(
            extrapolated["alpha"],
            extrapolated["Cd"],
        )

        cm_line_extrapolated.set_data(
            extrapolated["alpha"],
            extrapolated["Cm"],
        )

        fig.suptitle(
            f"APC {propeller_name}: extrapolated polars at r/R = "
            f"{rR:.3f} | Re = {int(Re)}",
            fontsize=12,
        )

        fig.canvas.draw_idle()

    # Draw the first selected polar before opening the interactive window
    update(None)

    # Connect the sliders after the initial draw
    rR_slider.on_changed(update)
    Re_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()