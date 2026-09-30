"""
Plot the generated local airfoil coordinate files for a selected APC propeller.

The script:
    1. Lets the user select a propeller with generated airfoil profiles.
    2. Reads all Airfoil_rR*.txt files from the airfoil_profiles folder.
    3. Sorts the airfoils from blade root to tip using the r/R value in the filename.
    4. Splits each airfoil contour into upper and lower surfaces.
    5. Opens an interactive Matplotlib plot.
    6. Uses a slider to move through the blade stations.
    7. Shows both the airfoil contour and the camber line.

Reads:
    <propeller>/airfoil_profiles/Airfoil_rR*.txt

Expected coordinate-file format:
    first line : airfoil name or header
    next lines : x/c    y/c

Shows:
    - airfoil contour at each radial station
    - corresponding camber line
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
# Airfoil loading
# =============================================================================

def load_airfoil_geometry(path):
    """
    Load one generated airfoil coordinate file.

    Input parameter:
        path                : path to an Airfoil_rR*.txt file

    Returned dictionary:
        name                : airfoil file name without extension
        r_R                 : nondimensional radial blade station [-]
        contour             : full airfoil contour coordinates [x/c, y/c]
        upper               : upper surface coordinates [x/c, y/c]
        lower               : lower surface coordinates [x/c, y/c]

    The generated coordinate files are assumed to contain one header line,
    followed by two numerical columns:
        x/c    y/c

    The airfoil contour is split at the leading edge. This works for the usual
    XFOIL-style ordering where the points run from the trailing edge over one
    surface to the leading edge, and then back to the trailing edge over the
    other surface.
    """

    # Read the coordinate table
    contour = np.loadtxt(path, skiprows=1)

    # Make sure a single-row file is still treated as a 2D array
    contour = np.atleast_2d(contour)

    if contour.shape[1] < 2:
        raise ValueError(f"Expected at least two coordinate columns in {path}")

    # Keep only x/c and y/c in case extra columns are accidentally present
    contour = contour[:, :2]

    if len(contour) < 4:
        raise ValueError(f"Too few coordinate points found in {path}")

    # The leading edge is the point with the smallest x/c value
    leading_edge_index = np.argmin(contour[:, 0])

    # Split the closed contour into two surfaces
    surface_1 = contour[:leading_edge_index + 1]
    surface_2 = contour[leading_edge_index:]

    if len(surface_1) < 2 or len(surface_2) < 2:
        raise ValueError(
            f"Could not split {path.name} into upper and lower surfaces."
        )

    # Identify which surface is upper/lower using the average y/c value
    if np.mean(surface_1[:, 1]) >= np.mean(surface_2[:, 1]):
        upper = surface_1
        lower = surface_2
    else:
        upper = surface_2
        lower = surface_1

    return {
        "name": path.stem,
        "r_R": extract_rR(path),
        "contour": contour,
        "upper": upper,
        "lower": lower,
    }


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the interactive airfoil-geometry plotting workflow.
    """

    # Ask the user to select a propeller that has generated airfoil profiles
    propeller_name, airfoil_dir = ask_for_propeller_with_profiles(
        APC_DIRECTORY,
        PROP_DIRECTORY,
    )

    # Find all generated airfoil files and sort them from root to tip
    airfoil_files = get_airfoil_files(airfoil_dir)

    if not airfoil_files:
        raise ValueError(f"No airfoil files found in {airfoil_dir.resolve()}.")

    # Load all airfoils before opening the figure
    airfoils = [
        load_airfoil_geometry(path)
        for path in airfoil_files
    ]

    # -------------------------------------------------------------------------
    # Create figure and axes
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Reserve space below the plot for the slider
    plt.subplots_adjust(bottom=0.35)

    # Empty line objects are created once and updated by the slider callback
    contour_line, = ax.plot(
        [],
        [],
        "-o",
        linewidth=0.8,
        markersize=2,
        label="Airfoil contour",
    )

    camber_line, = ax.plot(
        [],
        [],
        "--",
        linewidth=1.0,
        label="Camber line",
    )

    # Keep a fixed scale so thickness/camber changes can be compared directly
    # between blade stations
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticks(np.arange(-0.2, 0.21, 0.1))

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")

    # -------------------------------------------------------------------------
    # Create blade-station slider
    # -------------------------------------------------------------------------

    slider_ax = plt.axes([0.15, 0.10, 0.70, 0.04])

    slider = Slider(
        ax=slider_ax,
        label="Blade station",
        valmin=0,
        valmax=len(airfoils) - 1,
        valinit=0,
        valstep=1,
    )

    def update(index):
        """
        Update the plotted airfoil for the selected blade station.

        The slider value is a number, but it represents an index in the sorted
        airfoil list. Therefore it is converted to an integer before indexing.
        """

        index = int(index)
        airfoil = airfoils[index]

        contour = airfoil["contour"]
        upper = airfoil["upper"]
        lower = airfoil["lower"]

        # Update the airfoil contour
        contour_line.set_data(
            contour[:, 0],
            contour[:, 1],
        )

        # Compute the camber line on a clean chordwise grid
        x_grid = np.linspace(0.0, 1.0, 200)

        camber, _ = airfoil_to_camber_and_thickness(
            upper,
            lower,
            x_grid,
        )

        camber_line.set_data(x_grid, camber)

        ax.set_title(
            f"APC {propeller_name}: airfoil shape at r/R = "
            f"{airfoil['r_R']:.3f}"
        )

        fig.canvas.draw_idle()

    # Draw the first airfoil before opening the interactive window
    update(0)

    # Connect the slider after the initial draw
    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()