"""
Create interpolated and thickness-scaled airfoil profile files from APC
propeller geometry data and two reference airfoil coordinate files.

The script:
    1. Lists all available APC propeller geometry files.
    2. Lets the user select a propeller.
    3. Reads the APC blade geometry table.
    4. Reads the two reference airfoil names from the APC file.
    5. Reads the corresponding reference airfoil coordinate files.
    6. Converts the reference airfoils to upper and lower surface distributions.
    7. Blends the two reference airfoils over the radial transition region.
    8. Scales each local profile to the APC thickness distribution.
    9. Writes one airfoil coordinate file per station.

The script reads:
    - APC propeller geometry data from a <propeller>-PERF.PE0 file
    - Root airfoil coordinates from Airfoil Tools .dat file
    - Tip airfoil coordinates from Airfoil Tools .dat file

It writes:
    - one interpolated/scaled airfoil coordinate file per blade station
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

# Directory containing the airfoil data files from Airfoil Tools
AT_DIRECTORY = Path(
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "external"
    / "airfoils"
)

# Common chordwise grid used for all generated airfoils,
# the grid runs from leading edge x=0 to trailing edge x=1
X_GRID = np.linspace(0.0, 1.0, 101)


# =============================================================================
# Airfoil preparation
# =============================================================================

def airfoil_to_upper_and_lower(upper, lower, x_grid):
    """
    Convert one reference airfoil to upper and lower surface distributions.

    The conversion process is:
        1. Sort the upper surface coordinates by increasing x/c.
        2. Sort the lower surface coordinates by increasing x/c.
        3. Interpolate both surfaces onto the common chordwise grid.

    Input parameters:
        upper               : upper surface airfoil coordinates [-]
        lower               : lower surface airfoil coordinates [-]
        x_grid              : common chordwise grid [-]

    Returned arrays:
        y_upper             : upper surface y/c coordinates [-]
        y_lower             : lower surface y/c coordinates [-]
    """

    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    y_upper = np.interp(x_grid, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_grid, lower[:, 0], lower[:, 1])

    return y_upper, y_lower


# =============================================================================
# Airfoil blending
# =============================================================================

def blend_airfoils(
    station,
    transition_start,
    transition_end,
    target_thickness_ratio,
    airfoil1,
    airfoil2,
):
    """
    Create one local airfoil by blending two reference airfoils.

    The blending process is:
        1. Compute a blending weight based on the radial station.
        2. Blend the upper surface distributions.
        3. Blend the lower surface distributions.
        4. Scale the blended thickness distribution to match the local APC thickness ratio.
        5. Reconstruct the upper and lower airfoil surfaces.

    Input parameters:
        station                 : radial blade station [m]
        transition_start        : start of the airfoil transition region [m]
        transition_end          : end of the airfoil transition region [m]
        target_thickness_ratio  : target local thickness-to-chord ratio [-]
        airfoil_1               : first reference airfoil
        airfoil_2               : second reference airfoil

    Airfoil format:
        airfoil_i = (y_upper, y_lower)

    Returned arrays:
        y_upper                 : upper surface y/c coordinates [-]
        y_lower                 : lower surface y/c coordinates [-]

    A linear interpolation is used inside the transition region.
    """

    # Unpack upper and lower surface coordinates of the two reference airfoils
    y_upper_1, y_lower_1 = airfoil1
    y_upper_2, y_lower_2 = airfoil2

    # Outside the transition region, use the nearest reference airfoil directly
    if station < transition_start:
        y_upper, y_lower = y_upper_1, y_lower_1
    elif station > transition_end:
        y_upper, y_lower = y_upper_2, y_lower_2

    # Within the transition region, interpolate between both airfoils
    else:

        # Compute linear weight
        weight = linearweight(
            station,
            transition_start,
            transition_end,
        )

        # Blend upper and lower surfaces
        y_upper = blend(y_upper_1, y_upper_2, weight)
        y_lower = blend(y_lower_1, y_lower_2, weight)

    # Compute blended camber and thickness
    camber = 0.5 * (y_upper + y_lower)
    thickness = y_upper - y_lower

    # Current maximum thickness before scaling
    current_thickness_ratio = np.max(thickness)

    if current_thickness_ratio <= 0.0:
        raise ValueError(
            f"Invalid airfoil thickness at station {station:.4f}: "
            f"{current_thickness_ratio}"
        )

    # Scale thickness so the final profile matches the APC local thickness
    thickness_scale = target_thickness_ratio / current_thickness_ratio
    thickness *= thickness_scale

    # Reconstruct upper and lower surfaces from camber and thickness
    y_upper = camber + 0.5 * thickness
    y_lower = camber - 0.5 * thickness

    return y_upper, y_lower


# =============================================================================
# Output writing
# =============================================================================

def write_airfoil_file(filename, x_grid, y_upper, y_lower, output_dir):
    """
    Write one local airfoil coordinate file.

    Format
        filename as header
        upper surface from trailing edge to leading edge
        lower surface from leading edge to trailing edge
    """

    path = output_dir / filename

    with path.open("w") as file:
        file.write(f"{filename}\n")

        # Upper surface: trailing edge to leading edge
        for x, y in zip(x_grid[::-1], y_upper[::-1]):
            file.write(f"{x:.6f}     {y:.6f}\n")

        # Lower surface: leading edge to trailing edge
        for x, y in zip(x_grid[1:], y_lower[1:]):
            file.write(f"{x:.6f}     {y:.6f}\n")


def create_airfoil_profiles(
    blade_rows,
    airfoil1,
    airfoil2,
    transition_start,
    transition_end,
    output_dir,
):
    """
    Create and write local airfoil profile files for all blade stations.

    Input parameters:
        blade_rows          : blade geometry rows containing radial station data
        airfoil1            : first reference airfoil, stored as upper and lower surfaces
        airfoil2            : second reference airfoil, stored as upper and lower surfaces
        transition_start    : radial location where airfoil transition starts [m]
        transition_end      : radial location where airfoil transition ends [m]
        output_dir          : directory where the airfoil files are written

    Required blade row entries:
        r_R                 : nondimensional radial location [-]
        station             : radial blade station measured from propeller center [m]
        thickness_ratio     : local target thickness-to-chord ratio [-]

    For each blade station, the function:
        1. Creates the corresponding airfoil filename.
        2. Blends the two reference airfoils based on radial position.
        3. Scales the local thickness to match the APC thickness ratio.
        4. Writes the resulting airfoil coordinates to file.

    The generated airfoil filenames follow the convention:
        Airfoil_rR<radial_position>.txt
    """

    for row in blade_rows:
        r_R = row["r/R"]
        station = row["station_m"]

        airfoil_filename = f"Airfoil_rR{r_R:.6f}.txt"

        y_upper, y_lower = blend_airfoils(
            station,
            transition_start,
            transition_end,
            row["t/c"],
            airfoil1,
            airfoil2,
        )

        write_airfoil_file(
            airfoil_filename,
            X_GRID,
            y_upper,
            y_lower,
            output_dir
        )


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the full local-airfoil generation workflow.
    """

    # Ask the user to select an APC propeller geometry file
    # and retrieve the corresponding APC <propeller>-PERF.PE0 path
    apc_file, prop = ask_for_apc_geo_file(APC_DIRECTORY)

    # Create the directory used to store the generated
    # local airfoil profile coordinate files
    output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "airfoil_profiles"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse APC geometry and extract the original blade stations
    (blade_rows,
     _,
     _,
     _,
     transition_start_m,
     transition_end_m,
     airfoil1,
     airfoil2) = parse_apc_geo_file(apc_file)

    # Construct paths to the reference airfoil coordinate files
    airfoil1_file = AT_DIRECTORY / f"{airfoil1}.dat"
    airfoil2_file = AT_DIRECTORY / f"{airfoil2}.dat"

    # Ensure both referenced airfoil files exist before continuing
    if not airfoil1_file.exists():
        raise FileNotFoundError(f"Could not find airfoil file: {airfoil1_file}")

    if not airfoil2_file.exists():
        raise FileNotFoundError(f"Could not find airfoil file: {airfoil2_file}")

    # Sort blade stations from root to tip
    blade_rows = sorted(blade_rows, key=lambda row: row["station_m"])

    # Read the two reference airfoils
    upper_1, lower_1 = parse_airfoil_file(airfoil1_file)
    upper_2, lower_2 = parse_airfoil_file(airfoil2_file)

    # Convert both reference airfoils to upper/lower form on X_GRID
    airfoil1 = airfoil_to_upper_and_lower(upper_1, lower_1, X_GRID)
    airfoil2 = airfoil_to_upper_and_lower(upper_2, lower_2, X_GRID)

    # Generate and write all local profile files
    create_airfoil_profiles(
        blade_rows,
        airfoil1,
        airfoil2,
        transition_start_m,
        transition_end_m,
        output_dir
    )

    print("\nAirfoil profile creation finished.")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()