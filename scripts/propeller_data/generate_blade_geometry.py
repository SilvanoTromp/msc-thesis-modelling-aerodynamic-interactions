"""
Convert APC propeller geometry data from a <propeller>-PERF.PE0 file
into a simplified bladeGeom.txt file for later aerodynamic analysis.

The script:
    1. Lists all available APC propeller geometry files.
    2. Lets the user select a propeller.
    3. Reads the APC blade geometry table.
    4. Removes blade stations inside the hub transition region.
    5. Converts relevant APC data from inches to meters.
    7. Writes the processed blade geometry to bladeGeom.txt.

Output columns:
    r/R
    station [m]
    chord [m]
    t/c
    twist [deg]
    sweep [m]
    airfoil
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


# =============================================================================
# Blade geometry processing
# =============================================================================

def compute_average_chord(blade_rows):
    """
    Compute the spanwise average chord in meters.

    The average chord is computed using trapezoidal integration over the span:

        average_chord = integral(chord dr) / blade_span

    This value is used to estimate a simple blade aspect ratio.
    """

    r = np.array([row["station_m"] for row in blade_rows])
    c = np.array([row["chord_m"] for row in blade_rows])

    span = r[-1] - r[0]

    average_chord = np.trapezoid(c, r) / span

    return average_chord


def write_blade_geometry(prop, radius_m, AR, blades, blade_rows, OUTPUT_FILE):
    """
    Write the processed blade geometry to OUTPUT_FILE.

    Each row contains:
        r/R
        station [m]
        chord [m]
        t/c
        twist [deg]
        sweep [m]
        airfoil filename

    The airfoil filename follows the same naming convention as the generated
    airfoil profile files.
    """
    with OUTPUT_FILE.open("w") as file:
        file.write(f"# APC {prop} | blades: {blades:.0f} | radius: {radius_m} m | AR: {AR}")
        file.write("\n\n# r/R           station [m]     chord [m]       t/c"
                   "             twist [deg]      sweep [m]       airfoil")

        for row in blade_rows:

            file.write(
                f"\n{row['r/R']:.9f}     "
                f"{row['station_m']:.9f}     "
                f"{row['chord_m']:.9f}     "
                f"{row['t/c']:.9f}     "
                f"{row['twist_deg']:.9f}     "
                f"{row['sweep_y_m']:.9f}     "
                f"Airfoil_rR{row["r/R"]:.6f}.txt"
            )


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the full APC-to-bladeGeom.txt conversion workflow
    """
    apc_geo_file, prop = ask_for_apc_geo_file(APC_DIRECTORY)

    # Parse the original APC blade geometry
    (blade_rows,
     radius_m,
     blades,
     _,
     _,
     _,
     _,
     _) = parse_apc_geo_file(apc_geo_file)

    # Sort blade stations from root to tip
    blade_rows = sorted(blade_rows, key=lambda row: row["station_m"])

    # Estimate the blade aspect ratio using the mean chord length
    avg_chord_m = compute_average_chord(blade_rows)
    AR = radius_m / avg_chord_m

    # Create the output path for the converted blade geometry file
    output_file = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "bladeGeom.txt"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the converted blade geometry data to file
    write_blade_geometry(prop, radius_m, AR, blades, blade_rows, output_file)

    # Inform the user that the conversion has completed
    print("\nAPC blade geometry converted successfully.")
    print(f"Output file: {output_file.resolve()}")


if __name__ == "__main__":
    main()