"""
Create smoothened airfoil polar files from XFOIL polar output files.

The script:
    1. Lets the user select a propeller.
    2. Reads all airfoil polar files for that propeller.
    3. Applies a Savitzky-Golay filter to Cl, Cd, CDp and Cm.
    4. Writes one smoothened polar file per input file.

Reads:
    - XFOIL polar files

Writes:
    - one smoothened polar file per input polar file

The output format remains compatible with extrapolate_airfoil_polars.py.
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

# Savitzky-Golay filter settings
SMOOTH_WINDOW = 9
SMOOTH_POLYORDER = 3


# =============================================================================
# XFOIL polar parsing
# =============================================================================

def read_xfoil_polar(path):
    """
    Read an XFOIL polar file and return the aerodynamic polar data.

    Input parameter:
        path                : path to the XFOIL polar file

    Returned object:
        df                  : pandas DataFrame containing the polar data

    Only the first five numerical columns are retained:
        alpha, CL, CD, CDp, CM

    Invalid or non-numerical lines are ignored.
    Duplicate angle-of-attack entries are removed after sorting by alpha.
    """

    lines = path.read_text().splitlines()

    data_start = None

    # Locate the start of the numerical table
    for i, line in enumerate(lines):
        if re.match(r"\s*-+\s+-+", line):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find polar data in {path}")

    rows = []

    # Read the numerical polar data rows
    for line in lines[data_start:]:
        parts = line.split()

        if len(parts) >= 5:
            try:
                rows.append([float(x) for x in parts[:5]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numerical polar data found in {path}")

    df = pd.DataFrame(
        rows,
        columns=["alpha", "CL", "CD", "CDp", "CM"],
    )

    df = df.sort_values("alpha")
    df = df.drop_duplicates("alpha")

    return df


# =============================================================================
# Polar smoothening
# =============================================================================

def smoothen_polar(df):
    """
    Smoothen one airfoil polar using a Savitzky-Golay filter.

    Input parameter:
        df                  : DataFrame containing the original XFOIL polar

    Returned DataFrame:
        alpha               : angle of attack [deg]
        CL                  : smoothened lift coefficient [-]
        CD                  : smoothened drag coefficient [-]
        CDp                 : smoothened pressure drag coefficient [-]
        CM                  : smoothened pitching moment coefficient [-]
    """

    if len(df) < SMOOTH_WINDOW:
        raise ValueError("Not enough polar points for smoothening")

    if SMOOTH_WINDOW % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be odd")

    smooth = df.copy()

    # Apply Savitzky-Golay filter to the aerodynamic coefficients
    for col in ["CL", "CD", "CDp", "CM"]:
        smooth[col] = savgol_filter(
            df[col].to_numpy(),
            window_length=SMOOTH_WINDOW,
            polyorder=SMOOTH_POLYORDER,
            mode="interp",
        )

    return smooth.reset_index(drop=True)


# =============================================================================
# Output writing
# =============================================================================

def write_polar(df, path):
    """
    Write one smoothened airfoil polar file.

    Input parameters:
        df                  : DataFrame containing the smoothened polar
        path                : output file path

    The generated file keeps the same first five columns as XFOIL:
        alpha, CL, CD, CDp, CM

    This makes the file readable by extrapolate_airfoil_polars.py.
    """

    with open(path, "w") as f:
        # Write a compact XFOIL-like header
        f.write("Smoothened XFOIL polar\n")
        f.write(f" {SMOOTH_WINDOW+1}-point moving average applied\n")
        f.write("\n")

        # Write the polar table header
        f.write("   alpha    CL        CD       CDp       CM\n")
        f.write("  ------ -------- --------- --------- --------\n")

        # Write the aerodynamic polar data
        for _, row in df.iterrows():
            f.write(
                f"{row['alpha']:8.3f} "
                f"{row['CL']:8.4f} "
                f"{row['CD']:9.5f} "
                f"{row['CDp']:9.5f} "
                f"{row['CM']:8.4f}\n"
            )


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the smoothening workflow for all polar files of one propeller.
    """

    # Ask which propeller polars should be processed
    prop, input_dir = ask_for_propeller_with_polars(APC_DIRECTORY, PROP_DIRECTORY)
    print()

    # Create an output folder
    output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "airfoil_polars_smoothened"
    )

    # Delete existing folder and all contents
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Recreate empty folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all polar files
    files = sorted(input_dir.glob("*_Re*.txt"))
    total_files = len(files)

    for i, file in enumerate(files, start=1):
        try:

            # Show progress on a single terminal line
            print(
                f"\rSmoothening polar {i}/{total_files}: {file.name}",
                end="",
                flush=True,
            )

            # Polar smoothening
            polar = read_xfoil_polar(file)
            smoothened = smoothen_polar(polar)

            # Output writing
            out_file = output_dir / file.name
            write_polar(smoothened, out_file)

        except Exception as e:
            print(f"\nFailed for {file.name}: {e}")

    # Clear progress line before printing the final messages
    print("\r" + " " * 120, end="\r")

    print("Airfoil polar smoothening finished.")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()