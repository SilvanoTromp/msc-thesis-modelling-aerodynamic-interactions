"""
Create Viterna-extrapolated airfoil polar files from
smoothened XFOIL polar output files.

The script:
    1. Lets the user select a propeller.
    2. Reads all airfoil polar files for that propeller.
    3. Detects positive and negative stall points.
    4. Uses the original XFOIL polar inside the reliable range.
    5. Applies Viterna extrapolation outside stall.
    6. Smoothly blends between original and extrapolated data.
    7. Writes one extrapolated polar file per input file.

Reads:
    - XFOIL polar files

Writes:
    - one extrapolated polar file per input polar file

Output alpha range:
    - ALPHA_MIN to ALPHA_MAX degrees
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

# Final alpha range and resolution for the extrapolated polar
ALPHA_MIN = -90.0
ALPHA_MAX = 90.0
ALPHA_STEP = 2

# Width of the smooth transition region between XFOIL and Viterna data [deg]
BLEND_WIDTH = 6


# =============================================================================
# XFOIL polar parsing
# =============================================================================

def read_xfoil_polar(path):
    """
    Read an XFOIL polar file and return the aerodynamic polar data.

    Input parameter:
        path                : path to the XFOIL polar file

    Expected XFOIL columns:
        alpha               : angle of attack [deg]
        CL                  : lift coefficient [-]
        CD                  : drag coefficient [-]
        CDp                 : pressure drag coefficient [-]
        CM                  : pitching moment coefficient [-]

    Returned object:
        df                  : pandas DataFrame containing the polar data

    Only the first five numerical columns are retained.

    XFOIL marks the start of the polar table with a dashed separator line.
    The numerical data starts on the next line.

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

        # XFOIL polar rows contain at least five numerical columns
        if len(parts) >= 5:
            try:
                rows.append([float(x) for x in parts[:5]])

            # XFOIL polar rows contain at least five numerical columns
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numerical polar data found in {path}")

    df = pd.DataFrame(
        rows,
        columns=["alpha", "CL", "CD", "CDp", "CM"],
    )

    # Ensure alpha is increasing and remove duplicate alpha values
    df = df.sort_values("alpha")
    df = df.drop_duplicates("alpha")

    return df


# =============================================================================
# Extrapolation
# =============================================================================

def viterna_side(CD_MAX, alpha_deg, alpha_stall_deg, cl_stall, cd_stall):
    """
    Apply Viterna extrapolation on one side of an aerodynamic polar.

    The Viterna method is used to extend lift and drag coefficients beyond
    the detected stall angle. The extrapolation is applied independently to:
        - the positive angle-of-attack side
        - the negative angle-of-attack side

    Input parameters:
        CD_MAX              : maximum drag coefficient estimation [-]
        alpha_deg           : angles of attack requiring extrapolation [deg]
        alpha_stall_deg     : detected stall angle for this side [deg]
        cl_stall            : lift coefficient at the stall point [-]
        cd_stall            : drag coefficient at the stall point [-]

    Returned arrays:
        cl                  : extrapolated lift coefficients [-]
        cd                  : extrapolated drag coefficients [-]

    The extrapolation constants are computed such that the Viterna model
    matches the original polar continuously at the stall point.
    """

    # Convert angles to radians for the Viterna equations
    a = np.radians(alpha_deg)
    a_s = np.radians(alpha_stall_deg)

    # Precompute the sine and cosines of the (stall) angle of attack
    sin_a = np.sin(a)
    cos_a = np.cos(a)

    sin_s = np.sin(a_s)
    cos_s = np.cos(a_s)

    # Small number to avoid division by zero near singular angles
    eps = 1e-8

    # Drag extrapolation constants
    B1 = CD_MAX
    B2 = (cd_stall - B1 * sin_s**2) / (cos_s + eps)

    # Lift extrapolation constants
    A1 = B1 / 2.0
    A2 = (
        cl_stall - A1 * np.sin(2.0 * a_s)
    ) * sin_s / (cos_s**2 + eps)

    # Viterna lift and drag equations
    cd = B1 * sin_a**2 + B2 * cos_a
    cl = A1 * np.sin(2.0 * a) + A2 * cos_a**2 / (sin_a + eps)

    return cl, cd


def extrapolate_cm(CD_MAX, alpha_deg, cm_stall, alpha_stall_deg):
    """
    Extrapolate the pitching moment coefficient beyond stall.

    The extrapolated moment coefficient starts from the stall value and
    smoothly transitions toward a simple flat-plate estimate at +/-90 degrees:
        Cm(+90 deg) = -CD_MAX / 4
        Cm(-90 deg) = +CD_MAX / 4

    Input parameters:
        CD_MAX             : maximum drag coefficient estimation [-]
        alpha_deg          : angles of attack requiring extrapolation [deg]
        cm_stall           : pitching moment coefficient at stall [-]
        alpha_stall_deg    : detected stall angle [deg]

    Returned array:
        cm                 : extrapolated pitching moment coefficients [-]

    The transition from the stall value to the flat-plate estimate is
    controlled using a smoothstep weighting function.
    """

    # Ensure alpha is handled as a NumPy array
    alpha_deg = np.asarray(alpha_deg)

    # Use absolute angles because the blending depends only on the
    # angular distance from stall toward +/-90 degrees
    stall_abs = abs(alpha_stall_deg)
    alpha_abs = abs(alpha_deg)

    # Smooth transition weight: 0 at stall, 1 at +/- 90 degrees
    w = smoothstep(alpha_abs, stall_abs, 90)

    # Flat-plate moment estimate depends on the sign of alpha
    cm_90 = np.where(alpha_deg >= 0.0, -CD_MAX / 4.0, CD_MAX / 4.0)

    # Blend between the stall value and the flat-plate estimate
    cm = blend(cm_stall, cm_90, w)

    return cm


def extrapolate_polar(CD_MAX, df):
    """
    Extrapolate one airfoil polar to the requested angle-of-attack range.

    Input parameter:
        CD_MAX              : maximum drag coefficient estimation [-]
        df                  : DataFrame containing the original XFOIL polar

    Required input columns:
        alpha               : angle of attack [deg]
        CL                  : lift coefficient [-]
        CD                  : drag coefficient [-]
        CM                  : pitching moment coefficient [-]

    Returned DataFrame:
        alpha               : output angle-of-attack grid [deg]
        Cl                  : extrapolated lift coefficient [-]
        Cd                  : extrapolated drag coefficient [-]
        Cm                  : extrapolated pitching moment coefficient [-]

    Extrapolation procedure:
        1. Interpolate the original XFOIL polar onto the output alpha grid.
        2. Detect the positive and negative stall points from the original data.
        3. Use Viterna extrapolation for Cl and Cd beyond stall.
        4. Extrapolate Cm separately toward a simple flat-plate limit.
        5. Blend smoothly between the XFOIL data and extrapolated data over
           BLEND_WIDTH degrees around each stall point.
    """

    # Ensure data is handled as NumPy arrays
    alpha_data = df["alpha"].to_numpy()
    cl_data = df["CL"].to_numpy()
    cd_data = df["CD"].to_numpy()
    cm_data = df["CM"].to_numpy()

    # Create the uniform output angle-of-attack grid
    alpha_out = np.arange(
        ALPHA_MIN,
        ALPHA_MAX + ALPHA_STEP,
        ALPHA_STEP,
    )

    # Interpolate the original XFOIL data onto the output grid,
    # outside the original XFOIL range, np.interp keeps the nearest endpoint value;
    # these regions are later replaced/blended with Viterna data
    cl_interp = np.interp(alpha_out, alpha_data, cl_data)
    cd_interp = np.interp(alpha_out, alpha_data, cd_data)
    cm_interp = np.interp(alpha_out, alpha_data, cm_data)

    # Initialize the output arrays with the interpolated XFOIL data
    cl_out = cl_interp.copy()
    cd_out = cd_interp.copy()
    cm_out = cm_interp.copy()

    # Detect stall points used to match the extrapolated polar
    stall = find_stall_points(
        alpha_data,
        cl_data,
        cd_data,
        cm_data,
    )

    alpha_pos_stall = stall["alpha_pos"]
    cl_pos_stall = stall["cl_pos"]
    cd_pos_stall = stall["cd_pos"]
    cm_pos_stall = stall["cm_pos"]

    alpha_neg_stall = stall["alpha_neg"]
    cl_neg_stall = stall["cl_neg"]
    cd_neg_stall = stall["cd_neg"]
    cm_neg_stall = stall["cm_neg"]

    # -------------------------------------------------------------------------
    # Positive post-stall side
    # -------------------------------------------------------------------------

    # Select points starting slightly before positive stall
    mask_pos = alpha_out > (alpha_pos_stall - BLEND_WIDTH / 2.0)

    if np.any(mask_pos):
        alpha_pos = alpha_out[mask_pos]

        # Compute Viterna Cl/Cd continuation matched at positive stall
        cl_v_pos, cd_v_pos = viterna_side(
            CD_MAX,
            alpha_pos,
            alpha_pos_stall,
            cl_pos_stall,
            cd_pos_stall,
        )

        # Compute Cm continuation matched at positive stall
        cm_v_pos = extrapolate_cm(
            CD_MAX,
            alpha_pos,
            cm_pos_stall,
            alpha_pos_stall,
        )

        # Smoothly transition from XFOIL data to extrapolated data
        w_pos = smoothstep(
            alpha_pos,
            alpha_pos_stall - BLEND_WIDTH / 2,
            alpha_pos_stall + BLEND_WIDTH / 2)

        cl_out[mask_pos] = blend(
            cl_interp[mask_pos],
            cl_v_pos,
            w_pos,
        )

        cd_out[mask_pos] = blend(
            cd_interp[mask_pos],
            cd_v_pos,
            w_pos,
        )

        cm_out[mask_pos] = blend(
            cm_interp[mask_pos],
            cm_v_pos,
            w_pos,
        )

    # -------------------------------------------------------------------------
    # Negative post-stall side
    # -------------------------------------------------------------------------

    # Select points ending slightly after negative stall
    mask_neg = alpha_out < (alpha_neg_stall + BLEND_WIDTH / 2.0)

    if np.any(mask_neg):
        alpha_neg = alpha_out[mask_neg]

        # Compute Viterna Cl/Cd continuation matched at negative stall
        cl_v_neg, cd_v_neg = viterna_side(
            CD_MAX,
            alpha_neg,
            alpha_neg_stall,
            cl_neg_stall,
            cd_neg_stall,
        )

        # Compute Cm continuation matched at negative stall
        cm_v_neg = extrapolate_cm(
            CD_MAX,
            alpha_neg,
            cm_neg_stall,
            alpha_neg_stall,
        )

        # Smoothly transition from extrapolated data to XFOIL data,
        # the negative side uses the reversed blend direction
        w_neg = 1 - smoothstep(
            alpha_neg,
            alpha_neg_stall - BLEND_WIDTH / 2,
            alpha_neg_stall + BLEND_WIDTH / 2,
        )

        cl_out[mask_neg] = blend(
            cl_interp[mask_neg],
            cl_v_neg,
            w_neg,
        )

        cd_out[mask_neg] = blend(
            cd_interp[mask_neg],
            cd_v_neg,
            w_neg,
        )

        cm_out[mask_neg] = blend(
            cm_interp[mask_neg],
            cm_v_neg,
            w_neg,
        )

    return pd.DataFrame({
        "alpha": alpha_out,
        "Cl": cl_out,
        "Cd": cd_out,
        "Cm": cm_out,
    })


# =============================================================================
# Output writing
# =============================================================================

def write_polar(df, path):
    """
    Write one extrapolated airfoil polar file.

    Input parameters:
        df                  : DataFrame containing the extrapolated polar
        path                : output file path

    Required DataFrame columns:
        alpha               : angle of attack [deg]
        Cl                  : lift coefficient [-]
        Cd                  : drag coefficient [-]
        Cm                  : pitching moment coefficient [-]

    The generated file contains:
        1. A small header describing the extrapolation settings.
        2. A formatted aerodynamic polar table.

    Output table columns:
        alpha               : angle of attack [deg]
        Cl                  : lift coefficient [-]
        Cd                  : drag coefficient [-]
        Cm                  : pitching moment coefficient [-]
    """

    with open(path, "w") as f:
        # Write the extrapolation settings header
        f.write("Viterna extrapolated polar\n")
        f.write(f" alpha from {ALPHA_MIN:.1f} to {ALPHA_MAX:.1f} deg\n")
        f.write(f" BLEND_WIDTH = {BLEND_WIDTH:.3f} deg\n")
        f.write("\n")

        # Write the polar table header
        f.write(" alpha        Cl          Cd          Cm\n")
        f.write(" ------   ---------   ---------   ---------\n")

        # Write the aerodynamic polar data
        for _, row in df.iterrows():
            f.write(
                f"{row['alpha']:7.3f} "
                f"{row['Cl']:11.6f} "
                f"{row['Cd']:11.6f} "
                f"{row['Cm']:11.6f}\n"
            )


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the full extrapolation workflow for all polar files of one propeller.
    """

    # Ask which propeller polars should be processed
    prop, input_dir = ask_for_propeller_with_smoothened_polars(APC_DIRECTORY, PROP_DIRECTORY)
    print()

    # Create an output folder
    output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "airfoil_polars_extrapolated"
    )

    # Delete existing folder and all contents
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Recreate empty folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all polar files
    files = sorted(input_dir.glob("*_Re*.txt"))
    total_files = len(files)

    # Read blade aspect ratio used in the Viterna estimate for maximum drag
    bladeGeom = read_text(f"{prop}/bladeGeom.txt")
    AR = extract_scalar(bladeGeom, "AR")

    #  Estimate the maximum drag coefficient using the Viterna relation
    CD_MAX = 1.11 + 0.018 * AR

    for i, file in enumerate(files, start=1):
        try:

            # Show progress on a single terminal line
            print(
                f"\rEvaluating polar {i}/{total_files}: {file.name}",
                end="",
                flush=True,
            )

            # Polar extrapolation
            polar = read_xfoil_polar(file)
            extrapolated = extrapolate_polar(CD_MAX, polar)

            # Output writing
            out_file = output_dir / file.name
            write_polar(extrapolated, out_file)

        except Exception as e:
            print(f"\nFailed for {file.name}: {e}")

    # Clear progress line before printing the final messages
    print("\r" + " " * 120, end="\r")

    print("Airfoil polar extrapolation finished.")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()