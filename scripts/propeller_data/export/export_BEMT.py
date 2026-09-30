"""
Create BEMT propeller input files from processed APC propeller data.

The script:
    1. Reads the APC blade geometry from the <propeller>-PERF.PE0 file.
    2. Removes blade stations inside the hub transition region.
    3. Converts the blade geometry to non-dimensional form.
    4. Writes the processed blade geometry to bladeGeom.txt.
    5. Reads extrapolated airfoil polar files.
    6. Groups the polar files by radial airfoil station r/R.
    7. Validates the Reynolds-number and angle-of-attack grids.
    8. Writes the stacked Cl_mesh.dat file for use as BEMT input.
    9. Writes the stacked Cd_mesh.dat file for use as BEMT input.
"""

from utilities import *

import re
import pandas as pd


# =============================================================================
# Input settings
# =============================================================================

# Tolerance used when matching filename r/R values to blade stations
RR_TOLERANCE = 5.0e-7

# Directory containing the APC <propeller>-PERF.PE0 geometry files
APC_GEO_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "external"
    / "apc"
    / "geometry"
)

# Directory that contains the APC performance data files
APC_PERF_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "external"
    / "apc"
    / "performance"
)

# Directory that contains the processed propeller subfolders
PROP_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "processed_propellers"
)

# Polar filename format:
# Airfoil_rR0.160000_Re50000.txt
POLAR_FILENAME_PATTERN = re.compile(
    r"^Airfoil_rR"
    r"(?P<r_R>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"_Re"
    r"(?P<Re>\d+)"
    r"\.txt$"
)


# =============================================================================
# APC geometry parsing
# =============================================================================

def read_text(path):
    """
    Read a text file while ignoring unsupported characters.
    """

    return Path(path).read_text(errors="ignore")


def extract_scalar(text, keyword):
    """
    Extract a scalar value following a keyword from a text.

    Example format:
        RADIUS: 5.000
    """

    match = re.search(rf"{keyword}:\s*([-+]?\d*\.?\d+)", text)

    if match is None:
        raise ValueError(f"Could not find '{keyword}' in geometry file.")

    return float(match.group(1))


def extract_airfoil(text, airfoil_number):
    """
    Extract an airfoil name from the APC -PERF.PE0 file.

    Example:
        AIRFOIL1:  1.20, E63
    """

    match = re.search(
        rf"AIRFOIL{airfoil_number}:\s*[\d.]+,\s*([A-Za-z0-9]+)",
        text,
    )

    if match is None:
        raise ValueError(f"Could not find AIRFOIL{airfoil_number}")

    return match.group(1)


def parse_apc_geo_file(path):
    """
    Parse an APC propeller geometry (-PERF.PE0) file.

    Extracted global propeller parameters:
        blades              : number of blades [-]
        radius_m            : propeller radius [m]
        hub_transition_m    : end of the hub transition region [m]
        transition_start_m  : radial location where airfoil transition starts [m]
        transition_end_m    : radial location where airfoil transition ends [m]
        airfoil1            : first airfoil name
        airfoil2            : second airfoil name

    For each retained blade station, the script stores:
        r_R                 : nondimensional radial location [-]
        station_m           : radial station measured from the propeller center [m]
        chord_m             : local chord length [m]
        t/c                 : local thickness-to-chord ratio [-]
        twist_deg           : local geometric twist angle [deg]
        sweep_y_m           : local sweep offset in the y-direction [m]

    Blade stations located inside the hub transition region are discarded.
    """

    text = read_text(path)

    # APC geometry values are given in inches where applicable
    radius_m = extract_scalar(text, "RADIUS") * 0.0254
    hub_transition_m = extract_scalar(text, "HUBTRA") * 0.0254
    transition_start_m = extract_scalar(text, "AIRFOIL1") * 0.0254
    transition_end_m = extract_scalar(text, "AIRFOIL2") * 0.0254

    blades = extract_scalar(text, "BLADES")
    airfoil1 = extract_airfoil(text, 1)
    airfoil2 = extract_airfoil(text, 2)

    blade_rows = []

    for line in text.splitlines():
        values = re.findall(r"[-+]?\d+\.\d+", line)

        # APC geometry table rows contain 14 numerical columns
        if len(values) != 14:
            continue

        values = [float(value) for value in values]

        station_m = values[0] * 0.0254
        chord_m = values[1] * 0.0254
        t_c = values[7]
        twist_deg = values[8]
        sweep_y_m = values[5] * 0.0254

        # Skip stations inside the hub transition region
        if station_m < hub_transition_m:
            continue

        r_R = station_m / radius_m

        blade_rows.append({
            "r/R": r_R,
            "station_m": station_m,
            "chord_m": chord_m,
            "t/c": t_c,
            "twist_deg": twist_deg,
            "sweep_y_m": sweep_y_m,
        })

    return (
        blade_rows,
        radius_m,
        blades,
        hub_transition_m,
        transition_start_m,
        transition_end_m,
        airfoil1,
        airfoil2,
    )


# =============================================================================
# Blade-geometry output writing
# =============================================================================

def write_blade_geometry(radius_m, blade_rows, output_path):
    """
    Write the processed blade geometry for BEMT input.

    Each row contains:
        r/R
        chord/R
        twist [deg]
        sweep/R
        airfoil filename

    The airfoil filename follows the same naming convention as the generated
    airfoil profile files.
    """

    with output_path.open("w") as file:
        file.write("#  r/R     chord/R  Twist [deg] Sweep/R   airfoil")

        for row in blade_rows:
            r_R = row["r/R"]
            chord_R = row["chord_m"] / radius_m
            sweep_R = row["sweep_y_m"] / radius_m

            file.write(
                f"\n{r_R:.6f}  "
                f"{chord_R:.6f}  "
                f"{row['twist_deg']:.6f}  "
                f"{sweep_R:.6f}  "
                f"Airfoil_rR{r_R:.6f}.txt"
            )


# =============================================================================
# Polar-file parsing
# =============================================================================

def parse_polar_filename(path):
    """
    Extract r/R and Reynolds number from a polar filename.

    Expected filename:
        Airfoil_rR0.160000_Re50000.txt
    """

    match = POLAR_FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    return {
        "path": path,
        "r_R": float(match.group("r_R")),
        "Re": int(match.group("Re")),
    }


def find_polar_files(input_dir):
    """
    Find and parse all valid polar files in the input directory.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir.resolve()}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir.resolve()}"
        )

    polar_files = []

    for path in input_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_polar_filename(path)

        if parsed is not None:
            polar_files.append(parsed)

    if not polar_files:
        raise FileNotFoundError(
            f"No files matching "
            f"'Airfoil_rR*_Re*.txt' were found in {input_dir.resolve()}."
        )

    return polar_files


def group_files_by_station(polar_files, blade_stations):
    """
    Match the polar files to the blade radial stations.

    Returned list entries:
        station             : nondimensional radial station [-]
        matching_files      : polar files corresponding to the station
    """

    grouped = []

    for station in blade_stations:
        matching_files = [
            record
            for record in polar_files
            if abs(record["r_R"] - station) <= RR_TOLERANCE
        ]

        if not matching_files:
            raise FileNotFoundError(
                f"No polar files were found for blade station "
                f"r/R = {station:.6f}."
            )

        # Sort polar files by Reynolds number
        matching_files.sort(key=lambda record: record["Re"])
        grouped.append((station, matching_files))

    matched_paths = {
        record["path"]
        for _, records in grouped
        for record in records
    }

    unmatched_files = [
        record["path"]
        for record in polar_files
        if record["path"] not in matched_paths
    ]

    if unmatched_files:
        print("Warning: the following polar files do not correspond to a")
        print("station in the blade geometry and will be ignored:")

        for path in sorted(unmatched_files):
            print(f"  {path}")

        print()

    return grouped


def read_polar_file(path):
    """
    Read one extrapolated airfoil polar file.

    Expected data columns:
        alpha               : angle of attack [deg]
        Cl                  : lift coefficient [-]
        Cd                  : drag coefficient [-]
        Cm                  : moment coefficient [-]

    The Cm column may be present but is not required for the BEMT mesh files.
    """

    rows = []

    with open(path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            parts = stripped.split()

            # Skip lines that do not contain the expected columns
            if len(parts) < 3:
                continue

            try:
                alpha = float(parts[0])
                cl = float(parts[1])
                cd = float(parts[2])

            except ValueError:
                # Skip header or malformed lines
                continue

            rows.append({
                "alpha": alpha,
                "Cl": cl,
                "Cd": cd,
                "source_line": line_number,
            })

    if not rows:
        raise ValueError(f"No usable polar data found in {path}")

    polar = pd.DataFrame(rows)

    if polar["alpha"].duplicated().any():
        duplicate_alpha = sorted(
            polar.loc[
                polar["alpha"].duplicated(keep=False),
                "alpha",
            ].unique()
        )

        raise ValueError(
            f"Duplicate angle-of-attack values found in {path}: "
            f"{duplicate_alpha}"
        )

    # Sort polar data by angle of attack
    polar = polar.sort_values("alpha").reset_index(drop=True)

    return polar


# =============================================================================
# Polar-data validation
# =============================================================================

def validate_station_polars(station, station_polars):
    """
    Confirm that all Reynolds-number files at one blade station use the same
    angle-of-attack grid.
    """

    _, reference_polar, reference_path = station_polars[0]
    reference_alpha = reference_polar["alpha"].to_numpy()

    for _, polar, path in station_polars[1:]:
        alpha = polar["alpha"].to_numpy()

        if len(alpha) != len(reference_alpha):
            raise ValueError(
                f"Different numbers of alpha values at r/R = {station:.6f}:\n"
                f"  {reference_path.name}: {len(reference_alpha)} values\n"
                f"  {path.name}: {len(alpha)} values"
            )

        if not pd.Series(alpha).equals(pd.Series(reference_alpha)):
            raise ValueError(
                f"Angle-of-attack grid mismatch at r/R = {station:.6f} "
                f"between:\n"
                f"  {reference_path.name}\n"
                f"  {path.name}"
            )


def validate_global_structure(all_station_polars):
    """
    Confirm that every radial station has the same Reynolds numbers and
    angle-of-attack grid.

    This is required for a regular stacked BEMT mesh.
    """

    reference_station, reference_polars = all_station_polars[0]

    reference_Re_values = [
        item[0]
        for item in reference_polars
    ]
    reference_alpha = reference_polars[0][1]["alpha"].to_numpy()

    for station, station_polars in all_station_polars[1:]:
        Re_values = [
            item[0]
            for item in station_polars
        ]

        if Re_values != reference_Re_values:
            raise ValueError(
                f"Reynolds-number grid mismatch between blade stations:\n"
                f"  r/R = {reference_station:.6f}: {reference_Re_values}\n"
                f"  r/R = {station:.6f}: {Re_values}"
            )

        for _, polar, path in station_polars:
            alpha = polar["alpha"].to_numpy()

            if len(alpha) != len(reference_alpha):
                raise ValueError(
                    f"Alpha-grid length mismatch in {path.name}."
                )

            if not pd.Series(alpha).equals(pd.Series(reference_alpha)):
                raise ValueError(
                    f"Alpha-grid mismatch in {path.name}."
                )


# =============================================================================
# Aerodynamic-mesh output writing
# =============================================================================

def write_mesh_file(all_station_polars, output_path, coefficient_name):
    """
    Write Cl_mesh.dat or Cd_mesh.dat using the stacked BEMT mesh structure.

    Stacking order:
        blade station 1:
            Reynolds number 1: all alpha values
            Reynolds number 2: all alpha values
            ...

        blade station 2:
            Reynolds number 1: all alpha values
            ...
    """

    with open(output_path, "w") as file:
        for _, station_polars in all_station_polars:
            for Re, polar, _ in station_polars:
                for _, row in polar.iterrows():
                    coefficient = row[coefficient_name]

                    file.write(
                        f"{Re:10d} "
                        f"{row['alpha']:14.8f} "
                        f"{coefficient:18.10e}\n"
                    )


def write_aerodynamic_mesh_files(
    polar_folder,
    blade_rows,
    output_folder,
):
    """
    Write the aerodynamic coefficient mesh files for BEMT input.

    Input parameters:
        polar_folder        : folder containing extrapolated polar files
        blade_rows          : processed blade geometry data
        output_folder       : folder where BEMT input files are written

    Output files:
        Cl_mesh.dat         : stacked lift-coefficient mesh
        Cd_mesh.dat         : stacked drag-coefficient mesh
    """

    # Extract radial stations directly from the processed blade geometry
    blade_stations = [
        row["r/R"]
        for row in blade_rows
    ]

    # Find all available extrapolated polar files
    polar_files = find_polar_files(polar_folder)

    # Group polar files according to their radial blade station
    grouped_files = group_files_by_station(
        polar_files,
        blade_stations,
    )

    all_station_polars = []

    for station, station_files in grouped_files:
        station_polars = []

        print(f"Reading r/R = {station:.6f}")

        seen_Re = set()

        for record in station_files:
            Re = record["Re"]
            path = record["path"]

            if Re in seen_Re:
                raise ValueError(
                    f"Multiple polar files found for r/R = {station:.6f} "
                    f"and Re = {Re}."
                )

            seen_Re.add(Re)

            polar = read_polar_file(path)
            station_polars.append((Re, polar, path))

            print(
                f"  Re = {Re:<10d} "
                f"{len(polar):>4d} alpha values  "
                f"({path.name})"
            )

        # Validate polar data at the current radial station
        validate_station_polars(
            station,
            station_polars,
        )

        all_station_polars.append(
            (station, station_polars)
        )

    # Validate polar structure across all radial stations
    validate_global_structure(all_station_polars)

    # Define aerodynamic mesh output files
    cl_output_file = output_folder / "Cl_mesh.dat"
    cd_output_file = output_folder / "Cd_mesh.dat"

    # Write lift-coefficient mesh
    write_mesh_file(
        all_station_polars,
        cl_output_file,
        "Cl",
    )

    # Write drag-coefficient mesh
    write_mesh_file(
        all_station_polars,
        cd_output_file,
        "Cd",
    )

    return (
        cl_output_file,
        cd_output_file,
        all_station_polars,
    )


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    # Ask the user to select a propeller that has all required input files
    prop, _, _ = ask_for_processed_propeller(
        APC_GEO_DIRECTORY,
        APC_PERF_DIRECTORY,
        PROP_DIRECTORY,
    )

    # Define original APC blade-geometry input file
    apc_geo_file = (
        APC_GEO_DIRECTORY
        / f"{prop}-PERF.PE0"
    )

    # Define processed propeller input directory
    processed_propeller_folder = (
        PROP_DIRECTORY
        / f"apc_{prop}"
    )

    # Define extrapolated airfoil polar directory
    polar_folder = (
        processed_propeller_folder
        / "airfoil_polars_extrapolated"
    )

    # Directory where the BEMT input files are written
    output_dir = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "bemt_input"
            / f"apc_{prop}"
    )

    # Create output directory if it does not yet exist
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Parse the original APC blade geometry
    (
        blade_rows,
        radius_m,
        blades,
        _,
        _,
        _,
        _,
        _,
    ) = parse_apc_geo_file(apc_geo_file)

    # Sort blade stations from root to tip
    blade_rows = sorted(
        blade_rows,
        key=lambda row: row["station_m"],
    )

    # Define blade-geometry output file
    blade_geom_output_file = (
        output_dir
        / "bladeGeom.txt"
    )

    # Write processed blade geometry
    write_blade_geometry(
        radius_m,
        blade_rows,
        blade_geom_output_file,
    )

    print(f"\nWritten: {blade_geom_output_file}")

    # Write aerodynamic coefficient mesh files
    (
        cl_output_file,
        cd_output_file,
        all_station_polars,
    ) = write_aerodynamic_mesh_files(
        polar_folder,
        blade_rows,
        output_dir,
    )

    print(f"\nWritten: {cl_output_file}")
    print(f"Written: {cd_output_file}")

    # Define source and destination airfoil-profile directories
    airfoil_profile_source = (
            processed_propeller_folder
            / "airfoil_profiles"
    )

    airfoil_profile_output = (
            output_dir
            / "airfoil_profiles"
    )

    # Copy airfoil profiles to the BEMT input directory
    shutil.copytree(
        airfoil_profile_source,
        airfoil_profile_output,
        dirs_exist_ok=True,
    )

    print(f"\nCopied:  {airfoil_profile_output}")

    # Summarize generated BEMT input data
    total_stations = len(all_station_polars)

    Reynolds_numbers = [
        item[0]
        for item in all_station_polars[0][1]
    ]

    alpha_count = len(
        all_station_polars[0][1][0][1]
    )

    rows_per_mesh_file = (
        total_stations
        * len(Reynolds_numbers)
        * alpha_count
    )

    print()
    print("BEMT input export completed successfully.")
    print(f"Propeller:            {prop}")
    print(f"Blade stations:       {total_stations}")
    print(f"Reynolds numbers:     {len(Reynolds_numbers)}")
    print(f"Alpha values per Re:  {alpha_count}")
    print(f"Rows per mesh file:   {rows_per_mesh_file}")
    print(f"Output directory:     {output_dir.resolve()}")