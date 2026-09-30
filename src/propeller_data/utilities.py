import os
import sys
import re
from pathlib import Path
import numpy as np
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter


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
        raise ValueError(f"Could not find '{keyword}' in {text} file.")

    return float(match.group(1))


def extract_rR(path):
    """
    Extract the nondimensional radial station r/R from an airfoil or polar filename.

    Expected filename formats:
        Airfoil_rR<value>.txt
        Airfoil_rR<value>_Re<value>.txt
    """

    match = re.search(r"rR([0-9.]+)", path.stem)

    if match is None:
        raise ValueError(f"Could not extract r/R from filename: {path.name}")

    return float(match.group(1))


def extract_Re(path):
    """
    Extract the Reynolds number from a polar filename.

    Expected filename format:
        Airfoil_rR<value>_Re<value>.txt
    """

    match = re.search(r"Re([0-9.eE+-]+)", path.stem)

    if match is None:
        raise ValueError(f"Could not extract Re from filename: {path.name}")

    return float(match.group(1))


def extract_airfoil(text, airfoil_number):
    """
    Extract an airfoil name from the APC -PERF.PE0 file.

    Example:
        AIRFOIL1:  1.20, E63
    """

    match = re.search(
        rf"AIRFOIL{airfoil_number}:\s*[\d.]+,\s*([A-Za-z0-9]+)",
        text
    )

    if match is None:
        raise ValueError(f"Could not find AIRFOIL{airfoil_number}")

    return match.group(1)


def get_available_propellers(directory):
    """
    Return all propeller names for which a -PERF.PEO file exists.

    A propeller is identified by a file named:
        <propeller_name>-PERF.PE0
    """

    pe0_files = sorted(directory.glob("*-PERF.PE0"))
    return [file.name.replace("-PERF.PE0", "") for file in pe0_files]


def get_airfoil_files(airfoil_dir):
    """
    Get all generated airfoil files sorted from root to tip.

    Sorting is based on the r/R value extracted from the filename.
    """

    return sorted(
        airfoil_dir.glob("Airfoil_rR*.txt"),
        key=extract_rR,
    )


def get_polar_files(polar_dir):
    """
    Return all XFOIL polar files sorted by:
        1. blade station r/R
        2. Reynolds number
    """

    return sorted(
        polar_dir.glob("Airfoil_rR*_Re*.txt"),
        key=lambda path: (extract_rR(path), extract_Re(path)),
    )


def get_processed_propeller(apc_geo_directory, prop_directory):
    """
    Return all propellers with a complete aerodynamic analysis setup.

    Input parameters:
        apc_geo_directory   : directory containing APC geometry (-PERF.PE0) files
        prop_directory      : directory containing processed propeller data

    A propeller is considered valid if the following files and folders exist:
        1. APC geometry file: <propeller>-PERF.PE0
        2. Propeller output directory: <propeller>
        3. Processed blade geometry: bladeGeom.txt
        4. Generated airfoil profiles: airfoil_profiles/
        5. Extrapolated airfoil polars: airfoil_polars_extrapolated/
        6. At least one extrapolated polar file: Airfoil_rR*_Re*.txt

    Returned value:
        valid_prop          : list of propeller names with complete data
    """

    valid_prop = []

    # Scan all available APC geometry files
    for APC_GEO_FILE in sorted(apc_geo_directory.glob("*-PERF.PE0")):

        prop = APC_GEO_FILE.name.replace("-PERF.PE0", "")
        prop_folder = prop_directory / f"apc_{prop}"

        # Required processed files and directories
        blade_geom = prop_folder / "bladeGeom.txt"

        airfoil_profiles = (
            prop_folder / "airfoil_profiles"
        )

        polar_folder = (
            prop_folder / "airfoil_polars_extrapolated"
        )

        # Check whether the full aerodynamic dataset exists
        if (
            prop_folder.exists()
            and blade_geom.exists()
            and airfoil_profiles.exists()
            and polar_folder.exists()
            and list(polar_folder.glob("Airfoil_rR*_Re*.txt"))
        ):
            valid_prop.append(prop)

    return valid_prop


def print_available_propellers(message, available_propellers, columns=8):
    """
    Print available propellers in a compact column layout.
    """

    if not available_propellers:
        print("\nNo propellers found.")
        return

    max_width = max(len(propeller) for propeller in available_propellers) + 4

    print(f"\n{message}")

    for i, propeller in enumerate(available_propellers, start=1):
        print(f"{propeller:<{max_width}}", end="")

        if i % columns == 0:
            print()

    print()


def ask_for_apc_geo_file(directory):
    """
    Ask the user to select an APC propeller geometry file.

    A valid APC geometry file must follow the naming convention:
        <propeller_name>-PERF.PE0

    All available propellers found inside the APC geometry directory are listed.

    The function keeps asking until a valid propeller is selected.

    Returned values:
        apc_geo_file       : path to the selected APC geometry file
        prop               : selected propeller name
    """

    while True:

        # List all available APC propeller geometry files
        available_propellers = get_available_propellers(directory)

        print_available_propellers(
            "Available propellers:",
            available_propellers)

        prop = input("\nEnter propeller name: ").strip()

        apc_geo_file = directory / f"{prop}-PERF.PE0"

        # Check whether the selected APC geometry file exists
        if apc_geo_file.exists():
            return apc_geo_file, prop

        print(f"\nNo APC file found for '{prop}'.")
        print(f"Expected file: {Path(__file__).resolve()}/{directory}/{apc_geo_file.name}")
        print("Please choose one of the available propellers.")


def ask_for_propeller_with_profiles(apc_geo_directory, prop_directory):
    """
    Ask the user to select a propeller for which airfoil profiles exist.

    A valid propeller must contain:
        1. an APC geometry file:
               <propeller>-PERF.PE0
        2. a directory containing generated airfoil profile files:
               <propeller>/airfoil_profiles/Airfoil_rR*.txt

    Only propellers with available airfoil profile files are listed.

    The function keeps asking until a valid propeller is selected.

    Returned values:
        prop               : selected propeller name
        airfoil_folder     : directory containing the local airfoil profiles
    """
    while True:

        # Only list propellers for which airfoil profiles exist
        available_propellers = []

        for prop in get_available_propellers(apc_geo_directory):

            airfoil_folder = (
                prop_directory / f"apc_{prop}" / "airfoil_profiles"
            )

            if (
                airfoil_folder.exists()
                and list(airfoil_folder.glob("Airfoil_rR*.txt"))
            ):
                available_propellers.append(prop)

        print_available_propellers(
            "Available propellers with airfoil profiles:",
            available_propellers)

        prop = input("\nEnter propeller name: ").strip()

        airfoil_folder = (
            prop_directory / f"apc_{prop}" / "airfoil_profiles"
        )

        # Check whether airfoil profile files exist
        if (
                not airfoil_folder.exists()
                or not list(airfoil_folder.glob("Airfoil_rR*.txt"))
        ):
            print(f"\nNo airfoil profiles found for {prop}.")
            print(f"Expected folder: {Path(__file__).resolve()}/{airfoil_folder}")
            print("Please choose one of the available propellers.")
            continue

        return prop, airfoil_folder


def ask_for_propeller_with_polars(apc_geo_directory, prop_directory):
    """
    Ask the user to select a propeller for which airfoil polars exist.

    A valid propeller must contain:
        1. an APC geometry file:
                 <propeller>-PERF.PE0
        2. a directory containing generated airfoil polars files:
                <propeller>/airfoil_polars/Airfoil_rR*.txt

    Only propellers with available airfoil polars files are listed.

    The function keeps asking until a valid propeller is selected.

    Returned values:
        prop               : selected propeller name
        polar_folder       : directory containing the local airfoil polars
    """
    while True:

        # Only list propellers for which airfoil polars exist
        available_propellers = []

        for prop in get_available_propellers(apc_geo_directory):

            polar_folder = (
                    prop_directory / f"apc_{prop}" / "airfoil_polars"
            )

            if (
                    polar_folder.exists()
                    and list(polar_folder.glob("Airfoil_rR*.txt"))
            ):
                available_propellers.append(prop)

        print_available_propellers(
            "Available propellers with airfoil polars:",
            available_propellers)

        prop = input("\nEnter propeller name: ").strip()

        polar_folder = (
                prop_directory / prop / "airfoil_polars"
        )

        # Check whether airfoil polar files exist
        if (
                not polar_folder.exists()
                or not list(polar_folder.glob("Airfoil_rR*.txt"))
        ):
            print(f"\nNo airfoil polars found for {prop}.")
            print(f"Expected folder: {Path(__file__).resolve()}/{polar_folder}")
            print("Please choose one of the available propellers.")
            continue

        return prop, polar_folder


def ask_for_propeller_with_smoothened_polars(apc_geo_directory, prop_directory):
    """
    Ask the user to select a propeller for which smoothened airfoil polars exist.

    A valid propeller must contain:
        1. an APC geometry file:
                 <propeller>-PERF.PE0
        2. a directory containing smoothened airfoil polars files:
                <propeller>/airfoil_polars_smoothened/Airfoil_rR*.txt

    Only propellers with smoothened airfoil polars files are listed.

    The function keeps asking until a valid propeller is selected.

    Returned values:
        prop               : selected propeller name
        polar_folder       : directory containing the local smoothened airfoil polars
    """
    while True:

        # Only list propellers for which smoothened airfoil polars exist
        available_propellers = []

        for prop in get_available_propellers(apc_geo_directory):

            polar_folder = (
                    prop_directory / f"apc_{prop}" / "airfoil_polars_smoothened"
            )

            if (
                    polar_folder.exists()
                    and list(polar_folder.glob("Airfoil_rR*.txt"))
            ):
                available_propellers.append(prop)

        print_available_propellers(
            "Available propellers with smoothened airfoil polars:",
            available_propellers)

        prop = input("\nEnter propeller name: ").strip()

        polar_folder = (
                prop_directory / f"apc_{prop}" / "airfoil_polars_smoothened"
        )

        # Check whether airfoil polar files exist
        if (
                not polar_folder.exists()
                or not list(polar_folder.glob("Airfoil_rR*.txt"))
        ):
            print(f"\nNo smoothened airfoil polars found for {prop}.")
            print(f"Expected folder: {Path(__file__).resolve()}/{polar_folder}")
            print("Please choose one of the available propellers.")
            continue

        return prop, polar_folder


def ask_for_propeller_with_extrapolated_polars(apc_geo_directory, prop_directory):
    """
    Ask the user to select a propeller for which extrapolated airfoil polars exist.

    A valid propeller must contain:
        1. an APC geometry file:
                 <propeller>-PERF.PE0
        2. a directory containing extrapolated airfoil polars files:
                <propeller>/airfoil_polars_extrapolated/Airfoil_rR*.txt

    Only propellers with extrapolated airfoil polars files are listed.

    The function keeps asking until a valid propeller is selected.

    Returned values:
        prop               : selected propeller name
        polar_folder       : directory containing the local extrapolated airfoil polars
    """
    while True:

        # Only list propellers for which airfoil profiles exist
        available_propellers = []

        for prop in get_available_propellers(apc_geo_directory):

            polar_folder = (
                    prop_directory / f"apc_{prop}" / "airfoil_polars_extrapolated"
            )

            if (
                    polar_folder.exists()
                    and list(polar_folder.glob("Airfoil_rR*.txt"))
            ):
                available_propellers.append(prop)

        print_available_propellers(
            "Available propellers with extrapolated airfoil polars:",
            available_propellers)

        prop = input("\nEnter propeller name: ").strip()

        polar_folder = (
                prop_directory / f"apc_{prop}" / "airfoil_polars_extrapolated"
        )

        # Check whether airfoil polar files exist
        if (
                not polar_folder.exists()
                or not list(polar_folder.glob("Airfoil_rR*.txt"))
        ):
            print(f"\nNo extrapolated airfoil polars found for {prop}.")
            print(f"Expected folder: {Path(__file__).resolve()}/{polar_folder}")
            print("Please choose one of the available propellers.")
            continue

        return prop, polar_folder


def ask_for_processed_propeller(
    apc_geo_directory,
    apc_perf_directory,
    prop_directory,
):
    """
    Ask the user to select a propeller with all required analysis files.

    A valid propeller must contain:
        1. an APC geometry file: <propeller>-PERF.PE0
        2. generated blade geometry and extrapolated polar data
           inside the propeller directory
        3. an APC performance file: PER3_<propeller>.dat

    Only propellers with a complete analysis setup are listed.

    Returned values:
        prop                : selected propeller name
        APC_GEO_FILE        : APC geometry file path
        APC_PERF_FILE       : APC performance file path
    """

    while True:

        # Determine all propellers with a complete file setup
        valid_propellers = get_processed_propeller(
            apc_geo_directory,
            prop_directory,
        )

        if not valid_propellers:
            raise ValueError(
                "No valid propellers found. "
                "Please first create blade geometry and "
                "extrapolated airfoil polars."
            )

        # List all available valid propellers
        print("\nAvailable processed propellers:")

        for propeller in valid_propellers:
            print(f"  - {propeller}")

        prop = input("\nEnter propeller name: ").strip()

        # Check whether the selected propeller is valid
        if prop in valid_propellers:

            APC_GEO_FILE = (
                apc_geo_directory / f"{prop}-PERF.PE0"
            )

            APC_PERF_FILE = (
                apc_perf_directory / f"PER3_{prop}.dat"
            )

            # Ensure the APC performance file exists
            if not APC_PERF_FILE.exists():
                raise FileNotFoundError(
                    f"Could not find APC performance file: "
                    f"{APC_PERF_FILE}"
                )

            return prop, APC_GEO_FILE, APC_PERF_FILE

        print(f"\n'{prop}' is not available or incomplete.")


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
        r_R                  : nondimensional radial location [-]
        station_m            : radial station measured from the propeller center [m]
        chord_m              : local chord length [m]
        t/c                  : local thickness-to-chord ratio [-]
        twist_deg            : local geometric twist angle [deg]
        sweep_y_m            : local sweep offset in the y-direction [m]

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

        # Skip stations inside the hub transition region.
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


def parse_airfoil_file(path):
    """
    Parse an Airfoil Tools style airfoil coordinate file.

    Expected file structure:
        airfoil name
        number of upper and lower surface points
        upper surface coordinates
        lower surface coordinates

    Extracted airfoil parameters:
        n_upper            : number of upper surface coordinate points [-]
        n_lower            : number of lower surface coordinate points [-]

    Returned coordinate arrays:
        upper              : upper surface coordinates [x/c, y/c]
        lower              : lower surface coordinates [x/c, y/c]

    The coordinate arrays are split using the upper and lower point counts
    specified in the file.
    """

    lines = read_text(path).splitlines()

    count_index = None
    n_upper = None
    n_lower = None

    # Find the line containing the number of upper and lower surface points
    for index, line in enumerate(lines):
        numbers = re.findall(r"\d+\.?", line)

        if len(numbers) == 2:
            n_upper = int(float(numbers[0]))
            n_lower = int(float(numbers[1]))
            count_index = index
            break

    if count_index is None:
        raise ValueError(f"Could not find upper/lower point counts in {path}")

    coordinates = []

    # Read all coordinate pairs after the point-count line
    for line in lines[count_index + 1:]:
        values = re.findall(r"[-+]?\d+\.\d+", line)

        # Airfoil coordinate rows contain x/c and y/c
        if len(values) == 2:
            coordinates.append([float(values[0]), float(values[1])])

    coordinates = np.asarray(coordinates)

    # Split the coordinate array into upper and lower surfaces
    upper = coordinates[:n_upper]
    lower = coordinates[n_upper:n_upper + n_lower]

    return upper, lower


def parse_polar_file(path):
    """
    Parse one aerodynamic polar file into a pandas DataFrame.

    Input parameter:
        path                : path to the polar file

    Expected numerical columns:
        alpha               : angle of attack [deg]
        Cl                  : lift coefficient [-]
        Cd                  : drag coefficient [-]
        Cm                  : pitching moment coefficient [-]

    Returned object:
        df                  : DataFrame containing the polar data

    The parser automatically ignores:
        - header lines
        - empty lines
        - non-numerical lines

    The resulting polar data is sorted by angle of attack and duplicate
    alpha values are removed.
    """
    rows = []

    # Read the polar file line-by-line
    for line in Path(path).read_text().splitlines():
        values = line.split()

        # Numerical polar rows must contain at least four columns
        if len(values) < 4:
            continue

        try:
            alpha = float(values[0])
            cl = float(values[1])
            cd = float(values[2])
            cm = float(values[3])
            rows.append([alpha, cl, cd, cm])

        # Ignore header lines or invalid rows
        except ValueError:
            continue

    if len(rows) == 0:
        raise ValueError(f"No polar data found in {path}")

    # Create the polar DataFrame
    df = pd.DataFrame(
        rows,
        columns=["alpha", "Cl", "Cd", "Cm"],
    )

    # Ensure increasing alpha order and remove duplicate alpha values
    df = df.sort_values("alpha")
    df = df.drop_duplicates("alpha")

    return df


def parse_APC_perf_file(path):
    """
    Parse an APC propeller performance file.

    Input parameter:
        path                : APC performance file path

    APC performance files are organised into RPM blocks. Each block starts with:
        PROP RPM = <RPM>
    followed by rows containing operating-point data for that RPM.

    Returned object:
        data                : dictionary indexed by RPM

    Data structure:
        data[RPM] = [
            operating_point_1,
            operating_point_2,
            ...
        ]

    Stored operating-point quantities:
        V                   : flight speed [mph]
        J                   : advance ratio [-]
        Pe                  : propeller efficiency [-]
        Ct                  : thrust coefficient [-]
        Cp                  : power coefficient [-]
        Power_W             : shaft power [W]
        Torque_Nm           : shaft torque [Nm]
        Thrust_N            : thrust [N]
        Mach                : blade-tip Mach number [-]
        Re75                : Reynolds number at 75% radius [-]
        FoM                 : figure of merit [-]
    """

    data = {}

    current_rpm = None

    with open(path, "r", errors="ignore") as file:
        lines = file.readlines()

    for line in lines:

        # Detect the start of a new RPM block
        rpm_match = re.search(r"PROP RPM =\s+(\d+)", line)

        if rpm_match:
            current_rpm = int(rpm_match.group(1))
            data[current_rpm] = []
            continue

        # Ignore all lines before the first RPM block
        if current_rpm is None:
            continue

        values = line.split()

        # APC operating-point rows contain 15 numerical columns
        if len(values) != 15:
            continue

        try:

            # Store the relevant APC operating-point quantities
            row = {
                "V": float(values[0]),                 # flight speed [mph]
                "J": float(values[1]),                 # advance ratio [-]
                "Pe": float(values[2]),                # propeller efficiency [-]
                "Ct": float(values[3]),                # thrust coefficient [-]
                "Cp": float(values[4]),                # power coefficient [-]
                "Power_W": float(values[8]),           # shaft power [W]
                "Torque_Nm": float(values[9]),         # shaft torque [Nm]
                "Thrust_N": float(values[10]),         # thrust [N]
                "Mach": float(values[12]),             # blade-tip Mach number [-]
                "Re75": float(values[13].replace(".", "")),
                "FoM": float(values[-1]),              # figure of merit [-]
            }

            data[current_rpm].append(row)

        # Ignore invalid or partially corrupted rows
        except ValueError:
            continue

    return data


def smoothstep(x, x0, x1):
    """
    Return a smooth transition weight between 0 and 1.
    """

    s = (x - x0) / (x1 - x0)
    s = np.clip(s, 0.0, 1.0)

    return 3 * s**2 - 2 * s**3

def linearweight(x, x0, x1):
    """
    Return a linear weight between 0 and 1.
    """

    return (x - x0) / (x1 - x0)


def blend(x0, x1, w):
    """
    Linearly blend two values or arrays.
    """

    return (1.0 - w) * x0 + w * x1


def interpolate_surface(surface, x_grid):
    """
    Interpolate an airfoil surface onto a common chordwise x-grid.

    Input parameters:
        surface             : airfoil surface coordinates [x/c, y/c]
        x_grid              : target chordwise interpolation grid [-]

    Surface format:
        surface[:, 0]       : x/c coordinates [-]
        surface[:, 1]       : y/c coordinates [-]

    Returned array:
        interpolated_y      : interpolated y/c values on x_grid [-]

    The input surface may not be ordered in increasing x/c. Therefore,
    the surface coordinates are first sorted. Duplicate x/c values are
    removed because np.interp requires a strictly increasing x-array.
    """

    x = surface[:, 0]
    y = surface[:, 1]

    # Sort the surface coordinates in increasing x/c direction
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]

    # Remove duplicate x/c coordinates
    x_unique, unique_indices = np.unique(x, return_index=True)
    y_unique = y[unique_indices]

    # Interpolate the surface onto the common x-grid
    return np.interp(x_grid, x_unique, y_unique)


def airfoil_to_camber_and_thickness(upper, lower, x_grid):
    """
    Convert airfoil surface coordinates into camber and thickness distributions.

    Input parameters:
        upper               : upper surface coordinates [x/c, y/c]
        lower               : lower surface coordinates [x/c, y/c]
        x_grid              : common chordwise interpolation grid [-]

    Surface format:
        surface[:, 0]       : x/c coordinates [-]
        surface[:, 1]       : y/c coordinates [-]

    Returned arrays:
        camber              : camber-line distribution [y/c]
        thickness           : thickness distribution [t/c]

    Definitions:
        camber    = 0.5 * (y_upper + y_lower)
        thickness = y_upper - y_lower

    Both quantities are evaluated on the common chordwise x-grid.
    """

    # Interpolate upper and lower surfaces onto the common x-grid
    y_upper = interpolate_surface(upper, x_grid)
    y_lower = interpolate_surface(lower, x_grid)

    # Compute camber and thickness distributions
    camber = 0.5 * (y_upper + y_lower)
    thickness = y_upper - y_lower

    return camber, thickness


def find_stall_points(alpha, cl, cd, cm):
    """
    Determine the positive and negative stall points from an XFOIL polar.

    Input arrays:
        alpha               : angle of attack values [deg]
        cl                  : lift coefficients [-]
        cd                  : drag coefficients [-]
        cm                  : pitching moment coefficients [-]

    Returned dictionary:
        alpha_pos           : detected positive stall angle [deg]
        cl_pos              : lift coefficient at positive stall [-]
        cd_pos              : drag coefficient at positive stall [-]
        cm_pos              : pitching moment coefficient at positive stall [-]

        alpha_neg           : detected negative stall angle [deg]
        cl_neg              : lift coefficient at negative stall [-]
        cd_neg              : drag coefficient at negative stall [-]
        cm_neg              : pitching moment coefficient at negative stall [-]

    Stall detection method:
        Positive side:
            Use the positive-alpha point with maximum CL.

        Negative side:
            Starting from alpha = -4 deg, scan toward decreasing alpha and
            detect the first point where the local lift-curve slope becomes
            almost zero.

    Positive side:
        Select maximum CL for alpha >= 0.

    Negative side:
        Scan from alpha <= -4 toward decreasing alpha.
    """

    # Ensure arrays are handled as NumPy arrays
    alpha = np.asarray(alpha)
    cl = np.asarray(cl)
    cd = np.asarray(cd)
    cm = np.asarray(cm)

    # Compute the local lift-curve slope over the complete polar
    local_slope = np.gradient(cl, alpha)

    # -------------------------------------------------------------------------
    # Positive side: use maximum lift coefficient
    # -------------------------------------------------------------------------
    pos_indices = np.where(alpha >= 0.0)[0]

    if len(pos_indices) > 0:

        # Use the positive-alpha point with maximum CL as positive stall
        i_pos = pos_indices[np.argmax(cl[pos_indices])]

    else:
        i_pos = len(alpha) - 1

    # -------------------------------------------------------------------------
    # Negative side: scan from alpha = -4 toward smaller alpha
    # -------------------------------------------------------------------------
    neg_indices = np.where(alpha <= -4.0)[0]

    if len(neg_indices) > 0:

        # Default fallback: use the most negative available alpha point
        i_neg = neg_indices[0]

        for i in neg_indices[::-1]:

            # Use the first point where the slope becomes almost zero
            if local_slope[i] < 0.05:
                i_neg = i
                break

    else:
        i_neg = 0

    return {
        "alpha_pos": alpha[i_pos],
        "cl_pos": cl[i_pos],
        "cd_pos": cd[i_pos],
        "cm_pos": cm[i_pos],
        "alpha_neg": alpha[i_neg],
        "cl_neg": cl[i_neg],
        "cd_neg": cd[i_neg],
        "cm_neg": cm[i_neg],
    }