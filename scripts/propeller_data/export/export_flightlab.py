"""
Create FLIGHTLAB propeller input files from processed propeller data.

The script:
    1. Reads blade geometry from bladeGeom.txt.
    2. Extracts r/R, radius, chord, twist, and sweep.
    3. Converts chord from metres to feet.
    4. Converts sweep offset to sweep angle.
    5. Writes the blade-property .tab file for use as FLIGHTLAB input.
    6. Reads extrapolated airfoil polar files.
    7. Groups the polar files by radial airfoil station r/R.
    8. Extracts Cl, Cd, and Cm at selected angles of attack.
    9. Writes one aerodynamic node distribution .tab file for use as FLIGHTLAB input.
    10. Writes one blade-section .tab file for each aerodynamic node for use as FLIGHTLAB input.
"""

from utilities import *

import re
from collections import defaultdict


# =============================================================================
# Input settings
# =============================================================================

# Conversion factor from metres to feet
M_TO_FT = 3.280839895

# Angles of attack to write to the section files
AOA_VALUES = np.arange(-90.0, 90 + 1e-9, 2)

# Polar filename format:
# Airfoil_rR0.160000_Re50000.txt
POLAR_FILENAME_PATTERN = re.compile(
    r"Airfoil_rR(?P<r_R>[0-9.]+)_Re(?P<Re>[0-9.]+)\.txt"
)

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

# Directory that contains the propeller subfolders
PROP_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "processed_propellers"
)

# Directory where the FLIGHTLAB input files are written
FLIGHTLAB_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "flightlab_input"
)


# =============================================================================
# Blade-geometry parsing
# =============================================================================

def read_blade_geom(path):
    """
    Read blade geometry data from bladeGeom.txt.

    Expected columns:
        r/R station chord t/c twist sweep airfoil

    Returned list entries:
        r_R                 : nondimensional radial station [-]
        station_m           : radial station [m]
        chord_m             : local chord length [m]
        twist_deg           : local blade twist [deg]
        sweep_m             : local sweep offset [m]
    """

    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip lines that do not contain the expected columns
            if len(parts) < 7:
                continue

            try:
                r_R = float(parts[0])
                station_m = float(parts[1])
                chord_m = float(parts[2])
                twist_deg = float(parts[4])
                sweep_m = float(parts[5])

            except ValueError:
                # Skip header or malformed lines
                continue

            data.append((r_R, station_m, chord_m, twist_deg, sweep_m))

    if not data:
        raise ValueError(f"No usable blade geometry data found in {path}")

    return data


# =============================================================================
# Polar-file parsing
# =============================================================================

def read_polar_file(path):
    """
    Read an extrapolated airfoil polar file.

    Expected data columns:
        alpha Cl Cd Cm

    Returned arrays:
        alpha_deg           : angle of attack [deg]
        cl                  : lift coefficient [-]
        cd                  : drag coefficient [-]
        cm                  : moment coefficient [-]
    """

    alpha_deg = []
    cl = []
    cd = []
    cm = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split()

            # Skip header or malformed lines
            if len(parts) < 4:
                continue

            try:
                alpha_i = float(parts[0])
                cl_i = float(parts[1])
                cd_i = float(parts[2])
                cm_i = float(parts[3])

            except ValueError:
                continue

            alpha_deg.append(alpha_i)
            cl.append(cl_i)
            cd.append(cd_i)
            cm.append(cm_i)

    if not alpha_deg:
        raise ValueError(f"No usable polar data found in {path}")

    return (
        np.array(alpha_deg),
        np.array(cl),
        np.array(cd),
        np.array(cm),
    )


def collect_polar_files(polar_folder):
    """
    Collect all polar files and group them by radial station.

    Returned dictionary:
        grouped_polars[r_R] = [(Re, path), ...]
    """

    grouped_polars = defaultdict(list)

    for path in sorted(polar_folder.glob("Airfoil_rR*_Re*.txt")):

        match = POLAR_FILENAME_PATTERN.match(path.name)

        if match is None:
            continue

        r_R = float(match.group("r_R"))
        Re = float(match.group("Re"))

        grouped_polars[r_R].append((Re, path))

    if not grouped_polars:
        raise ValueError(f"No usable polar files found in {polar_folder}")

    # Sort Reynolds numbers within each radial station
    for r_R in grouped_polars:
        grouped_polars[r_R].sort(key=lambda item: item[0])

    return dict(sorted(grouped_polars.items()))


# =============================================================================
# Blade-property output writing
# =============================================================================

def write_propblade_file(input_path, output_path):
    """
    Write a propeller blade-property file for FLIGHTLAB input.

    Input parameters:
        input_path          : path to bladeGeom.txt
        output_path         : path to the output .tab file

    The output file contains:
        BCHORD              : blade chord distribution [ft]
        BTW                 : blade twist distribution [deg]
        BCGOFF              : chordwise c.g. offset [ft]
        BSEGIXX             : rotary inertia distribution
        BSEGIYY             : flapwise inertia distribution
        BSEGIZZ             : chordwise inertia distribution
        BMPL                : blade mass distribution
        BSEGE0              : midchord offset from elastic axis [ft]
        BSWEEP              : blade sweep angle [deg]
        BDROOP              : blade droop angle [deg]
    """

    data = read_blade_geom(input_path)

    with open(output_path, "w") as f:
        f.write("""##
#Rigid Blade Property data for tiltrotor blade element model
#
# {Blade chord} {Blade twist} {Chordwise c.g. offset}
# {Blade rotary inertia distribution}
# {Blade flapwise inertia distribution}
# {Blade chordwise inertia distribution}
# {Blade mass distribution}
# {Blade midchord offset from e.a.}
# {Blade tip sweep}   {Blade tip droop}
##
# Propeller data; source: bladeGeom.txt
!M BCHORD
!U nd ft
""")

        # Blade chord distribution
        for r_R, _, chord_m, _, _ in data:
            chord_ft = chord_m * M_TO_FT
            f.write(f"{r_R:.10f}\t{chord_ft:.9f}\n")

        f.write("""!M BTW
!U nd deg
""")

        # Blade twist distribution
        for r_R, _, _, twist_deg, _ in data:
            f.write(f"{r_R:.10f}\t{twist_deg:.9f}\n")

        f.write("""!M BCGOFF
!U nd ft
    0.0000    0.0100
    1.0000    0.0100
!M BSEGIXX
!U nd slug-ft
   0.0000    0.0000
   1.0000    0.0000
!M BSEGIYY
!U nd slug-ft
   0.0000    0.0000
   1.0000    0.0000
!M BSEGIZZ
!U nd slug-ft
   0.0000    0.0000
   1.0000    0.0000
!M BMPL
!U nd slug/ft
     0.000  3.8012E-03
     1.000  3.8012E-03
# see data/reference/propblade.txt constant bmpl over rotordiameter
!M BSEGE0
!U nd ft
    0.1000    0.0163
    0.2000    0.0207
    0.3000    0.0229
    0.4000    0.0230
    0.5000    0.0215
    0.6000    0.0191
    0.7000    0.0160
    0.8000    0.0126
    0.9000    0.0094
    1.0000    0.0094
# ^ 1 cm, solidity weighted blade chord of disk model / 4 = 0.032808 ft
!M BSWEEP
!U nd deg
""")

        # Blade sweep angle distribution
        for r_R, station_m, _, _, sweep_m in data:
            sweep_rad = np.atan2(sweep_m, station_m)
            sweep_deg = np.degrees(sweep_rad)

            f.write(f"{r_R:.10f}\t{sweep_deg:.9f}\n")

        f.write("""!M BDROOP
!U nd deg
   0.0000    0.0
   1.0000    0.0
""")


# =============================================================================
# Aerodynamic-node output writing
# =============================================================================

def write_blade_aero_nodes_file(prop, grouped_polars, output_folder):
    """
    Write blade aerodynamic nodal distribution file for FLIGHTLAB input.
    """

    output_path = output_folder / f"EVO-{prop}-bladeaeronodes.tab"

    with open(output_path, "w") as f:
        f.write("# Title: Blade aerodynamic nodal distribution\n\n")
        f.write("!T\tAEROXNODE\n")
        f.write("!U\tnd\n")

        for r_R in grouped_polars.keys():
            f.write(f"{r_R:.10f}\n")

    return output_path


# =============================================================================
# Blade-section output writing
# =============================================================================

def write_section_file(prop, section_index, r_R, polar_entries, output_folder):
    """
    Write one propeller blade-section file for FLIGHTLAB input.

    Input parameters:
        prop                : propeller name used in the output filename
        section_index       : section number used in the output filename
        r_R                 : nondimensional radial station [-]
        polar_entries       : list of (Re, path) polar entries
        output_folder       : folder where the output .tab file is written

    The output file contains:
        AOACL               : angle-of-attack grid for Cl [deg]
        AOACD               : angle-of-attack grid for Cd [deg]
        AOACM               : angle-of-attack grid for Cm [deg]
        RECL                : Reynolds-number index grid for Cl [-]
        RECD                : Reynolds-number index grid for Cd [-]
        RECM                : Reynolds-number index grid for Cm [-]
        CLTAB               : lift coefficient table [-]
        CDTAB               : drag coefficient table [-]
        CMTAB               : moment coefficient table [-]
    """

    cl_columns = []
    cd_columns = []
    cm_columns = []

    for Re, path in polar_entries:

        alpha_deg, cl, cd, cm = read_polar_file(path)

        # Interpolate the polar data onto the requested angle-of-attack grid
        cl_interp = np.interp(AOA_VALUES, alpha_deg, cl)
        cd_interp = np.interp(AOA_VALUES, alpha_deg, cd)
        cm_interp = np.interp(AOA_VALUES, alpha_deg, cm)

        cl_columns.append(cl_interp)
        cd_columns.append(cd_interp)
        cm_columns.append(cm_interp)

    cl_table = np.column_stack(cl_columns)
    cd_table = np.column_stack(cd_columns)
    cm_table = np.column_stack(cm_columns)

    output_path = (
        output_folder
        / f"EVO-{prop}-propellerblade-section-{section_index}.tab"
    )

    with open(output_path, "w") as f:

        f.write("!T\tAOACL\n")
        f.write("!U\tdeg\n")
        for aoa in AOA_VALUES:
            f.write(f"{aoa:g}\t\n")

        f.write("!T\tAOACD\n")
        f.write("!U\tdeg\n")
        for aoa in AOA_VALUES:
            f.write(f"{aoa:g}\t\n")

        f.write("!T\tAOACM\n")
        f.write("!U\tdeg\n")
        for aoa in AOA_VALUES:
            f.write(f"{aoa:g}\t\n")

        f.write("!T\tRECL\n")
        f.write("!U\tnd\n")
        for Re, _ in polar_entries:
            f.write(f"{int(Re)}\t\n")

        f.write("!T\tRECD\n")
        f.write("!U\tnd\n")
        for Re, _ in polar_entries:
            f.write(f"{int(Re)}\t\n")

        f.write("!T\tRECM\n")
        f.write("!U\tnd\n")
        for Re, _ in polar_entries:
            f.write(f"{int(Re)}\t\n")

        f.write("!M\tCLTAB\n")
        f.write("!U\tnd\n")
        for row in cl_table:
            f.write("\t".join(f"{value:.4E}" for value in row) + "\n")

        f.write("!M\tCDTAB\n")
        f.write("!U\tnd\n")
        for row in cd_table:
            f.write("\t".join(f"{value:.4E}" for value in row) + "\n")

        f.write("!M\tCMTAB\n")
        f.write("!U\tnd\n")
        for row in cm_table:
            f.write("\t".join(f"{value:.4E}" for value in row) + "\n")

    return output_path


def write_propblade_section_files(prop, polar_folder, output_folder):
    """
    Write all propeller aerodynamic input files.

    Input parameters:
        prop                : propeller name used in the output filenames
        polar_folder        : folder containing extrapolated polar files
        output_folder       : folder where section files are written
    """

    grouped_polars = collect_polar_files(polar_folder)

    written_files = []

    # Write aerodynamic node distribution
    aero_nodes_path = write_blade_aero_nodes_file(
        prop,
        grouped_polars,
        output_folder,
    )
    written_files.append(aero_nodes_path)

    # Write one blade-section file for each aerodynamic node
    for section_index, (r_R, polar_entries) in enumerate(
        grouped_polars.items(),
        start=1,
    ):
        output_path = write_section_file(
            prop,
            section_index,
            r_R,
            polar_entries,
            output_folder,
        )

        written_files.append(output_path)

    return written_files


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

    # Define processed propeller input directory
    processed_propeller_folder = (
        PROP_DIRECTORY
        / f"apc_{prop}"
    )

    # Define blade-geometry input file
    blade_geom_file = (
        processed_propeller_folder
        / "bladeGeom.txt"
    )

    # Define extrapolated airfoil polar directory
    polar_folder = (
        processed_propeller_folder
        / "airfoil_polars_extrapolated"
    )

    # Define blade-property output file
    propblade_output_file = (
        FLIGHTLAB_OUTPUT_DIRECTORY
        / f"EVO-{prop}-propellerblade-properties.tab"
    )

    # Create output directory if it does not yet exist
    FLIGHTLAB_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write propeller blade-property file
    write_propblade_file(
        blade_geom_file,
        propblade_output_file,
    )

    print(f"Written: {propblade_output_file}")

    # Write aerodynamic node and blade-section files
    written_files = write_propblade_section_files(
        prop,
        polar_folder,
        FLIGHTLAB_OUTPUT_DIRECTORY,
    )

    for output_file in written_files:
        print(f"Written: {output_file}")