"""
Run XFOIL analyses for all generated airfoil profile files
of a selected APC propeller.

The script:
    1. Lets the user select a propeller for which airfoil profiles are available.
    2. Finds all generated airfoil profile files for that propeller.
    3. Creates a clean output folder for the polar files.
    4. Runs XFOIL for each airfoil at several Reynolds numbers.
    5. Sweeps through the requested angle-of-attack range.
    6. Saves one polar file per airfoil/Reynolds-number combination.

Output files:
    <airfoil_name>_Re<Reynolds_number>.txt

Each output file contains:
    alpha
    CL
    CD
    CDp
    CM
    Top_Xtr
    Bot_Xtr
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

# Reynolds numbers for which each airfoil is analysed
Reynolds_numbers = np.arange(50000, 250000 + 1e-6, 50000)

# Reynolds numbers known to trigger floating-point errors in XFOIL
Reynolds_numbers_err = np.array([130000])

# XFOIL transition amplification factor
Ncrit = 9

# XFOIL maximum number of iterations per angle of attack
MAX_ITER = 100

# Name or path of the XFOIL executable
xfoil_exe = "xfoil"

# Maximum allowed runtime per requested alpha value
alpha_timeout_seconds = 10

# Angle-of-attack sweep used for each airfoil/Reynolds-number combination
alphas = np.arange(-20, 20 + 1e-6, 0.25)

# Number of parallel airfoil/Reynolds-number jobs
max_workers = 4


# =============================================================================
# XFOIL utilities
# =============================================================================

def count_polar_alpha_entries(polar_file):
    """
    Count the number of completed angle-of-attack entries in an XFOIL polar file.

    Input parameter:
        polar_file          : XFOIL polar file path

    Returned value:
        n                   : number of successfully computed alpha entries

    The function scans the polar file and counts numerical polar-data rows.
    Header lines, separator lines, and other non-numerical content are ignored
    automatically.

    If the polar file does not exist, zero entries are returned.
    """

    # No polar file exists yet
    if not polar_file.exists():
        return 0

    n = 0

    with open(polar_file, "r", errors="ignore") as f:

        for line in f:

            parts = line.split()

            # Count rows with the seven numerical XFOIL polar columns
            if len(parts) == 7:

                try:
                    for value in parts:
                        float(value)

                    n += 1

                # Ignore header and non-numerical rows
                except ValueError:
                    pass

    return n


# =============================================================================
# XFOIL execution
# =============================================================================

def run_xfoil_alpha(airfoil_file, airfoil_name, Re, alpha, polar_file):
    """
    Run XFOIL for one airfoil, Reynolds number, and angle of attack.

    Input parameters:
        airfoil_file       : airfoil coordinate file
        airfoil_name       : airfoil filename without extension
        Re                 : Reynolds number [-]
        alpha              : angle of attack [deg]
        polar_file         : XFOIL polar file for this airfoil/Re pair

    XFOIL workflow:
        1. Load and repanel the airfoil geometry.
        2. Apply the viscous solver settings.
        3. Open polar accumulation.
        4. Append to the existing polar file if it already exists.
        5. Try the requested alpha command once.
        6. Close polar accumulation and exit XFOIL.

    Convergence strategy:
        The angle of attack is attempted once. If XFOIL does not converge
        within the allowed runtime, no polar row is written for that alpha.

    Returned value:
        success            : True if XFOIL terminated normally, False otherwise
    """

    # Build the initial XFOIL command script
    commands = f"""
PLOP
G F

LOAD {airfoil_file.as_posix()}
OPER
ITER {MAX_ITER}
VISC {Re}
VPAR
N {Ncrit}

PACC
{polar_file.as_posix()}
{"y\n" if (Re in Reynolds_numbers_err and polar_file.exists()) else ""}
ALFA {alpha}
PACC

QUIT
"""

    try:
        result = subprocess.run(
            [xfoil_exe],
            input=commands,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=alpha_timeout_seconds,
        )

    # Stop this alpha value if XFOIL exceeds the allowed runtime
    except subprocess.TimeoutExpired:
        return False

    # If XFOIL terminated abnormally, report this alpha value as failed
    if result.returncode != 0:
        return False

    return True


def run_xfoil_airfoil(airfoil_file, Re, polar_folder):
    """
    Run the full angle-of-attack sweep for one airfoil and Reynolds number.

    Input parameters:
        airfoil_file       : airfoil coordinate file
        Re                 : Reynolds number [-]
        polar_folder       : directory where the XFOIL polar file is written

    Generated output:
        polar_file         : XFOIL polar file for this airfoil/Re pair

    XFOIL workflow:
        1. Remove any existing polar file for this airfoil/Re pair.
        2. Sweep through all requested angles of attack.
        3. Start a new XFOIL process for each alpha value.
        4. Allow each alpha command to run for at most 2 seconds.
        5. Append successful alpha results directly to the polar file.
        6. Skip alpha values that timeout or fail.

    Convergence strategy:
        Each angle of attack is attempted once. If XFOIL does not converge
        within 2 seconds, that alpha value is skipped and the sweep continues.

    Returned values:
        airfoil_name       : airfoil filename without extension
        Re                 : Reynolds number [-]
        status             : text describing whether the sweep completed
                             fully, partially, or failed
    """

    airfoil_name = airfoil_file.stem
    polar_file = polar_folder / f"{airfoil_name}_Re{int(Re)}.txt"

    # Start from a clean polar file so old results cannot affect the new sweep
    if polar_file.exists():
        polar_file.unlink()

    # Sweep through all requested angles of attack
    for alpha in alphas:

        # Count how many operating points were available before this alpha
        n_before = count_polar_alpha_entries(polar_file)

        # Try this alpha value once with its own timeout
        run_xfoil_alpha(
            airfoil_file,
            airfoil_name,
            Re,
            alpha,
            polar_file,
        )

        # Count how many operating points were written after this alpha
        n_after = count_polar_alpha_entries(polar_file)

        # Continue to the next alpha whether this one converged or failed
        if n_after <= n_before:
            continue

    # Count the number of successful operating points written to the polar file
    n_success = count_polar_alpha_entries(polar_file)

    # Total number of requested alpha values
    n_requested = len(alphas)

    # No operating points converged
    if n_success == 0:
        return airfoil_name, Re, "failed: no alpha values converged"

    # Only part of the requested alpha range converged
    if n_success < n_requested:
        return (
            airfoil_name,
            Re,
            f"finished, {n_success}/{n_requested} alpha values converged",
        )

    return airfoil_name, Re, "finished alpha sweep"


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the full XFOIL polar generation workflow.
    """

    # Ask which propeller folder should be processed
    prop, airfoil_folder = ask_for_propeller_with_profiles(
        APC_DIRECTORY,
        PROP_DIRECTORY,
    )

    # Create the output folder for the XFOIL polar files
    output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "airfoil_polars"
    )

    # Delete existing polar folder and all previous polar files
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Recreate an empty polar output folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all generated airfoil coordinate files
    airfoil_files = sorted(
        airfoil_folder.glob("*.txt")
    )

    n_airfoils = len(airfoil_files)

    # Create one job for each airfoil/Reynolds-number combination
    jobs = [
        (airfoil_file, Re)
        for airfoil_file in airfoil_files
        for Re in Reynolds_numbers
    ]

    n_jobs = len(jobs)

    # Print run settings before starting XFOIL
    print("=" * 70)
    print("Running XFOIL polar generation")
    print(f"Number of airfoils        : {n_airfoils}")
    print(f"Number of jobs            : {n_jobs}")
    print(f"Reynolds numbers          : {Reynolds_numbers[0]} to {Reynolds_numbers[-1]}")
    print(f"Ncrit                     : {Ncrit}")
    print(f"Alpha range               : {alphas[0]} to {alphas[-1]} deg")
    print(f"Alpha step                : {alphas[1] - alphas[0]} deg")
    print(f"Timeout per alpha value   : {alpha_timeout_seconds} s")
    print(f"Parallel workers          : {max_workers}")
    print(f"Output folder             : {output_dir.resolve()}")
    print("=" * 70)
    print()

    if n_airfoils == 0:
        print(f"No .txt airfoil files found in '{airfoil_folder}'.")

    else:

        # Run all airfoil/Reynolds-number jobs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = {
                executor.submit(
                    run_xfoil_airfoil,
                    airfoil_file,
                    Re,
                    output_dir,
                ): (airfoil_file, Re)
                for airfoil_file, Re in jobs
            }

            # Print each job result as soon as it finishes
            for i, future in enumerate(as_completed(futures), start=1):

                airfoil_file, Re = futures[future]

                try:
                    airfoil_name, Re, status = future.result()

                # Report unexpected Python errors without stopping the full run
                except Exception as error:
                    airfoil_name = airfoil_file.stem
                    status = f"failed: Python error: {error}"

                print(
                    f"[{i:03d}/{n_jobs:03d}] "
                    f"{airfoil_name:<25} "
                    f"Re={Re:<7.0f} "
                    f"{status}"
                )

    print()
    print("=" * 70)
    print("All airfoil polar analyses completed.")
    print(f"Polar files saved in: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()