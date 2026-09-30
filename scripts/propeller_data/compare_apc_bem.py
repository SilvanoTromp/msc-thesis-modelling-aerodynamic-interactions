"""
Compare APC performance data against BEM analysis results.

The script:
    1. Lets the user select a fully processed propeller.
    2. Reads APC performance data from the selected APC performance file.
    3. Lets the user enter:
            - flight speed V [m/s]
            - rotational speed RPM
    4. Finds the closest available APC operating point.
    5. Runs BEM analysis at that operating point.
    6. Prints APC data, BEM results, and relative differences.

Compared quantities:
    J           advance ratio [-]
    Pe          propeller efficiency [-]
    Ct          thrust coefficient [-]
    Cp          power coefficient [-]
    Power       propeller shaft power [W]
    Torque      propeller shaft torque [N m]
    Thrust      propeller thrust [N]
    M_tip       blade-tip Mach number [-]
    Re75        Reynolds number at 75% radius [-]
    FoM         figure of merit [-]
"""

import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))
from propeller_data.utilities import *
from BEM.analysis import BEM_analysis

# =============================================================================
# Input settings
# =============================================================================

# Directory containing the APC <propeller>-PERF.PE0 geometry files
APC_GEO_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "external"
    / "apc"
    / "geometry"
)

# Directory that contains the APC performance data files
APC_PERF_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "external"
    / "apc"
    / "performance"
)

# Directory that contains the propeller subfolders
PROP_DIRECTORY = Path(
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed_propellers"
)


# =============================================================================
# Operating-point selection
# =============================================================================

def find_closest_operating_point(data, requested_RPM, requested_V_ms):
    """
    Find the closest available APC operating point.

    User input velocity is given in m/s, while APC files store velocity in mph.
    Therefore, the requested velocity is first converted to mph.

    Selection method:
        1. Pick the available RPM closest to requested_RPM.
        2. Within that RPM block, pick the available V closest to requested V.

    Returns
    -------
    selected_RPM : int
        Closest available APC RPM.

    selected_V_mph : float
        Closest available APC velocity in mph.

    selected_V_ms : float
        Closest available APC velocity in m/s.

    operating_point : dict
        APC data row at the selected operating point.
    """

    requested_V_mph = requested_V_ms / 0.44704

    available_RPMs = sorted(data.keys())

    if not available_RPMs:
        return None, None, None, None

    selected_RPM = min(
        available_RPMs,
        key=lambda rpm: abs(rpm - requested_RPM),
    )

    rows = data[selected_RPM]

    if not rows:
        return selected_RPM, None, None, None

    operating_point = min(
        rows,
        key=lambda row: abs(row["V"] - requested_V_mph),
    )

    selected_V_mph = operating_point["V"]
    selected_V_ms = selected_V_mph * 0.44704

    return selected_RPM, selected_V_mph, selected_V_ms, operating_point


# =============================================================================
# Output writing
# =============================================================================

def print_comparison_table(prop, APC_results, analysis_results, V_ms, RPM, method_name):
    """
    Print APC reference performance data and analysis results side-by-side.

    Input values:
        prop                : selected propeller name
        APC_results         : dictionary containing APC reference results
        analysis_results    : dictionary containing computed analysis results
        V_ms                : flight speed [m/s]
        RPM                 : propeller rotational speed [RPM]
        method_name         : name of the analysis method

    Displayed quantities:
        J                   : advance ratio [-]
        Pe                  : propeller efficiency [-]
        Ct                  : thrust coefficient [-]
        Cp                  : power coefficient [-]
        FoM                 : figure of merit [-]
        Power [W]           : shaft power [W]
        Torque [Nm]         : shaft torque [Nm]
        Thrust [N]          : thrust [N]
        M_tip               : blade-tip Mach number [-]
        Re75                : Reynolds number at 75% radius [-]

    The relative difference is computed as:
        relative_difference = 100 * (analysis - APC) / APC

    If the APC reference value is zero, the relative difference is set to zero
    to avoid division by zero.
    """

    # Define the number of printed decimal places for each quantity
    decimals = {
        "J": 4,
        "Pe": 4,
        "Ct": 4,
        "Cp": 4,
        "FoM": 4,
        "Power [W]": 3,
        "Torque [Nm]": 3,
        "Thrust [N]": 3,
        "M_tip": 2,
        "Re75": 0,
    }

    # Print operating conditions and comparison header
    print("\n" + "=" * 88)
    print(f"APC vs {method_name} COMPARISON | Propeller: APC {prop}")
    print(f"Velocity: {V_ms:.3f} m/s | RPM: {RPM}")
    print("=" * 88)

    # Print column headers
    print(
        f"{'Quantity':<15s}"
        f"{'APC':>18s}"
        f"{method_name:>18s}"
        f"{'Rel. Diff. [%]':>20s}"
    )

    print("-" * 88)

    # Print APC data, computed results, and relative differences
    for key in analysis_results:

        # Select the number of printed decimals for this quantity
        n_decimals = decimals[key]

        apc_value = APC_results[key]
        analysis_value = round(analysis_results[key], n_decimals)

        # Compute relative difference while avoiding division by zero
        if abs(apc_value) < 1e-12:
            diff = 0.0
        else:
            diff = 100 * (analysis_value - apc_value) / apc_value

        print(
            f"{key:<15s}"
            f"{apc_value:>18.{n_decimals}f}"
            f"{analysis_value:>18.{n_decimals}f}"
            f"{diff:>20.2f}"
        )


# =============================================================================
# Main script
# =============================================================================

def main():
    """
    Run the APC-versus-BEM comparison workflow.
    """

    # Ask the user to select a propeller that has all required input files
    prop, APC_GEO_FILE, APC_PERF_FILE = ask_for_processed_propeller(
        APC_GEO_DIRECTORY,
        APC_PERF_DIRECTORY,
        PROP_DIRECTORY,
    )

    # Load APC performance data
    data = parse_APC_perf_file(APC_PERF_FILE)

    # Ask for the operating point to compare
    requested_V_ms = float(input("\nEnter flight speed V [m/s]: "))
    requested_RPM = int(input("Enter RPM: "))
    print()

    # Select the closest operating point available in the APC data
    selected_RPM, selected_V_mph, selected_V_ms, operating_point = (
        find_closest_operating_point(data, requested_RPM, requested_V_ms)
    )

    if operating_point is None:
        print("\nNo usable APC operating point found.")
        return

    # Store APC reference results
    APC_results = {
        "J": operating_point["J"],
        "Pe": operating_point["Pe"],
        "Ct": operating_point["Ct"],
        "Cp": operating_point["Cp"],
        "Power [W]": operating_point["Power_W"],
        "Torque [Nm]": operating_point["Torque_Nm"],
        "Thrust [N]": operating_point["Thrust_N"],
        "M_tip": operating_point["Mach"],
        "Re75": operating_point["Re75"],
        "FoM": operating_point["FoM"],
    }

    # Inform the user if the exact requested operating point was not available
    if selected_RPM != requested_RPM or abs(selected_V_ms - requested_V_ms) > 1e-9:
        print("Requested operating point not found exactly.")
        print("Using closest available APC operating point:")
        print(f"  Requested: V = {requested_V_ms:.3f} m/s, RPM = {requested_RPM}")
        print(f"  Selected : V = {selected_V_ms:.3f} m/s, RPM = {selected_RPM}")
        print()

    # Run BEM analysis at the selected APC operating point
    J, Pe, Ct, Cp, P, Q, T, M_tip, Re75, FoM = BEM_analysis(
        prop,
        selected_V_mph,
        selected_RPM,
    )

    # Store BEM results using the same keys as the APC reference results
    BEM_results = {
        "J": J,
        "Pe": Pe,
        "Ct": Ct,
        "Cp": Cp,
        "Power [W]": P,
        "Torque [Nm]": Q,
        "Thrust [N]": T,
        "M_tip": M_tip,
        "Re75": Re75,
        "FoM": FoM,
    }

    # Print APC data, BEM results, and relative differences
    print_comparison_table(
        prop,
        APC_results,
        BEM_results,
        selected_V_ms,
        selected_RPM,
        "BEM",
    )


if __name__ == "__main__":
    main()