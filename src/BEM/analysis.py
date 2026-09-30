def BEM_analysis(prop, Uinf_mph, RPM):
    """
    Perform BEM analysis for propeller.

    Parameters:
        prop (string): name of the propeller.
        Uinf_mph (float): flow speed far upstream [mph].
        RPM (float): rotor revolutions per minute [-].

    Returns:
        Pe (float): propeller efficiency [-].
        Ct (float): thrust coefficient [-].
        Cp (float): power coefficient [-].
        P (float): propeller power [W].
        Q (float): propeller torque [N⋅m].
        T (float): propeller thrust [N].
        M (float): propeller tip Mach number [-].
        Re75 (float): Reynolds number at 75% of span.
    """

    # Import required functions and libraries
    import numpy as np
    from pathlib import Path
    from BEM.blade_geometry import theta_function, c_function
    from BEM.streamtube_solver import solve_streamtube
    from BEM.polar_data import load_polars_once

    # Path to geometry file
    geom_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "bladeGeom.txt"
    )

    # Path to APC file
    APC_file = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "external"
            / "apc"
            / "geometry"
            / f"{prop}-PERF.PE0"
    )

    # Extract radius and blade count
    with open(APC_file, "r") as f:
        for line in f:
            if line.startswith(" RADIUS:"):
                R = float(line.split()[1]) * 0.0254

            if line.startswith(" BLADES:"):
                B = int(line.split()[1])

    # Load non-dimensional radial blade stations (r/R)
    r_R = np.genfromtxt(
        geom_path,
        comments="#",
        usecols=(0, 1)  # r/R, chord [m]
    )[:, 0]

    # Geometric specifications of propeller
    D = 2 * R                           # rotor diameter [m]
    A = np.pi * R**2                    # rotor area [m2]
    r_R_hub = r_R[0]                    # non-dimensional radial location of blade hub [-]
    r_R_tip = r_R[-1]                   # non-dimensional radial location of blade tip [-]

    # ISA sea-level atmospheric properties
    R_air = 287.05                      # specific gas constant air [J/(kg⋅K)]
    T = 294                             # flow temperature [K]
    rho = 1.225                         # flow density [kg/m3]
    mu = 1.8e-5                         # flow dynamic viscosity [Pa⋅s]
    gamma = 1.4                         # specific heat capacity

    # Speed of sound from ideal gas relation [m/s]
    Vsound = np.sqrt(gamma * R_air * T)

    # Convert RPM to rad/s
    omega = RPM * 2 * np.pi / 60

    # Convert Uinf to m/s
    Uinf = Uinf_mph * 0.44704

    # Convert RPM to rev/s
    n = RPM / 60

    # Compute propeller advance ratio
    J = Uinf / (n * D)

    # Initialize dictionary to store results
    results = {
        "c" : [],
        "theta" : [],
        "vi" : [],
        "vtheta" : [],
        "r_R": [],
        "r" : [],
        "dr_R" : [],
        "dr" : [],
        "W" : [],
        "phi": [],
        "alfa": [],
        "dT" : [],
        "dQ" : []
    }

    # Load propeller polar data
    polar_database = load_polars_once(prop)

    # Solve the BEM model for each blade element
    vi_init = 0.0
    vtheta_init = 0.0

    for i in range(int((len(r_R)-1))):
        element_results = solve_streamtube(prop, Uinf, omega, vi_init, vtheta_init,
                                           rho, mu, Vsound,
                                           r_R[i], r_R[1+i], r_R_hub, r_R_tip,
                                           R, B, c_function, theta_function,
                                           polar_database)
        for result in results:
            results[result].append(element_results[result])

        # Use converged values as initial guess for next blade element
        vi_init = element_results["vi"]
        vtheta_init = element_results["vtheta"]

    # Convert lists to numpy arrays
    for result in results:
        results[result] = np.array(results[result])

    # Confirm completion of the BEM analysis
    print(f"\rBEM analysis completed for Uinf = {Uinf:.2f} m/s | RPM = {RPM:.0f}.")

    # Sum sectional blade loads to obtain total thrust, torque, and power
    T = np.sum(results["dT"])          # thrust [N]
    Q = np.sum(results["dQ"])          # torque [N⋅m]
    P = Q * omega                      # power [W]

    # Compute thrust and power coefficients
    Ct = T / (rho * n**2 * D**4)
    Cp = P / (rho * n**3 * D**5)

    # Compute propeller efficiency
    if abs(Uinf) > 1e-12:
        Pe = Ct * J / Cp
    else:
        Pe = 0.0

    # Compute tip Mach number
    W_tip = results["W"][-1]
    M_tip = W_tip / Vsound

    # Compute Reynolds number at 75% span
    W_75 = np.interp(0.75, results["r_R"], results["W"])
    c_75 = c_function(prop, 0.75)
    Re_75 = rho * W_75 * c_75 / mu

    # Compute ideal power required to generate static thrust
    Pideal = T**(3/2) / np.sqrt(2 * rho * A)

    # Compute Figure of Merit
    FOM = Pideal / P

    return J, Pe, Ct, Cp, P, Q, T, M_tip, Re_75, FOM