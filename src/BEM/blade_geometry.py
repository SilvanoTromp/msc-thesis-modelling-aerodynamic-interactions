def theta_function(prop, r_R):
    """
    Compute the blade twist angle at a given non-dimensional radial position.

    Parameters:
        prop (string): name of the propeller.
        r_R (float): non-dimensional radial position, where r is the local radius and R is the rotor radius.

    Returns:
        theta (float): blade twist angle [rad] at the given radial position.
    """

    # Import required libraries
    import numpy as np
    from pathlib import Path
    from scipy.interpolate import CubicSpline

    # Load blade geometry data from blade geometry file
    data = np.genfromtxt(
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "processed_propellers"
        / f"apc_{prop}"
        / "bladeGeom.txt",
        comments="#",
        usecols=(0, 4)  # r/R, twist [deg]
    )

    # Extract radial positions and twist distribution
    r_R_data = data[:, 0]
    twist_deg = data[:, 1]

    # Return zero outside blade span
    if r_R < r_R_data.min() or r_R > r_R_data.max():
        return 0.0

    # Interpolate twist angle and convert to radians
    cs = CubicSpline(r_R_data, twist_deg)
    return np.radians(float(cs(r_R)))


def c_function(prop, r_R):
    """
    Compute the blade chord length at a given non-dimensional radial position.

    Parameters:
        prop (string): name of the propeller.
        r_R (float): non-dimensional radial position, where r is the local radius and R is the rotor radius.

    Returns:
        c: chord length at the given radial position [m].
    """

    # Import required libraries
    import numpy as np
    from pathlib import Path
    from scipy.interpolate import CubicSpline

    # Load blade geometry data from blade geometry file
    data = np.genfromtxt(
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "processed_propellers"
        / f"apc_{prop}"
        / "bladeGeom.txt",
        comments="#",
        usecols=(0, 2)  # r/R, chord [m]
    )

    # Extract radial positions and chord distribution
    r_R_data = data[:, 0]
    chord_m_data = data[:, 1]

    # Return zero outside blade span
    if r_R < r_R_data.min() or r_R > r_R_data.max():
        return 0.0

    # Interpolate chord length
    cs = CubicSpline(r_R_data, chord_m_data)
    return float(cs(r_R))