def get_ClCd_polar(polar_database, target_alfa, r_R, Re):
    """
    Interpolate lift and drag coefficients from loaded airfoil polar data.

    Interpolation is performed sequentially in:
        1. angle of attack alpha
        2. Reynolds number Re
        3. radial position r/R

    Parameters:
        polar_database (dict): dictionary containing all propeller polar data.
        target_alfa (float): angle of attack [deg].
        r_R (float): non-dimensional radial position.
        Re (float): Reynolds number [-].

    Returns:
        Cl (float): lift coefficient [-].
        Cd (float): drag coefficient [-].
    """

    # Import required functions and libraries
    import numpy as np
    from scipy.interpolate import CubicSpline

    def interpolate_alpha(selected_r_R, selected_Re):
        """
        Interpolate Cl and Cd in angle of attack alpha.
        """

        # Read polar data
        polar = polar_database[(selected_r_R, selected_Re)]

        # Read available angles of attack
        available_alpha = polar["alpha"]

        # Read lift and drag coefficient arrays
        Cl_array = polar["Cl"]
        Cd_array = polar["Cd"]

        # Clamp angle of attack to available range
        alpha_used = np.clip(target_alfa, available_alpha.min(), available_alpha.max())

        # Interpolate lift coefficient linearly in alpha
        Cl = np.interp(alpha_used, available_alpha, Cl_array)

        # Interpolate drag coefficient linearly in alpha
        Cd = np.interp(alpha_used, available_alpha, Cd_array)

        return Cl, Cd

    def interpolate_Re(selected_r_R):
        """
        Interpolate Cl and Cd in Reynolds number for one radial station.
        """

        # Determine available Reynolds numbers for current radial station
        available_Re = np.array(sorted([
            key[1]
            for key in polar_database.keys()
            if np.isclose(key[0], selected_r_R, atol=1e-8)
        ]))

        # Check whether polar files exist for this radial station
        if available_Re.size == 0:
            raise FileNotFoundError(
                f"No polar files found for r/R = {selected_r_R}"
            )

        # Clamp Reynolds number to available range
        Re_used = np.clip(Re, available_Re.min(), available_Re.max())

        # Determine lower and upper Reynolds numbers
        Re_low = available_Re[available_Re <= Re_used][-1]
        Re_high = available_Re[available_Re >= Re_used][0]

        # Interpolate in alpha for lower Reynolds number
        Cl_low, Cd_low = interpolate_alpha(selected_r_R, Re_low)

        # Interpolate in alpha for upper Reynolds number
        Cl_high, Cd_high = interpolate_alpha(selected_r_R, Re_high)

        # Check whether lower and upper Reynolds numbers are equal
        if np.isclose(Re_low, Re_high, rtol=0, atol=1e-8):

            # Use exact Reynolds number result directly
            Cl = Cl_low
            Cd = Cd_low

        else:

            # Interpolate lift and drag coefficients linearly in Re
            w_Re = (Re_used - Re_low) / (Re_high - Re_low)
            Cl = (1 - w_Re) * Cl_low + w_Re * Cl_high
            Cd = (1 - w_Re) * Cd_low + w_Re * Cd_high

        return Cl, Cd

    # Determine all available radial stations
    available_r_R = np.array(sorted(set(key[0] for key in polar_database.keys())))

    # Clamp radial position to available range
    r_used = np.clip(r_R, available_r_R.min(), available_r_R.max())

    # Evaluate Cl, Cd at every available radial station
    Cl_values = []
    Cd_values = []

    for station in available_r_R:

        # Interpolate in Reynolds number at current station
        Cl_station, Cd_station = interpolate_Re(station)

        # Store interpolated values
        Cl_values.append(Cl_station)
        Cd_values.append(Cd_station)

    # Convert lists to numpy arrays
    Cl_values = np.array(Cl_values)
    Cd_values = np.array(Cd_values)

    # Cubic spline interpolation in r/R
    Cl_spline = CubicSpline(available_r_R, Cl_values, bc_type="natural")
    Cd_spline = CubicSpline(available_r_R, Cd_values, bc_type="natural")

    # Evaluate cubic splines at requested radial position
    Cl = Cl_spline(r_used)
    Cd = Cd_spline(r_used)

    return float(Cl), float(Cd)