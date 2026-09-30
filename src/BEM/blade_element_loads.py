def calc_load_blade_element(prop, r_R, r_R1, r_R2, c_function,
                            W, phi, alfa, rho, mu, Vsound,
                            polar_database):
    """
    Calculate aerodynamic loads and circulation on a blade element.

    Parameters:
        prop (string): name of the propeller.
        r_R: non-dimensional radial midposition of the blade element.
        r_R1 (float): non-dimensional radial start position of the blade element.
        r_R2 (float): non-dimensional radial end position of the blade element.
        c_function (function): function returning chord length [m] at a given radial position r_R.
        W (float): total velocity perceived by the blade element [m/s]
        phi (float): wind inflow angle [rad].
        alfa (float): angle of attack [deg].
        rho (float): flow density [kg/m3].
        mu (float): flow dynamic viscosity [Pa⋅s].
        Vsound (float): speed of sound [m/s].
        polar_database (dict): dictionary containing all propeller polar data.

    Returns:
        cn (float): normal force coefficient [-].
        ct (float): tangential force coefficient [-].
        Nprime (float): normal force per unit span [N/m].
        Tprime (float): tangential force per unit span [N/m].
    """

    # Import required functions and libraries
    import numpy as np
    from BEM.airfoil_coefficients import get_ClCd_polar

    # Determine local flow conditions
    M = W / Vsound                              # local Mach number [-]
    c = c_function(prop, r_R)                   # local chord length [m]
    Re = rho * W * c / mu                       # local Reynolds number [-]

    # Obtain sectional lift and drag coefficients from polar data
    cl, cd = get_ClCd_polar(polar_database, alfa, r_R, Re)

    # Apply Prandtl-Glauert correction for subsonic compressibility
    Mcrit = 0.7
    if M < Mcrit:
        cl = cl / np.sqrt(1 - M**2)
    else:
        cl = cl / np.sqrt(1 - Mcrit**2)

    # Aerodynamic force calculations
    c = c_function(prop, r_R)                   # local chord length [m]
    cn = cl * np.cos(phi) - cd * np.sin(phi)    # normal force coefficient [-]
    ct = cl * np.sin(phi) + cd * np.cos(phi)    # tangential force coefficient [-]
    q = 0.5 * rho * W**2                        # dynamic pressure [-]
    Nprime = cn * q * c                         # normal force per unit span [N/m]
    Tprime = ct * q * c                         # tangential force per unit span [N/m]

    return cn, ct, Nprime, Tprime