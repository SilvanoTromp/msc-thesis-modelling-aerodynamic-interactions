def calc_velocity_components(Uinf, omega, r, vi, vtheta):
    """
    Calculate the velocity components and inflow angle at a blade element.

    Parameters:
        Uinf (float): flow speed far upstream [m/s].
        omega (float): rotor rotational velocity [rad/s].
        r (float): radial position of blade element.
        vi (float): axial induced velocity at the rotor disk [m/s].
        vtheta (float): tangential induced velocity at the rotor disk [m/s].

    Returns:
        Vx (float): axial velocity [m/s].
        Vy (float): tangential velocity [m/s].
        W (float): total velocity perceived by the blade element [m/s].
        phi (float): wind inflow angle at the blade element [rad].
    """

    # Import required libraries
    import numpy as np

    # Compute axial velocity perceived by rotor [m/s]
    Vx = Uinf + vi

    # Compute tangential velocity perceived by rotor [m/s]
    Vy = omega * r - vtheta

    # Calculate velocity perceived by blade element [m/s]
    W = np.sqrt(Vx**2 + Vy**2)

    # Calculate perceived wind inflow angle at blade element [rad]
    phi = np.arctan2(Vx, Vy)

    return Vx, Vy, W, phi