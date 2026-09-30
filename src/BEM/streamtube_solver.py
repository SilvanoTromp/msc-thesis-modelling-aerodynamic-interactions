def solve_streamtube(prop, Uinf, omega, vi_init, vtheta_init,
                     rho, mu, Vsound,
                     r_R1, r_R2, r_R_hub, r_R_tip,
                     R, B, c_function, theta_function,
                     polar_database):
    """
    Solve the momentum balance in a blade element streamtube segment using BEM theory.

    Parameters:
        prop (string): name of the propeller.
        Uinf (float): flow speed far upstream [m/s].
        omega (float): rotor rotational speed [rad/s].
        vi_init (float): initial guess for axial induced velocity [m/s].
        vtheta_init (float): initial guess for tangential induced velocity [m/s].
        rho (float): flow density [kg/m3].
        mu (float): flow dynamic viscosity [Pa⋅s].
        Vsound (float): speed of sound [m/s].
        r_R1 (float): non-dimensional radial start position of the blade element.
        r_R2 (float): non-dimensional radial end position of the blade element.
        r_R_hub (float): non-dimensional radial position of blade hub.
        r_R_tip (float): non-dimensional radial position of blade tip.
        R (float): rotor radius [m].
        B (int): number of blades.
        c_function (function): function returning chord length [m] at a given radial position r_R.
        theta_function (function): function returning twist angle [rad] at a given radial position r_R.
        polar_database (dict): dictionary containing all propeller polar data.

    Returns:
        results (dict): dictionary containing results of each blade element:
            c (float): local chord length [m].
            theta (float): local blade twist angle [rad].
            vi (float): axial induced velocity [m/s].
            vtheta (float): tangential induced velocity [m/s].
            r_R (float): non-dimensional radial position of centroid.
            r (float): radial position of centroid [m].
            dr_R (float): non-dimensional radial step size.
            dr (float): radial step size [m].
            Vx (float): axial velocity [m/s].
            Vy (float): tangential velocity [m/s].
            W (float): total velocity perceived by the blade element [m/s].
            phi (float): inflow angle [rad].
            alfa (float): angle of attack [deg].
            dT (float): blade element thrust [N].
            dQ (float): blade element torque [N⋅m].
    """

    # Import required functions and libraries
    import numpy as np
    from BEM.velocity_decomposition import calc_velocity_components
    from BEM.blade_element_loads import calc_load_blade_element
    from BEM.prandtl_losses import calc_Prandtl_correction

    # Calculate geometric and positional properties of blade element
    r_R = (r_R1 + r_R2) / 2             # non-dimensional radial position of blade element center [-]
    r = r_R * R                         # radial position of blade element center [m]
    dr_R = r_R2 - r_R1                  # non-dimensional blade element width [-]
    dr = dr_R * R                       # blade element width [m]

    # Compute local propeller chord length [m]
    c = c_function(prop, r_R)

    # Compute local propeller twist angle [rad]
    theta = theta_function(prop, r_R)

    # Initialize
    vi = vi_init                            # axial induced velocity [m/s]
    vtheta = vtheta_init                    # tangential induced velocity [m/s]
    relax = 0.20                            # relaxation factor [-]
    relax_min = 0.025                       # minimum relaxation factor [-]
    converged = False                       # flag to track convergence status
    iteration_count = 1                     # counter to track number of iterations performed
    max_iter = 1000                         # maximum number of iterations

    # Convergence criterion
    tol = 1e-3

    # Limits used as numerical safeguards
    vi_max, vtheta_max = abs(omega) * R, abs(omega) * r

    while not converged:
        # Monitor progress
        print(f"\rSolving (BEM): V = {Uinf:.3f} m/s | RPM = {omega*60/(2*np.pi):.0f} | "
              f"r/R = {np.round(r_R, 5)} | "
              f"iteration = {iteration_count}...", end="")

        # Calculate velocities and wind inflow angle perceived by the blade element [m/s]
        Vx, Vy, W, phi = calc_velocity_components(Uinf, omega, r, vi, vtheta)

        # Calculate angle of attack at blade element [deg]
        alfa = float(np.degrees(theta - phi))

        # Compute Prandtl's tip/hub loss correction factor
        F = calc_Prandtl_correction(r, r_R_hub * R, r_R_tip * R, phi, B)

        # Calculate loads and circulating strength on blade element
        cn, ct, Nprime, Tprime = calc_load_blade_element(prop, r_R, r_R1, r_R2, c_function,
                                                         W, phi, alfa, rho, mu, Vsound,
                                                         polar_database)

        # Calculate blade element thrust [N] and torque [N⋅m]
        dT = B * Nprime * dr
        dQ = B * r * Tprime * dr

        # Compute new axial induced velocity [m/s]
        # dT = 4*pi*rho*F*r*vi*(Uinf + vi)*dr
        vi_new = 0.5 * (-Uinf + np.sqrt(Uinf**2 + 4.0 * dT / max(4.0 * np.pi * rho * F * r * dr, 1e-12)))
        vi_new = np.clip(vi_new, 0.0, vi_max)

        # Compute new tangential induced velocity [m/s]
        # dQ = 4*pi*rho*F*r^2*(Uinf + vi)*vtheta*dr
        vtheta_new = dQ / (4.0 * np.pi * rho * F * r**2 * max(abs(Uinf + vi_new), 1e-6) * dr)
        vtheta_new = np.clip(vtheta_new, 0.0, vtheta_max)

        # Compute residual before relaxation
        diff_vi = abs(vi_new - vi)
        diff_vtheta = abs(vtheta_new - vtheta)

        # Adaptive tolerance
        tol_vi = tol * max(1.0, abs(vi_new))
        tol_vtheta = tol * max(1.0, abs(vtheta_new))

        # Check convergence
        if diff_vi < tol_vi and diff_vtheta < tol_vtheta:
            vi = vi_new
            vtheta = vtheta_new
            break

        # Adaptive relaxation reduction
        if iteration_count in [max_iter // 4, max_iter // 2, 3 * max_iter // 4]:
            relax = max(relax / 2, relax_min)

        # Under-relaxed induced velocity update
        vi = (1.0 - relax) * vi + relax * vi_new
        vtheta = (1.0 - relax) * vtheta + relax * vtheta_new

        # Update iteration counter
        iteration_count += 1

        # Failure handling
        if iteration_count >= max_iter:
            if diff_vi > tol or diff_vtheta> tol:
                print(
                    f"\nWarning: BEM not fully converged at r/R={r_R:.4f}. "
                    f"diff_vi={diff_vi:.3e}, diff_vtheta={diff_vtheta:.3e}"
                )
            break

    return {
        "c" : c,
        "theta" : theta,
        "vi" : vi,
        "vtheta" : vtheta,
        "r_R" : r_R,
        "r" : r,
        "dr_R" : dr_R,
        "dr" : dr,
        "Vx" : Vx,
        "Vy" : Vy,
        "W" : W,
        "phi" : np.degrees(phi),
        "alfa" : alfa,
        "dT" : dT,
        "dQ" : dQ,
        }