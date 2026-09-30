def calc_Prandtl_correction(r, r_hub, R, phi, B):
    """
    Computes Prandtl's tip and hub loss correction factors.

    Parameters:
        r (float): local blade radius [m]
        r_hub (float): blade hub radius [m].
        R (float): blade radius [m].
        phi (float): wind inflow angle at the blade element [rad].
        B (int): number of blades.

    Returns:
        F (float): total Prandtl correction factor.
    """

    # Import required libraries
    import numpy as np
    import warnings
    warnings.simplefilter('ignore')

    # Calculate tip loss factor
    ftip = B / 2 * (R - r) / (r * abs(np.sin(phi)))
    Ftip = 2 / np.pi * np.arccos(np.exp(-ftip))

    # Calculate hub loss factor
    fhub = B / 2 * (r - r_hub) / (r_hub * abs(np.sin(phi)))
    Fhub = 2 / np.pi * np.arccos(np.exp(-fhub))

    # Calculate total loss factor
    F = Ftip * Fhub

    # Avoid division by zero
    if F < 1e-4 or np.isnan(F):
        F = 1e-4

    return F