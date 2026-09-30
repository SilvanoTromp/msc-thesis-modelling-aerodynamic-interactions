def load_polars_once(prop):
    """
    Load all extrapolated airfoil polar files once and store them in memory.

    Parameters:
        prop (string): name of the propeller.

    Returns:
        polar_database (dict): dictionary containing all polar data.
    """

    # Import required functions and libraries
    import os
    import re
    import numpy as np
    from pathlib import Path

    # Define polar directory
    polar_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed_propellers"
            / f"apc_{prop}"
            / "airfoil_polars_extrapolated"
    )

    # Read all files in the polar directory
    files = os.listdir(polar_dir)

    # Initialize empty polar database
    polar_database = {}

    # Loop over all files
    for fname in files:

        # Extract r/R and Reynolds number from filename
        match = re.match(r"Airfoil_rR([0-9.]+)_Re([0-9]+)", fname)

        if match:

            # Read values from filename
            file_r_R = float(match.group(1))
            file_Re = float(match.group(2))

            # Construct full file path
            filepath = polar_dir / fname

            # Initialize empty data array
            data = []

            # Open polar file
            with open(filepath, "r") as f:

                # Read file line-by-line
                for line in f:

                    # Split line into columns
                    parts = line.split()

                    # Check whether line contains aerodynamic data
                    if len(parts) == 4:

                        try:

                            # Read aerodynamic coefficients
                            alpha = float(parts[0])
                            Cl = float(parts[1])
                            Cd = float(parts[2])
                            Cm = float(parts[3])

                            # Store row
                            data.append([alpha, Cl, Cd, Cm])

                        except ValueError:
                            pass

            # Convert data into numpy array
            data = np.array(data)

            # Check whether file contained valid data
            if data.size == 0:
                continue

            # Sort data by angle of attack
            sort_idx = np.argsort(data[:, 0])
            data = data[sort_idx, :]

            # Store polar data in database
            polar_database[(file_r_R, file_Re)] = {
                "alpha": data[:, 0],
                "Cl": data[:, 1],
                "Cd": data[:, 2],
                "Cm": data[:, 3],
            }

    # Check whether any polar files were found
    if not polar_database:
        raise FileNotFoundError("No polar files found.")

    return polar_database