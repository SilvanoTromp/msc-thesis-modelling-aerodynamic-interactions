"""
Average FLIGHTLAB velocity fields and calculate vorticity.

The script:
1. Processes four predefined FLIGHTLAB simulation cases.
2. Searches all velocityfield-step-*.tab files in each case.
3. Reads Cartesian coordinates and velocity components.
4. Combines wind, VVPM-induced, and panel-induced velocities.
5. Averages the total velocity over all saved simulation steps.
6. Reconstructs the structured Cartesian grid.
7. Calculates vorticity from the averaged velocity field.
8. Calculates velocity and vorticity magnitudes.
9. Exports the averaged fields to ParaView-compatible .vtr files.
10. Saves the output in each case's velocityfield folder.

All coordinates are in m, velocities in m/s, and vorticities in 1/s.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# ==============================================================
# Settings
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

FLIGHTLAB_DIR = PROJECT_ROOT / "results" / "FLIGHTLAB"

# Simulation cases to process
CASE_FOLDERS = [
    "V-0_alpha-0_RPMs-6000-6000-1",
    "V-5.10_alpha-6_RPMs-6700-6169-2189",
    "V-10.30-alpha-6_RPMS-6700-5754-3705",
    "V-16.50-alpha-6_RPMS-1-1-5500",
]

# Output filename
OUTPUT_FILENAME = "velocityfield-averaged.vtr"

# Grid coordinate tolerance [m]
GRID_TOLERANCE = 1e-8

# ==============================================================
# FLIGHTLAB TAB reader
# ==============================================================

def read_tab_file(file_path):
    """
    Read numerical blocks from a FLIGHTLAB .tab file.

    Each block starts with:

        !B VARIABLE NROWS NCOLUMNS

    Returns
    -------
    dict
        Numerical arrays indexed by variable name.
    """

    data = {}
    current_variable = None
    current_values = []

    with open(file_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("!B"):

                if current_variable is not None:
                    data[current_variable] = np.asarray(
                        current_values, dtype=float
                    )

                parts = line.split()
                current_variable = parts[1].upper()
                current_values = []
                continue

            # Ignore other FLIGHTLAB metadata lines
            if line.startswith("!"):
                continue

            if current_variable is not None:
                current_values.extend(
                    float(value) for value in line.split()
                )

    if current_variable is not None:
        data[current_variable] = np.asarray(
            current_values, dtype=float
        )

    return data

# ==============================================================
# Extract velocity field
# ==============================================================

def extract_velocity(data):
    """
    Combine wind, VVPM, and panel-induced velocities.

    Returns
    -------
    ndarray
        Total velocity array with shape (N, 3).
    """

    velocity_components = [
        ["WINDX", "VINTFVVPMX", "VINTFPANELX"],
        ["WINDY", "VINTFVVPMY", "VINTFPANELY"],
        ["WINDZ", "VINTFVVPMZ", "VINTFPANELZ"],
    ]

    velocity = []

    for variables in velocity_components:

        # Sum all contributions for this direction
        component = sum(
            (data[name] for name in variables)
        )

        velocity.append(component)

    return np.column_stack(velocity)

# ==============================================================
# Reconstruct Cartesian grid
# ==============================================================

def reconstruct_grid(x, y, z):
    """
    Reconstruct a structured Cartesian grid from point coordinates.

    Grid arrays use shape (nz, ny, nx), with x varying fastest.

    Returns
    -------
    tuple
        Unique coordinate arrays, sorting indices, and grid shape.
    """

    # Round coordinates to remove insignificant floating-point noise
    decimals = int(np.ceil(-np.log10(GRID_TOLERANCE)))

    xr = np.round(x, decimals)
    yr = np.round(y, decimals)
    zr = np.round(z, decimals)

    # Unique grid coordinates
    x_unique = np.unique(xr)
    y_unique = np.unique(yr)
    z_unique = np.unique(zr)

    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)

    npoints = len(x)

    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Total points: {npoints}")

    # Check whether the grid is complete
    if nx * ny * nz != npoints:
        raise ValueError(
            "Point coordinates do not form a complete "
            "Cartesian grid."
        )

    # Sort points with x varying fastest
    sort_indices = np.lexsort((xr, yr, zr))

    # Verify that each Cartesian grid location occurs once
    ix = np.searchsorted(x_unique, xr[sort_indices])
    iy = np.searchsorted(y_unique, yr[sort_indices])
    iz = np.searchsorted(z_unique, zr[sort_indices])

    expected = np.arange(npoints)

    linear_indices = (
        iz * ny * nx + iy * nx + ix
    )

    if not np.array_equal(linear_indices, expected):
        raise ValueError(
            "Duplicate or missing Cartesian grid points detected."
        )

    grid_shape = (nz, ny, nx)

    return (
        x_unique,
        y_unique,
        z_unique,
        sort_indices,
        grid_shape,
    )


# ==============================================================
# Calculate vorticity field
# ==============================================================

def calculate_vorticity(velocity, x, y, z):
    """
    Calculate vorticity from a structured Cartesian velocity field.

    Parameters
    ----------
    velocity : ndarray
        Velocity array with shape (nz, ny, nx, 3).
    x, y, z : ndarray
        Cartesian grid coordinates [m].

    Returns
    -------
    ndarray
        Vorticity array with shape (nz, ny, nx, 3).
    """

    # Extract velocity components
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]

    # Calculate spatial derivatives
    def derivative(field, coordinates, axis):

        if len(coordinates) == 1:
            return np.zeros_like(field)

        edge_order = 2 if len(coordinates) >= 3 else 1

        return np.gradient(
            field,
            coordinates,
            axis=axis,
            edge_order=edge_order,
        )

    # Derivatives required for curl
    dvz_dy = derivative(vz, y, axis=1)
    dvy_dz = derivative(vy, z, axis=0)

    dvx_dz = derivative(vx, z, axis=0)
    dvz_dx = derivative(vz, x, axis=2)

    dvy_dx = derivative(vy, x, axis=2)
    dvx_dy = derivative(vx, y, axis=1)

    # Vorticity components
    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy

    return np.stack(
        [omega_x, omega_y, omega_z],
        axis=-1,
    )


# ==============================================================
# ParaView VTR writer
# ==============================================================

def write_vtr(file_path, x, y, z, velocity, vorticity):
    """
    Export velocity and vorticity to a VTK RectilinearGrid file.

    The file can be opened directly in ParaView.

    Arrays are written as ASCII Float64 data.
    """

    nx = len(x)
    ny = len(y)
    nz = len(z)

    extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"

    # ----------------------------------------------------------
    # Create VTK XML structure
    # ----------------------------------------------------------

    vtk_file = ET.Element(
        "VTKFile",
        type="RectilinearGrid",
        version="1.0",
        byte_order="LittleEndian",
    )

    grid = ET.SubElement(
        vtk_file,
        "RectilinearGrid",
        WholeExtent=extent,
    )

    piece = ET.SubElement(
        grid,
        "Piece",
        Extent=extent,
    )

    point_data = ET.SubElement(
        piece,
        "PointData",
        Vectors="Velocity",
        Scalars="V_magnitude",
    )

    # ----------------------------------------------------------
    # Helper function for writing arrays
    # ----------------------------------------------------------

    def add_array(parent, name, values, ncomponents=1):

        attributes = {
            "type": "Float64",
            "Name": name,
            "format": "ascii",
        }

        if ncomponents > 1:
            attributes["NumberOfComponents"] = str(ncomponents)

        array = ET.SubElement(
            parent,
            "DataArray",
            **attributes,
        )

        # Flatten in VTK point order: x fastest, then y, then z
        flat_values = np.asarray(values).ravel(order="C")

        array.text = " ".join(
            f"{value:.12e}" for value in flat_values
        )

    # ----------------------------------------------------------
    # Velocity variables
    # ----------------------------------------------------------

    velocity_magnitude = np.linalg.norm(
        velocity, axis=-1
    )

    add_array(
        point_data, "Velocity", velocity, ncomponents=3
    )

    add_array(point_data, "Vx", velocity[..., 0])
    add_array(point_data, "Vy", velocity[..., 1])
    add_array(point_data, "Vz", velocity[..., 2])

    add_array(
        point_data, "V_magnitude", velocity_magnitude
    )

    # ----------------------------------------------------------
    # Vorticity variables
    # ----------------------------------------------------------

    vorticity_magnitude = np.linalg.norm(
        vorticity, axis=-1
    )

    add_array(
        point_data, "Vorticity", vorticity, ncomponents=3
    )

    add_array(
        point_data, "Vorticity_x", vorticity[..., 0]
    )

    add_array(
        point_data, "Vorticity_y", vorticity[..., 1]
    )

    add_array(
        point_data, "Vorticity_z", vorticity[..., 2]
    )

    add_array(
        point_data,
        "Vorticity_magnitude",
        vorticity_magnitude,
    )

    # ----------------------------------------------------------
    # Cartesian grid coordinates
    # ----------------------------------------------------------

    coordinates = ET.SubElement(
        piece, "Coordinates"
    )

    add_array(coordinates, "XCoordinates", x)
    add_array(coordinates, "YCoordinates", y)
    add_array(coordinates, "ZCoordinates", z)

    # ----------------------------------------------------------
    # Save VTR file
    # ----------------------------------------------------------

    tree = ET.ElementTree(vtk_file)

    tree.write(
        file_path,
        encoding="utf-8",
        xml_declaration=True,
    )


# ==============================================================
# Process individual simulation
# ==============================================================

def process_simulation(case_dir):
    """
    Average all velocityfield snapshots and export ParaView file.
    """

    velocity_dir = case_dir / "velocityfield"

    if not velocity_dir.is_dir():
        print(f"Skipping {case_dir.name}: velocityfield not found.")
        return

    # ----------------------------------------------------------
    # Find velocityfield snapshots
    # ----------------------------------------------------------

    velocity_files = sorted(
        velocity_dir.glob("velocityfield-step-*.tab"),
        key=lambda file: int(file.stem.split("-")[-1]),
    )

    if not velocity_files:
        print(f"Skipping {case_dir.name}: no velocityfield files.")
        return

    print(f"\nProcessing: {case_dir.name}")
    print(f"Found {len(velocity_files)} velocityfield snapshots.")

    # ----------------------------------------------------------
    # Initialize averaging variables
    # ----------------------------------------------------------

    velocity_sum = None
    reference_coordinates = None

    # ----------------------------------------------------------
    # Read and accumulate velocity fields
    # ----------------------------------------------------------

    for i, file_path in enumerate(velocity_files):

        print(
            f"Reading snapshot {i+1}/{len(velocity_files)}: "
            f"{file_path.name}"
        )

        data = read_tab_file(file_path)

        # Extract Cartesian coordinates
        coordinates = np.column_stack([
            data["XPTS"],
            data["YPTS"],
            data["ZPTS"],
        ])

        # Extract total velocity
        velocity = extract_velocity(data)

        # Check coordinate consistency
        if reference_coordinates is None:

            reference_coordinates = coordinates

            velocity_sum = np.zeros_like(
                velocity, dtype=np.float64
            )

        else:

            if coordinates.shape != reference_coordinates.shape:
                raise ValueError(
                    f"Grid size mismatch in {file_path.name}"
                )

            if not np.allclose(
                coordinates,
                reference_coordinates,
                rtol=0,
                atol=GRID_TOLERANCE,
            ):
                raise ValueError(
                    f"Grid coordinates mismatch in {file_path.name}"
                )

        # Accumulate velocity components
        velocity_sum += velocity

    # ----------------------------------------------------------
    # Calculate averaged velocity field
    # ----------------------------------------------------------

    n_snapshots = len(velocity_files)

    velocity_mean = velocity_sum / n_snapshots

    print(f"\nAveraged {n_snapshots} velocity snapshots.")

    # ----------------------------------------------------------
    # Reconstruct Cartesian grid
    # ----------------------------------------------------------

    x = reference_coordinates[:, 0]
    y = reference_coordinates[:, 1]
    z = reference_coordinates[:, 2]

    (
        x_unique,
        y_unique,
        z_unique,
        sort_indices,
        grid_shape,
    ) = reconstruct_grid(x, y, z)

    # Sort averaged velocities into Cartesian grid order
    velocity_grid = velocity_mean[sort_indices].reshape(
        (*grid_shape, 3)
    )

    # ----------------------------------------------------------
    # Calculate vorticity field
    # ----------------------------------------------------------

    print("Calculating vorticity field...")

    vorticity_grid = calculate_vorticity(
        velocity_grid,
        x_unique,
        y_unique,
        z_unique,
    )

    # ----------------------------------------------------------
    # Save ParaView file
    # ----------------------------------------------------------

    output_file = velocity_dir / OUTPUT_FILENAME

    print("\nWriting ParaView file...")

    write_vtr(
        output_file,
        x_unique,
        y_unique,
        z_unique,
        velocity_grid,
        vorticity_grid,
    )

    print(f"Saved: {output_file}")

# ==============================================================
# Main execution
# ==============================================================

def main():

    if not FLIGHTLAB_DIR.is_dir():
        raise FileNotFoundError(
            f"FLIGHTLAB results directory not found: {FLIGHTLAB_DIR}"
        )

    print(f"Processing {len(CASE_FOLDERS)} FLIGHTLAB cases.")

    # Process each predefined simulation
    for case_name in CASE_FOLDERS:

        case_dir = FLIGHTLAB_DIR / case_name

        if not case_dir.is_dir():
            print(f"\nSkipping: folder not found: {case_name}")
            continue

        try:
            process_simulation(case_dir)

        except (
            ValueError,
            KeyError,
            OSError,
            IndexError,
        ) as error:

            print(
                f"ERROR processing {case_name}: {error}"
            )

    print("\nFinished processing FLIGHTLAB velocity fields.")

if __name__ == "__main__":
    main()