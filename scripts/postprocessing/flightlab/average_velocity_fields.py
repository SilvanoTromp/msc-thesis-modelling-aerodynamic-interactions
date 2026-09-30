"""
Average FLIGHTLAB velocity-field results and export them to ParaView.

For each FLIGHTLAB test condition, the script:
1. Locates all velocityfield-step-*.tab files.
2. Reads the Cartesian coordinates and velocity components.
3. Verifies that the velocity-field grid is identical between steps.
4. Time-averages VX, VY, and VZ over all sampled steps.
5. Converts the averaged field to a structured Cartesian grid.
6. Writes the averaged velocity field as a ParaView-ready .vtr file.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


# ==============================================================
# Settings
# ==============================================================

# FLIGHTLAB results directory
RESULTS_DIRECTORY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "results"
    / "flightlab"
)

# Name of the velocity-field subdirectory inside each test case
VELOCITYFIELD_DIRECTORY = "velocityfield"

# Velocity-field filename pattern
VELOCITYFIELD_PATTERN = "velocityfield-step-*.tab"

# Name of the averaged ParaView output file
OUTPUT_FILENAME = "average-velocity-field.vtr"


# ==============================================================
# FLIGHTLAB .tab reader
# ==============================================================

def read_flightlab_tab(file_path):
    """
    Read selected arrays and scalar values from a FLIGHTLAB .tab file.

    Parameters
    ----------
    file_path : Path
        Path to the FLIGHTLAB velocity-field file.

    Returns
    -------
    data : dict
        Dictionary containing the coordinates, velocity components,
        and relevant test-condition information.
    """

    required_variables = {
        "CASEVELOCITY",
        "CASEALPHA",
        "CASERPM1",
        "CASERPM4",
        "CASERPM5",
        "XPTS",
        "YPTS",
        "ZPTS",
        "VX",
        "VY",
        "VZ",
    }

    data = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

    line_index = 0

    while line_index < len(lines):

        line = lines[line_index].strip()

        # FLIGHTLAB data blocks start with "!B"
        if not line.startswith("!B"):
            line_index += 1
            continue

        header = line.split()

        if len(header) < 4:
            line_index += 1
            continue

        variable_name = header[1]

        try:
            n_rows = int(header[2])
            n_columns = int(header[3])

        except ValueError:
            line_index += 1
            continue

        number_of_values = n_rows * n_columns

        line_index += 1

        values = []

        # Read all numerical values belonging to the current block
        while (
            line_index < len(lines)
            and len(values) < number_of_values
        ):

            current_line = lines[line_index].strip()

            # Stop if another FLIGHTLAB block unexpectedly starts
            if current_line.startswith("!B"):
                break

            if current_line and not current_line.startswith("#"):

                try:
                    values.extend(
                        float(value)
                        for value in current_line.split()
                    )

                except ValueError:
                    pass

            line_index += 1

        if variable_name in required_variables:

            values = np.asarray(
                values[:number_of_values],
                dtype=float,
            )

            # Store single-value quantities as scalars
            if number_of_values == 1:
                data[variable_name] = values[0]

            else:
                data[variable_name] = values

    # Check whether all required quantities were found
    missing_variables = required_variables - data.keys()

    if missing_variables:

        missing = ", ".join(sorted(missing_variables))

        raise ValueError(
            f"Missing variables in {file_path.name}: {missing}"
        )

    return data


# ==============================================================
# Velocity-field averaging
# ==============================================================

def average_velocity_fields(velocity_files):
    """
    Average all FLIGHTLAB velocity-field files for one test condition.

    Parameters
    ----------
    velocity_files : list[Path]
        Velocity-field files belonging to one test condition.

    Returns
    -------
    averaged_data : dict
        Coordinates, averaged velocity components, and test-condition
        information.
    """

    if not velocity_files:
        raise ValueError("No velocity-field files were provided.")

    # Read first file as reference
    reference_data = read_flightlab_tab(
        velocity_files[0]
    )

    x = reference_data["XPTS"]
    y = reference_data["YPTS"]
    z = reference_data["ZPTS"]

    # Initialize accumulated velocity components
    vx_sum = np.zeros_like(
        reference_data["VX"],
        dtype=float,
    )

    vy_sum = np.zeros_like(
        reference_data["VY"],
        dtype=float,
    )

    vz_sum = np.zeros_like(
        reference_data["VZ"],
        dtype=float,
    )

    print(
        f"    Number of sampled steps: "
        f"{len(velocity_files)}"
    )

    for file_number, file_path in enumerate(
        velocity_files,
        start=1,
    ):

        print(
            f"\r    Reading step "
            f"{file_number}/{len(velocity_files)}",
            end="",
            flush=True,
        )

        data = read_flightlab_tab(file_path)

        # Check that all steps use the same velocity-field coordinates
        if not (
            np.allclose(data["XPTS"], x)
            and np.allclose(data["YPTS"], y)
            and np.allclose(data["ZPTS"], z)
        ):
            raise ValueError(
                f"\nVelocity-field grid in {file_path.name} "
                "does not match the reference grid."
            )

        vx_sum += data["VX"]
        vy_sum += data["VY"]
        vz_sum += data["VZ"]

    print()

    number_of_steps = len(velocity_files)

    averaged_data = {
        "XPTS": x,
        "YPTS": y,
        "ZPTS": z,
        "VX": vx_sum / number_of_steps,
        "VY": vy_sum / number_of_steps,
        "VZ": vz_sum / number_of_steps,
        "CASEVELOCITY": reference_data["CASEVELOCITY"],
        "CASEALPHA": reference_data["CASEALPHA"],
        "CASERPM1": reference_data["CASERPM1"],
        "CASERPM4": reference_data["CASERPM4"],
        "CASERPM5": reference_data["CASERPM5"],
    }

    return averaged_data


# ==============================================================
# Convert FLIGHTLAB point ordering to Cartesian grid
# ==============================================================

def create_rectilinear_grid(data):
    """
    Convert the FLIGHTLAB point arrays to VTK rectilinear-grid ordering.

    VTK rectilinear grids are defined by unique x, y, and z coordinate
    arrays. Point data must be ordered with x varying fastest, followed
    by y and then z.

    Parameters
    ----------
    data : dict
        Averaged FLIGHTLAB velocity-field data.

    Returns
    -------
    grid : dict
        Rectilinear-grid coordinates and reordered velocity components.
    """

    x = data["XPTS"]
    y = data["YPTS"]
    z = data["ZPTS"]

    vx = data["VX"]
    vy = data["VY"]
    vz = data["VZ"]

    # Determine the unique Cartesian coordinates
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    z_unique = np.unique(z)

    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)

    number_of_points = len(x)
    expected_points = nx * ny * nz

    print(
        f"    Grid dimensions: "
        f"{nx} x {ny} x {nz}"
    )

    print(
        f"    Number of grid points: "
        f"{number_of_points}"
    )

    # A VTK rectilinear grid requires every possible Cartesian
    # combination of x, y, and z to be present.
    if number_of_points != expected_points:

        raise ValueError(
            "Velocity field does not form a complete Cartesian "
            "rectilinear grid.\n"
            f"Found {number_of_points} points, but "
            f"{nx} x {ny} x {nz} = {expected_points} "
            "points are required."
        )

    # Determine the Cartesian-grid index of every FLIGHTLAB point
    ix = np.searchsorted(x_unique, x)
    iy = np.searchsorted(y_unique, y)
    iz = np.searchsorted(z_unique, z)

    # Convert (ix, iy, iz) to VTK's linear point ordering.
    # VTK expects x to vary fastest, then y, then z.
    vtk_index = (
        ix
        + nx * iy
        + nx * ny * iz
    )

    # Verify that every Cartesian-grid location occurs exactly once
    if len(np.unique(vtk_index)) != number_of_points:

        raise ValueError(
            "Duplicate or missing Cartesian grid locations "
            "were detected in the FLIGHTLAB velocity field."
        )

    # Reorder the velocity components into VTK ordering
    order = np.argsort(vtk_index)

    vx_vtk = vx[order]
    vy_vtk = vy[order]
    vz_vtk = vz[order]

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(
        vx_vtk**2
        + vy_vtk**2
        + vz_vtk**2
    )

    return {
        "X": x_unique,
        "Y": y_unique,
        "Z": z_unique,
        "VX": vx_vtk,
        "VY": vy_vtk,
        "VZ": vz_vtk,
        "VelocityMagnitude": velocity_magnitude,
        "NX": nx,
        "NY": ny,
        "NZ": nz,
    }


# ==============================================================
# ParaView .vtr writer
# ==============================================================

def array_to_ascii(values):
    """
    Convert a NumPy array to an ASCII string for a VTK XML file.
    """

    return " ".join(
        f"{value:.12e}"
        for value in values
    )


def write_vtr(data, output_file):
    """
    Write an averaged FLIGHTLAB velocity field as a VTK XML
    RectilinearGrid (.vtr) file for ParaView.

    Parameters
    ----------
    data : dict
        Averaged FLIGHTLAB velocity-field data.

    output_file : Path
        Path of the .vtr file to write.
    """

    grid = create_rectilinear_grid(data)

    nx = grid["NX"]
    ny = grid["NY"]
    nz = grid["NZ"]

    # VTK uses zero-based point extents
    extent = (
        f"0 {nx - 1} "
        f"0 {ny - 1} "
        f"0 {nz - 1}"
    )

    # Combine velocity components into one three-component vector
    velocity = np.column_stack(
        (
            grid["VX"],
            grid["VY"],
            grid["VZ"],
        )
    )

    velocity_ascii = " ".join(
        f"{vx:.12e} {vy:.12e} {vz:.12e}"
        for vx, vy, vz in velocity
    )

    # Create VTK XML file
    vtk_content = f"""<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
  <RectilinearGrid WholeExtent="{extent}">
    <Piece Extent="{extent}">

      <PointData Vectors="Velocity" Scalars="VelocityMagnitude">

        <DataArray
          type="Float64"
          Name="Velocity"
          NumberOfComponents="3"
          format="ascii">
          {velocity_ascii}
        </DataArray>

        <DataArray
          type="Float64"
          Name="VelocityMagnitude"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["VelocityMagnitude"])}
        </DataArray>

        <DataArray
          type="Float64"
          Name="VX"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["VX"])}
        </DataArray>

        <DataArray
          type="Float64"
          Name="VY"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["VY"])}
        </DataArray>

        <DataArray
          type="Float64"
          Name="VZ"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["VZ"])}
        </DataArray>

      </PointData>

      <CellData>
      </CellData>

      <Coordinates>

        <DataArray
          type="Float64"
          Name="X"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["X"])}
        </DataArray>

        <DataArray
          type="Float64"
          Name="Y"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["Y"])}
        </DataArray>

        <DataArray
          type="Float64"
          Name="Z"
          NumberOfComponents="1"
          format="ascii">
          {array_to_ascii(grid["Z"])}
        </DataArray>

      </Coordinates>

    </Piece>
  </RectilinearGrid>
</VTKFile>
"""

    with open(output_file, "w") as file:
        file.write(vtk_content)

    print(
        f"    Saved ParaView velocity field:\n"
        f"      {output_file}"
    )


# ==============================================================
# Natural sorting of velocity-field files
# ==============================================================

def get_step_number(file_path):
    """
    Extract the numerical step number from a velocity-field filename.
    """

    match = re.search(
        r"velocityfield-step-(\d+)",
        file_path.stem,
    )

    if match:
        return int(match.group(1))

    return 0


# ==============================================================
# Main
# ==============================================================

def main():
    """
    Average and export all available FLIGHTLAB velocity fields.
    """

    print("=" * 70)
    print("FLIGHTLAB VELOCITY-FIELD AVERAGING")
    print("=" * 70)

    print(
        f"\nResults directory:\n"
        f"  {RESULTS_DIRECTORY}"
    )

    if not RESULTS_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Results directory does not exist:\n"
            f"{RESULTS_DIRECTORY}"
        )

    # Find all test-condition directories
    case_directories = sorted(
        directory
        for directory in RESULTS_DIRECTORY.iterdir()
        if directory.is_dir()
    )

    if not case_directories:
        raise FileNotFoundError(
            "No FLIGHTLAB test-condition directories were found."
        )

    print(
        f"\nFound {len(case_directories)} "
        "test-condition directories."
    )

    # Process every test condition
    for case_number, case_directory in enumerate(
        case_directories,
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"Case {case_number}/{len(case_directories)}: "
            f"{case_directory.name}"
        )

        print("=" * 70)

        velocity_directory = (
            case_directory
            / VELOCITYFIELD_DIRECTORY
        )

        # Skip test conditions without velocity-field results
        if not velocity_directory.exists():

            print(
                "    No velocityfield directory found. "
                "Skipping case."
            )

            continue

        velocity_files = sorted(
            velocity_directory.glob(
                VELOCITYFIELD_PATTERN
            ),
            key=get_step_number,
        )

        # Skip empty velocity-field directories
        if not velocity_files:

            print(
                "    No velocity-field files found. "
                "Skipping case."
            )

            continue

        print(
            f"    Found {len(velocity_files)} "
            "velocity-field files."
        )

        # Average all sampled velocity fields
        averaged_data = average_velocity_fields(
            velocity_files
        )

        print(
            "    Velocity-field averaging complete."
        )

        # Write averaged field directly inside the velocityfield folder
        output_file = (
            velocity_directory
            / OUTPUT_FILENAME
        )

        write_vtr(
            averaged_data,
            output_file,
        )

    print("\n" + "=" * 70)
    print("POST-PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()