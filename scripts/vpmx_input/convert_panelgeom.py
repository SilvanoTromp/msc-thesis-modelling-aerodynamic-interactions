"""
Convert a FLIGHTLAB structured panel geometry to VPMx airframe sections.

The script:
1. Reads NPCOL, NPROW, and PEDGE from a FLIGHTLAB .tab file.
2. Reconstructs the XYZ coordinates stored column-wise.
3. Converts the geometry coordinates from feet to metres.
4. Reconstructs the structured airframe sections.
5. Removes the duplicate closing point from each section.
6. Preserves the original point ordering and section correspondence.
7. Exports the structured sections to a VPMx-compatible .txt file.
"""

from pathlib import Path

import numpy as np

# ==============================================================
# User settings
# ==============================================================

INPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "EVO-body-panelgeom.tab"
)

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "EVO-body-sections.txt"
)

# FLIGHTLAB coordinates are stored in feet
FT_TO_M = 0.3048

# ==============================================================
# Read FLIGHTLAB data blocks
# ==============================================================

lines = INPUT_FILE.read_text().splitlines()

def read_block(name):
    """Read a numeric data block from a FLIGHTLAB .tab file."""

    header = next(
        i for i, line in enumerate(lines)
        if line.startswith(f"!B {name} ")
    )

    count = int(lines[header].split()[2])
    n_components = int(lines[header].split()[3])

    values = []

    for line in lines[header + 1:]:
        if line.startswith("!B "):
            break

        values.extend(float(v) for v in line.split())

    expected = count * n_components

    if len(values) != expected:
        raise ValueError(
            f"{name}: expected {expected} values, "
            f"found {len(values)}."
        )

    return np.asarray(values)

# ==============================================================
# Extract panel dimensions
# ==============================================================

ncol = int(read_block("NPCOL")[0])
nrow = int(read_block("NPROW")[0])

n_sections = ncol + 1

# NPROW represents the number of panels around each contour
n_points_per_section = nrow

# FLIGHTLAB includes an additional closing point
n_vertices = n_sections * (n_points_per_section + 1)

# ==============================================================
# Reconstruct XYZ coordinates
# ==============================================================

values = read_block("PEDGE")

if len(values) != 3 * n_vertices:
    raise ValueError(
        f"Expected {3 * n_vertices} coordinate values, "
        f"found {len(values)}."
    )

# Coordinates are stored column-wise: all X, then Y, then Z
points = values.reshape(3, n_vertices).T

# Convert FLIGHTLAB coordinates from feet to metres
points *= FT_TO_M

# Original ordering: section index first, contour index second
grid_points = points.reshape(
    n_sections,
    n_points_per_section + 1,
    3
)

# ==============================================================
# Extract structured airframe sections
# ==============================================================

structured_sections = []

for i in range(n_sections):

    section = grid_points[i]

    # Verify that the contour is closed
    if not np.allclose(section[0], section[-1], atol=1e-7):
        raise ValueError(
            f"Section {i} does not contain a matching closing point."
        )

    # Remove duplicate closing point
    section = section[:-1]

    structured_sections.append(section)

structured_sections = np.asarray(structured_sections)

# ==============================================================
# Export structured sections
# ==============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:

    f.write(
        "# Structured airframe sections, world coordinates (x, y, z in metres).\n"
    )
    f.write(
        "# Point ordering and section correspondence are preserved from FLIGHTLAB.\n"
    )
    f.write(
        "# Each section contains a closed contour without a duplicate endpoint.\n"
    )
    f.write("# nsections npoints\n")

    f.write(f"{n_sections} {n_points_per_section}\n")

    for section in structured_sections:
        for point in section:
            f.write(
                f"{point[0]:.9f} "
                f"{point[1]:.9f} "
                f"{point[2]:.9f}\n"
            )

# ==============================================================
# Print geometry information
# ==============================================================

print(f"Saved structured airframe sections to {OUTPUT_FILE.resolve()}")
print(f"Number of sections: {n_sections}")
print(f"Points per section: {n_points_per_section}")
print(f"Total points: {n_sections * n_points_per_section}")
print("Coordinate units: metres")