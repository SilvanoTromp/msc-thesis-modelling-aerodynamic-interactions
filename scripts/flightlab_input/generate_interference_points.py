"""
Generate a Cartesian point cloud for FLIGHTLAB intftabxyz.tab
and visualize the interference points together with the aircraft STL.

The script:
1. Loads the previously aligned aircraft STL geometry in feet.
2. Uses the FLIGHTLAB inertial coordinate frame.
3. Generates interference points using uniform spacing.
4. Normalizes the interference-point coordinates using INTFREFLEN_FT.
5. Exports the interference points to a FLIGHTLAB-compatible .tab file.
6. Prints the point-cloud dimensions and total number of points.
7. Visualizes the STL, interference points, and point-cloud boundaries.

The script is intended for preparation and verification of interference
point locations used for flow-field evaluation in FLIGHTLAB.
"""

from pathlib import Path
from itertools import product, combinations

import numpy as np
import trimesh
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==============================================================
# User settings
# ==============================================================

STL_FILE = (
    Path(__file__).resolve().parent
    / "simplified airframe_aligned.stl"
)

# Verify that the aligned STL file exists
if not STL_FILE.is_file():
    raise FileNotFoundError(
        f"\nAligned STL file not found: {STL_FILE}\n"
        "Please run 'generate_panelgeom.py' first to generate the rotated and translated STL geometry."
    )

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "intftabxyz.tab"
)

# ==============================================================
# Conversion factors
# ==============================================================

# Conversion factors
MM_TO_FT = 1.0 / 304.8
FT_TO_MM = 304.8

# FLIGHTLAB normalization reference length (mean chord)
INTFREFLEN_MM = 312.267658
INTFREFLEN_FT = INTFREFLEN_MM * MM_TO_FT

# ==============================================================
# Coordinate systems
# ==============================================================

# Pusher propeller hub location in inertial coordinates
PUSHER_ORIGIN_FT = np.array([
    -1.2720,
     0.0,
    -89.9636
])
PUSHER_ORIGIN_MM = PUSHER_ORIGIN_FT * FT_TO_MM

# Wing location in inertial coordinates
WING_ORIGIN_MM = np.array([
     150,
     0.0,
    -11.1
])
WING_ORIGIN_FT = WING_ORIGIN_MM * MM_TO_FT

# Inertial frame to reference frame
INERTIAL_ORIGIN_FT = np.array([
     0.0,
     0.0,
    -90,
])
INERTIAL_ORIGIN_MM = INERTIAL_ORIGIN_FT * FT_TO_MM

# ==============================================================
# Point-cloud settings
# ==============================================================

# Flow-field plane location relative to aircraft origin [mm]
X_PLANE_BODY = -50.0

# Plane boundaries relative to aircraft origin [mm]
Y_MIN = -248
Y_MAX = 698

Z_MIN = -200
Z_MAX = 200

# Uniform YZ-spacing [mm]
DY = 1
DZ = 1

# ==============================================================
# Create coordinates in aircraft frame
# ==============================================================

x_values = np.array([X_PLANE_BODY])

y_values = np.arange(
    Y_MIN,
    Y_MAX + 0.5 * DY,
    DY
)

z_values = np.arange(
    Z_MIN,
    Z_MAX + 0.5 * DZ,
    DZ
)

# ==============================================================
# Create point cloud
# ==============================================================

# Generate points on the y-z plane
points_body_mm = np.array([
    [x, y, z]
    for y, z, x in product(
        y_values,
        z_values,
        x_values
    )
])

# Transform body-frame coordinates into inertial frame
points_inertial_mm = points_body_mm + PUSHER_ORIGIN_MM

# Convert to feet
points_body_ft = points_body_mm * MM_TO_FT
points_inertial_ft = points_inertial_mm * MM_TO_FT

# ==============================================================
# Write FLIGHTLAB file
# ==============================================================

# Create output directory if necessary
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("# Template: intftabxyz\n")
    file.write("# Desc:     Position of flowfield interference points in inertial frame\n")
    file.write("#\n")
    file.write("# INTFREFLEN: Reference length for position normalization\n")
    file.write("# INTFXYZ:    Normalized flowfield point positions in interial frame\n")
    file.write("##\n")
    file.write("!T INTFREFLEN\n")
    file.write("!U ft\n")
    file.write(f"   {INTFREFLEN_FT:.9f}\n")
    file.write("!M INTFXYZ\n")
    file.write("!U nd\n")

    # Normalize all coordinates
    points_nd = points_inertial_mm / INTFREFLEN_MM

    # Write interference points
    np.savetxt(
        file,
        points_nd,
        fmt="   % .9f   % .9f   % .9f"
    )

# ==============================================================
# Print information
# ==============================================================

number_of_points = len(points_nd)

print(f"\nCreated: {OUTPUT_FILE}")

print("\nPOINT-CLOUD INFORMATION")
print("-" * 45)
print(f"X points:       {len(x_values)}")
print(f"Y points:       {len(y_values)}")
print(f"Z points:       {len(z_values)}")
print(f"Total points:   {number_of_points:,}")

print("\nPOINT-CLOUD BOUNDARIES [mm]")
print("-" * 45)
print(f"X: {x_values.min():.1f} to {x_values.max():.1f}")
print(f"Y: {y_values.min():.1f} to {y_values.max():.1f}")
print(f"Z: {z_values.min():.1f} to {z_values.max():.1f}")

print(f"\nReference length: {INTFREFLEN_FT:.9f} ft")

# ==============================================================
# Load aligned STL
# ==============================================================

# Load previously aligned STL geometry
# Coordinates are already in feet
mesh = trimesh.load(STL_FILE)

if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

# Rotate STL 180 degrees around the Y-axis
rotation_matrix = trimesh.transformations.rotation_matrix(
   np.pi,
    [0, 1, 0]
)
mesh.apply_transform(rotation_matrix)

# Translate STL from aircraft frame to inertial frame
mesh.vertices += INERTIAL_ORIGIN_FT

# Align STL
mesh.vertices += WING_ORIGIN_FT

# ==============================================================
# Plot STL and interference points
# ==============================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# --------------------------------------------------------------
# STL mesh
# --------------------------------------------------------------

mesh_collection = Poly3DCollection(
    mesh.triangles,
    alpha=1
)

mesh_collection.set_facecolor("lightgray")
mesh_collection.set_edgecolor("none")

ax.add_collection3d(mesh_collection)

# --------------------------------------------------------------
# Interference points
# ==============================================================

# Generate structured visualization grid
plot_points = points_inertial_ft

ax.scatter(
    plot_points[:, 0],
    plot_points[:, 1],
    plot_points[:, 2],
    s=2,
    alpha=0.25,
    color="tab:blue",
    label="Interference points"
)

# --------------------------------------------------------------
# Axis labels
# ==============================================================

ax.set_xlabel("x [ft]")
ax.set_ylabel("y [ft]")
ax.set_zlabel("z [ft]")

# --------------------------------------------------------------
# Plot limits
# ==============================================================

mins = np.minimum(
    mesh.vertices.min(axis=0),
    points_inertial_ft.min(axis=0)
)

maxs = np.maximum(
    mesh.vertices.max(axis=0),
    points_inertial_ft.max(axis=0)
)

center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

ax.set_xlim(
    center[0] - radius,
    center[0] + radius
)

ax.set_ylim(
    center[1] - radius,
    center[1] + radius
)

ax.set_zlim(
    center[2] - radius,
    center[2] + radius
)

ax.set_box_aspect((1, 1, 1))

# --------------------------------------------------------------
# Final visualization
# ==============================================================

plt.tight_layout()
plt.show()