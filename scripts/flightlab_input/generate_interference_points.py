"""
Generate a simple Cartesian point cloud for FlightLab intftabxyz.tab
and visualize the interference points together with the aircraft STL.

Input coordinates are in millimetres.
Output coordinates are normalized using INTFREFLEN_FT.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ==============================================================
# Settings
# ==============================================================

STL_FILE = "simplified airframe.stl"

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "intftabxyz.tab"
)

# Point-cloud boundaries [mm]
X_MIN = 668
X_MAX = 669

Y_MIN = -698
Y_MAX = 248

Z_MIN = -200
Z_MAX = 200

# Uniform spacing in all directions [mm]
DX = 2

# Conversion factor
MM_TO_FT = 1.0 / 304.8

# FlightLab normalization reference length (mean chord)
INTFREFLEN_MM = 312.267658
INTFREFLEN_FT = INTFREFLEN_MM * MM_TO_FT


# ==============================================================
# Create coordinates
# ==============================================================

x_values = np.arange(X_MIN, X_MAX + 0.5 * DX, DX)
y_values = np.arange(Y_MIN, Y_MAX + 0.5 * DX, DX)
z_values = np.arange(Z_MIN, Z_MAX + 0.5 * DX, DX)


# ==============================================================
# Create point cloud
# ==============================================================

# x changes fastest, followed by z and then y
points_mm = np.array([
    [x, y, z]
    for y in y_values
    for z in z_values
    for x in x_values
])


# ==============================================================
# Write FlightLab file
# ==============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("# Template: intftabxyz\n")
    file.write("# Desc:     Position of flowfield interference points in CG-frame\n")
    file.write("#\n")
    file.write("# INTFREFLEN: Reference length for position normalization\n")
    file.write("# INTFXYZ:    Normalized flowfield point positions in CG-frame\n")
    file.write("##\n")
    file.write("!T INTFREFLEN\n")
    file.write("!U ft\n")
    file.write(f"   {INTFREFLEN_FT:.9f}\n")
    file.write("!M INTFXYZ\n")
    file.write("!U nd\n")

    for x_mm, y_mm, z_mm in points_mm:

        x_nd = x_mm * MM_TO_FT / INTFREFLEN_FT
        y_nd = y_mm * MM_TO_FT / INTFREFLEN_FT
        z_nd = z_mm * MM_TO_FT / INTFREFLEN_FT

        file.write(
            f"   {x_nd: .9f}"
            f"   {y_nd: .9f}"
            f"   {z_nd: .9f}\n"
        )


# ==============================================================
# Print information
# ==============================================================

number_of_points = len(points_mm)

print(f"Created: {OUTPUT_FILE}")
print(f"x points: {len(x_values)}")
print(f"y points: {len(y_values)}")
print(f"z points: {len(z_values)}")
print(f"Total points: {number_of_points}")


# ==============================================================
# Load and transform STL
# ==============================================================

mesh = trimesh.load(STL_FILE)

if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)


# --------------------------------------------------------------
# Rotate STL 270 degrees around x-axis
# --------------------------------------------------------------

rotation_matrix = trimesh.transformations.rotation_matrix(
    np.deg2rad(270.0),
    [1, 0, 0]
)

mesh.apply_transform(rotation_matrix)

rotation_matrix = trimesh.transformations.rotation_matrix(
    np.deg2rad(180.0),
    [0, 0, 1]
)

mesh.apply_transform(rotation_matrix)


# --------------------------------------------------------------
# Translate STL to aircraft position
# --------------------------------------------------------------

# Translation [mm]:
#   +632.7 mm forward  -> +x
#    -11.1 mm downward -> -z

translation = np.array([
    632.7,
    0.0,
    -11.1
])

mesh.apply_translation(translation)


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
    alpha=0.4
)

mesh_collection.set_facecolor("lightgray")
mesh_collection.set_edgecolor("none")

ax.add_collection3d(mesh_collection)


# --------------------------------------------------------------
# Interference points
# --------------------------------------------------------------

ax.scatter(
    points_mm[:, 0],
    points_mm[:, 1],
    points_mm[:, 2],
    s=3,
    alpha=0.4
)


# --------------------------------------------------------------
# Axis labels
# --------------------------------------------------------------

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")


# --------------------------------------------------------------
# Plot limits
# --------------------------------------------------------------

# Include both STL and complete interference-point domain
all_plot_points = np.vstack([
    mesh.vertices,
    points_mm
])

mins = all_plot_points.min(axis=0)
maxs = all_plot_points.max(axis=0)

center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

ax.invert_xaxis()

ax.set_box_aspect((1, 1, 1))

plt.tight_layout()
plt.show()