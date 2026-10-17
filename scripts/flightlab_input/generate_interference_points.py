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

<<<<<<< HEAD

=======
>>>>>>> origin/main
# ==============================================================
# Settings
# ==============================================================

<<<<<<< HEAD
STL_FILE = "simplified airframe.stl"
=======
STL_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "EVO-body-translated.stl"
)
>>>>>>> origin/main

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "intftabxyz.tab"
)

# Point-cloud boundaries [mm]
<<<<<<< HEAD
X_MIN = 668
X_MAX = 669
=======
X_MIN = -500
X_MAX = 1500
>>>>>>> origin/main

Y_MIN = -698
Y_MAX = 248

Z_MIN = -200
Z_MAX = 200

<<<<<<< HEAD
# Uniform spacing in all directions [mm]
DX = 2
=======
# Region of interest [mm]
X_CENTER = 714.7       # 50 mm downstream of pusher propeller hub
X_FINE_RADIUS = 24     # fine spacing within +/- 24 mm

# Non-uniform x-spacing [mm]
DX_MIN = 2
DX_MAX = 100
DX_GROWTH = 1.4

# Uniform yz-spacing [mm]
DY = 2
DZ = 2
>>>>>>> origin/main

# Conversion factor
MM_TO_FT = 1.0 / 304.8

# FlightLab normalization reference length (mean chord)
INTFREFLEN_MM = 312.267658
INTFREFLEN_FT = INTFREFLEN_MM * MM_TO_FT

<<<<<<< HEAD

=======
>>>>>>> origin/main
# ==============================================================
# Create coordinates
# ==============================================================

<<<<<<< HEAD
x_values = np.arange(X_MIN, X_MAX + 0.5 * DX, DX)
y_values = np.arange(Y_MIN, Y_MAX + 0.5 * DX, DX)
z_values = np.arange(Z_MIN, Z_MAX + 0.5 * DX, DX)

=======
def generate_x_coordinates(
    x_min,
    x_max,
    x_center,
    fine_radius,
    dx_min,
    dx_max,
    growth_factor
):
    """Generate smoothly stretched x-planes around a fine-resolution region."""

    def generate_direction(boundary, direction):

        coordinates = [x_center]
        position = x_center

        # Fine spacing within the region of interest
        while abs(position - x_center) < fine_radius:

            next_position = position + direction * dx_min

            if direction > 0 and next_position >= boundary:
                coordinates.append(boundary)
                return coordinates

            if direction < 0 and next_position <= boundary:
                coordinates.append(boundary)
                return coordinates

            coordinates.append(next_position)
            position = next_position

        # Gradually increase spacing outside the fine region
        spacing = dx_min

        while True:

            spacing = min(
                spacing * growth_factor,
                dx_max
            )

            next_position = position + direction * spacing

            if direction > 0 and next_position >= boundary:
                coordinates.append(boundary)
                break

            if direction < 0 and next_position <= boundary:
                coordinates.append(boundary)
                break

            coordinates.append(next_position)
            position = next_position

        return coordinates

    negative = generate_direction(x_min, -1)
    positive = generate_direction(x_max, 1)

    coordinates = np.unique(
        np.round(negative + positive, 6)
    )

    return coordinates

x_values = generate_x_coordinates(
    X_MIN,
    X_MAX,
    X_CENTER,
    X_FINE_RADIUS,
    DX_MIN,
    DX_MAX,
    DX_GROWTH
)

y_values = np.arange(Y_MIN, Y_MAX + 0.5 * DY, DY)
z_values = np.arange(Z_MIN, Z_MAX + 0.5 * DZ, DZ)
>>>>>>> origin/main

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

<<<<<<< HEAD

=======
>>>>>>> origin/main
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

<<<<<<< HEAD

=======
>>>>>>> origin/main
# ==============================================================
# Print information
# ==============================================================

number_of_points = len(points_mm)

print(f"Created: {OUTPUT_FILE}")
print(f"x points: {len(x_values)}")
print(f"y points: {len(y_values)}")
print(f"z points: {len(z_values)}")
print(f"Total points: {number_of_points}")

<<<<<<< HEAD

=======
>>>>>>> origin/main
# ==============================================================
# Load and transform STL
# ==============================================================

mesh = trimesh.load(STL_FILE)

if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

<<<<<<< HEAD

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

=======
# Convert STL coordinates from metres to millimetres
mesh.apply_scale(1000)

# Mirror STL geometry along the Y-axis (FlightLab convention)
mesh.vertices[:, 1] *= -1

# Correct face winding after reflection
mesh.invert()
>>>>>>> origin/main

# ==============================================================
# Plot STL and interference points
# ==============================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

<<<<<<< HEAD

=======
>>>>>>> origin/main
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

<<<<<<< HEAD

=======
>>>>>>> origin/main
# --------------------------------------------------------------
# Interference points
# --------------------------------------------------------------

<<<<<<< HEAD
ax.scatter(
    points_mm[:, 0],
    points_mm[:, 1],
    points_mm[:, 2],
    s=3,
    alpha=0.4
)

=======
# Maximum number of points to visualize
MAX_PLOT_POINTS = 20000

# Randomly select points for visualization
rng = np.random.default_rng(seed=42)

n_plot = min(MAX_PLOT_POINTS, len(points_mm))

indices = rng.choice(
    len(points_mm),
    size=n_plot,
    replace=False
)

plot_points = points_mm[indices]

ax.scatter(
    plot_points[:, 0],
    plot_points[:, 1],
    plot_points[:, 2],
    s=2,
    alpha=0.25
)

# --------------------------------------------------------------
# Interference point-cloud bounding box
# --------------------------------------------------------------

from itertools import product, combinations

# Bounding box limits [mm]
bounds = [
    (X_MIN, X_MAX),
    (Y_MIN, Y_MAX),
    (Z_MIN, Z_MAX)
]

# Generate the eight corners
corners = np.array(list(product(*bounds)))

# Draw the twelve edges of the bounding box
for start, end in combinations(corners, 2):

    # Only connect corners differing in one coordinate
    if np.sum(np.abs(start - end) > 1e-9) == 1:

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="black",
            linewidth=1.2,
            alpha=0.7
        )
>>>>>>> origin/main

# --------------------------------------------------------------
# Axis labels
# --------------------------------------------------------------

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")

<<<<<<< HEAD

=======
>>>>>>> origin/main
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