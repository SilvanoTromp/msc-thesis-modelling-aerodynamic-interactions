
"""
Visualize aligned aircraft geometry and rotor configurations for VPMx simulations.

The script:
1. Loads the previously aligned STL geometry.
2. Loads the structured airframe sections.
3. Determines the rotor reference origin from the STL geometry.
4. Determines the wing-root midpoint and load balance reference point.
5. Defines rotor hub locations, radii, and orientations.
6. Calculates absolute rotor positions and positions relative to the load balance.
7. Prints reference coordinates and rotor positions.
8. Visualizes the aircraft mesh, sections, rotor disks, axes, and reference points in 3D.

The script is intended for verification of aircraft geometry, reference
coordinates, and rotor positioning before VPMx simulations.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

# ==============================================================
# User settings
# ==============================================================

stl_file = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "LpCVTOL_AIAA-model_remesh.stl"
)

# Structured airframe sections
sections_file = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "airframeSections.txt"
)

# ==============================================================
# Load STL geometry
# ==============================================================

mesh = pv.read(stl_file)

# ==============================================================
# Load structured airframe sections
# ==============================================================

with open(sections_file, "r", encoding="utf-8") as f:

    # Skip comment lines and empty lines
    lines = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

# Read number of sections and points per section
n_sections, n_points = map(int, lines[0].split())

# Read section coordinates
points = np.array([
    list(map(float, line.split()))
    for line in lines[1:]
])

# Verify expected number of points
if len(points) != n_sections * n_points:
    raise ValueError(
        f"Expected {n_sections * n_points} section points, "
        f"found {len(points)}."
    )

# Reshape into structured sections
structured_sections = points.reshape(
    n_sections,
    n_points,
    3
)

# ==============================================================
# Determine rotor reference origin from STL geometry
# ==============================================================

vertices = mesh.points

# Find vertex with minimum X-coordinate
min_x_idx = np.argmin(vertices[:, 0])
min_x_vertex = vertices[min_x_idx]

# Reference X and Y coordinates
x_ref = min_x_vertex[0]
y_ref = min_x_vertex[1]

# Find vertices at the corresponding Y-coordinate
y_tolerance = 1e-5

same_y_vertices = vertices[
    np.isclose(vertices[:, 1], y_ref, atol=y_tolerance, rtol=0)
]

if len(same_y_vertices) == 0:
    raise ValueError("No vertices found at the reference Y-coordinate.")

# Find maximum X-coordinate among these vertices
max_x_idx = np.argmax(same_y_vertices[:, 0])
max_x_vertex = same_y_vertices[max_x_idx]

# Reference Z-coordinate
z_ref = max_x_vertex[2] + 0.0111

# Construct rotor reference origin
rotor_reference = np.array([
    x_ref,
    y_ref,
    z_ref
])

print("\nRotor reference origin [m]:")
print(f"  X = {rotor_reference[0]:.4f}")
print(f"  Y = {rotor_reference[1]:.4f}")
print(f"  Z = {rotor_reference[2]:.4f}")

# ==============================================================
# Determine root midpoint and load balance location
# ==============================================================

# Find minimum Y coordinate
min_y = np.min(vertices[:, 1])

# Select vertices at minimum Y
section_points = vertices[
    np.isclose(vertices[:, 1], min_y, atol=1e-5, rtol=0)
]

if len(section_points) == 0:
    raise ValueError("No vertices found at the minimum Y-coordinate.")

# Calculate midpoint of X and Z extents
x_mid = 0.5 * (
    np.min(section_points[:, 0]) +
    np.max(section_points[:, 0])
)

z_mid = 0.5 * (
    np.min(section_points[:, 2]) +
    np.max(section_points[:, 2])
)

# Root midpoint in STL coordinates
root_midpoint = np.array([
    x_mid,
    min_y,
    z_mid
])

# Load balance is 0.10125 m in negative Y direction
load_balance = root_midpoint + np.array([
    0.0,
    -0.10125,
    0.0
])

# ==============================================================
# Print reference points
# ==============================================================

print("\nRoot midpoint [m]:")
print(f"  X = {root_midpoint[0]:.4f}")
print(f"  Y = {root_midpoint[1]:.4f}")
print(f"  Z = {root_midpoint[2]:.4f}")

print("\nLoad balance location [m]:")
print(f"  X = {load_balance[0]:.4f}")
print(f"  Y = {load_balance[1]:.4f}")
print(f"  Z = {load_balance[2]:.4f}")

# ==============================================================
# Aircraft rotor locations
# ==============================================================

# Rotor locations relative to reference origin [m]
rotor_locations_relative = {
    "Front hover rotor": np.array([-0.0306, 0.3552, 0.0320]),
    "Aft hover rotor":   np.array([ 0.6492, 0.3567, 0.0418]),
    "Pusher propeller":  np.array([ 0.6647, 0.0000,-0.0111]),
}

# Calculate absolute rotor locations in STL coordinates
rotor_locations = {
    name: rotor_reference + position
    for name, position in rotor_locations_relative.items()
}

# Calculate locations relative to load balance
rotor_locations_balance = {
    name: position - load_balance
    for name, position in rotor_locations.items()
}

# ==============================================================
# Print results
# ==============================================================

def print_coordinates(title, locations):

    print(f"\n{title}")
    print("-" * 65)
    print(f"{'Component':<25} {'X [m]':>12} {'Y [m]':>12} {'Z [m]':>12}")

    for name, position in locations.items():
        print(
            f"{name:<25}"
            f"{position[0]:>12.4f}"
            f"{position[1]:>12.4f}"
            f"{position[2]:>12.4f}"
        )


print_coordinates(
    "ROTOR LOCATIONS RELATIVE TO REFERENCE ORIGIN",
    rotor_locations_relative
)

print_coordinates(
    "ROTOR LOCATIONS IN STL COORDINATES",
    rotor_locations
)

print_coordinates(
    "ROTOR LOCATIONS RELATIVE TO LOAD BALANCE",
    rotor_locations_balance
)

# ==============================================================
# Rotor geometry definitions
# ==============================================================

# Rotor radii [m]
rotor_radii = {
    "Front hover rotor": 0.1524,
    "Aft hover rotor":   0.1524,
    "Pusher propeller":  0.1905,
}

# Rotor axis directions
rotor_axes = {
    "Front hover rotor": np.array([0.0, 0.0, -1.0]),
    "Aft hover rotor":   np.array([0.0, 0.0, -1.0]),
    "Pusher propeller":  np.array([-1.0, 0.0, 0.0]),
}

# ==============================================================
# Visualization settings
# ==============================================================

N_ROTOR_POINTS = 100
ROTOR_AXIS_LENGTH_FACTOR = 0.75

STL_OPACITY = 0.50
ROTOR_OPACITY = 0.25

REFERENCE_POINT_RADIUS = 0.008
ROTOR_HUB_RADIUS = 0.006

# ==============================================================
# Create 3D visualization
# ==============================================================

plotter = pv.Plotter(window_size=(1400, 1000))

# Plot aircraft STL
plotter.add_mesh(
    mesh,
    color="lightgray",
    opacity=STL_OPACITY,
    show_edges=True,
    edge_color="black",
    line_width=0.5,
    smooth_shading=False,
    label="Aircraft panel mesh",
)

# Visualize structured airframe sections
for section in structured_sections:

    # Show section points
    plotter.add_points(
        section,
        color="red",
        point_size=4,
        render_points_as_spheres=True,
    )

    # Show closed section contour
    closed_section = np.vstack([
        section,
        section[0]
    ])

    line = pv.lines_from_points(
        closed_section,
        close=False
    )

    plotter.add_mesh(
        line,
        color="red",
        line_width=1.5,
    )

# Plot rotor reference origin
plotter.add_mesh(
    pv.Sphere(
        radius=REFERENCE_POINT_RADIUS,
        center=rotor_reference,
    ),
    color="blue",
    label="Rotor reference origin",
)

# Plot root midpoint
plotter.add_mesh(
    pv.Sphere(
        radius=REFERENCE_POINT_RADIUS,
        center=root_midpoint,
    ),
    color="green",
    label="Root midpoint",
)

# Plot load balance reference point
plotter.add_mesh(
    pv.Sphere(
        radius=REFERENCE_POINT_RADIUS,
        center=load_balance,
    ),
    color="red",
    label="Load balance",
)

# Plot rotor disks and axes
for name, hub in rotor_locations.items():

    radius = rotor_radii[name]
    axis = rotor_axes[name]

    # Create rotor disk
    disk = pv.Disc(
        center=hub,
        inner=0.0,
        outer=radius,
        normal=axis,
        r_res=1,
        c_res=N_ROTOR_POINTS,
    )

    plotter.add_mesh(
        disk,
        color="gray",
        opacity=ROTOR_OPACITY,
        show_edges=False,
        label=name,
    )

    # Rotor circumference
    edges = disk.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    plotter.add_mesh(
        edges,
        color="black",
        line_width=2,
    )

    # Rotor hub
    plotter.add_mesh(
        pv.Sphere(
            radius=ROTOR_HUB_RADIUS,
            center=hub,
        ),
        color="black",
    )

    # Rotor axis
    axis_length = ROTOR_AXIS_LENGTH_FACTOR * radius

    arrow = pv.Arrow(
        start=hub,
        direction=axis,
        scale=axis_length,
        tip_length=0.20,
        tip_radius=0.08,
        shaft_radius=0.02,
    )

    plotter.add_mesh(
        arrow,
        color="black",
    )

# Coordinate grid
plotter.show_grid(
    xtitle="x [m]",
    ytitle="y [m]",
    ztitle="z [m]",
)

# Orientation axes
plotter.add_axes()

# Equal geometric proportions
plotter.set_scale(1.0, 1.0, 1.0)

# Camera
plotter.view_isometric()

# White background
plotter.set_background("white")

# Show interactive visualization
plotter.show()