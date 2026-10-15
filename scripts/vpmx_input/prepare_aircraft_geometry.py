"""
Visualize aligned aircraft geometry and rotor configurations for VPMx simulations.

The script:
1. Loads the previously aligned STL geometry.
2. Determines the wing-root midpoint and load balance reference point.
3. Defines rotor hub locations, radii, and orientations.
4. Calculates rotor positions relative to the load balance.
5. Prints reference coordinates and rotor positions.
6. Visualizes the aircraft mesh, rotor disks, axes, and reference points in 3D.

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
    / "EVO-body-translated.stl"
)

# Structured airframe sections
sections_file = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "EVO-body-sections.txt"
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
        if line.strip() and not line.startswith("#")
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
# Determine root midpoint and load balance location
# ==============================================================

# Find minimum Y coordinate
min_y = np.min(mesh.points[:, 1])

# Select vertices at minimum Y
section_points = mesh.points[
    np.isclose(mesh.points[:, 1], min_y, atol=1e-5, rtol=0)
]

# Calculate midpoint of X and Z extents
x_mid = 0.5 * (np.min(section_points[:, 0]) + np.max(section_points[:, 0]))
z_mid = 0.5 * (np.min(section_points[:, 2]) + np.max(section_points[:, 2]))

# Root midpoint in translated STL coordinates
root_midpoint = np.array([x_mid, min_y, z_mid])

# Load balance is 0.10125 m in negative Y direction
load_balance = root_midpoint + np.array([0.0, -0.10125, 0.0])

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

# Rotor locations relative to LE reference [m]
rotor_locations = {
    "Front hover rotor": np.array([-0.0306, 0.3552, 0.0320]),
    "Aft hover rotor":   np.array([ 0.6492, 0.3567, 0.0418]),
    "Pusher propeller":  np.array([ 0.6647, 0.0000,-0.0111])
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
    "ROTOR LOCATIONS RELATIVE TO LE",
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