"""
Generate structured FLIGHTLAB panel geometry from an STL airframe model.

The script:
1. Loads the STL geometry.
2. Aligns the geometry using predefined reference points.
3. Converts the geometry coordinates from metres to feet.
4. Determines the spanwise boundaries and generates equally spaced sections.
5. Extracts the main airframe contour from each section.
6. Identifies the leading and trailing edges of each section.
7. Resamples the upper and lower contours using cosine-clustered points.
8. Ensures consistent point ordering and correspondence between sections.
9. Generates small end-cap sections to close the panel geometry.
10. Exports the structured panel geometry to a FLIGHTLAB-compatible .tab file.
11. Visualizes the aligned STL geometry and generated panel points in 3D.

The script is intended for airframe geometry alignment, structured panel generation,
and verification of aerodynamic panel discretization in FLIGHTLAB.
"""

import numpy as np
import trimesh
from pathlib import Path

# ==================================================
# User settings
# ==================================================

STL_FILE = "LpCVTOL_AIAA-model_remesh.stl"

SPAN_AXIS = "y"
N_SLICES = 40
N_POINTS_PER_SLICE = 80

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "EVO-body-panelgeom.tab"
)

# FlightLab wants feet
M_TO_FT = 3.280839895

# Clockwise when looking from body-front / negative span side
CLOCKWISE_LOOKING_FROM_NEGATIVE_SPAN = False

# End-cap settings, in STL units
CAP_OFFSET = 0.001 * M_TO_FT
CAP_CONTOUR_SIZE = 0.001 * M_TO_FT

# Trailing edge is min x for aircraft geometry.
TRAILING_EDGE_AXIS = "x"
TRAILING_EDGE_IS_MAX = True

# ==================================================
# Helper functions
# ==================================================

axis_map = {"x": 0, "y": 1, "z": 2}
span_idx = axis_map[SPAN_AXIS]
te_axis_idx = axis_map[TRAILING_EDGE_AXIS]

normal = np.zeros(3)
normal[span_idx] = 1.0

def get_plane_axes(span_idx):
    """Return the two coordinate indices inside the slice plane."""
    return [idx for idx in range(3) if idx != span_idx]

def signed_area_2d(points_2d):
    """Signed polygon area. Positive = CCW, negative = CW."""
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def rotate_start_point(points, plane_axes):
    """Rotate points so every section starts from a consistent location."""
    a, b = plane_axes

    start_idx = np.lexsort((points[:, b], points[:, a]))[0]

    return np.roll(points, -start_idx, axis=0)

def enforce_clockwise(points, span_idx):
    """Make point order clockwise when viewed from negative span side."""
    plane_axes = get_plane_axes(span_idx)
    pts_2d = points[:, plane_axes]

    area = signed_area_2d(pts_2d)

    is_clockwise = area < 0
    desired_clockwise = CLOCKWISE_LOOKING_FROM_NEGATIVE_SPAN

    if is_clockwise != desired_clockwise:
        points = points[::-1]

    points = rotate_start_point(points, plane_axes)

    return points

def circular_shift_to_match_previous(points, previous_points):
    """Shift current section so point indices match the previous section."""
    best_shift = 0
    best_error = np.inf

    for shift in range(len(points)):
        shifted = np.roll(points, -shift, axis=0)
        error = np.sum(np.linalg.norm(shifted - previous_points, axis=1))

        if error < best_error:
            best_error = error
            best_shift = shift

    return np.roll(points, -best_shift, axis=0)

def resample_curve(points, n_points):
    """Resample a closed contour with cosine clustering near LE and TE."""
    points = np.asarray(points)

    if n_points % 2 != 0:
        raise ValueError("N_POINTS_PER_SLICE must be even.")

    # Remove duplicate closing point
    if np.linalg.norm(points[0] - points[-1]) < 1e-9:
        points = points[:-1]

    # Enforce consistent contour orientation
    points = enforce_clockwise(points, span_idx)

    # Identify leading and trailing edges
    x = points[:, te_axis_idx]

    if TRAILING_EDGE_IS_MAX:
        te_idx = np.argmax(x)
        le_idx = np.argmin(x)
    else:
        te_idx = np.argmin(x)
        le_idx = np.argmax(x)

    # Rotate contour so it starts at the trailing edge
    points = np.roll(points, -te_idx, axis=0)

    # Update leading-edge index
    le_idx = (le_idx - te_idx) % len(points)

    if le_idx == 0:
        raise ValueError("Leading and trailing edges coincide.")

    # Split into two paths: TE -> LE and LE -> TE
    surface_1 = points[:le_idx + 1]
    surface_2 = np.vstack([points[le_idx:], points[0]])

    def cosine_resample(surface, n):
        """Resample an open surface using cosine-clustered arc length."""
        ds = np.linalg.norm(np.diff(surface, axis=0), axis=1)
        s = np.insert(np.cumsum(ds), 0, 0.0)

        # Remove zero-length segments
        valid = np.insert(np.diff(s) > 1e-12, 0, True)
        s = s[valid]
        surface = surface[valid]

        if len(s) < 2:
            raise ValueError("Insufficient points for surface resampling.")

        # Cosine distribution from 0 to 1
        theta = np.linspace(0, np.pi, n)
        xi = 0.5 * (1.0 - np.cos(theta))

        target_s = xi * s[-1]

        sampled = np.zeros((n, 3))

        for i in range(3):
            sampled[:, i] = np.interp(target_s, s, surface[:, i])

        return sampled

    # Number of points per surface
    n_surface = n_points // 2 + 1

    sampled_1 = cosine_resample(surface_1, n_surface)
    sampled_2 = cosine_resample(surface_2, n_surface)

    # Combine surfaces without duplicating LE and TE
    sampled = np.vstack([
        sampled_1[:-1],
        sampled_2[:-1]
    ])

    return sampled

def create_tiny_cap_slice(section, span_location):
    """Create a tiny end-cap contour while preserving the trailing-edge point."""
    cap = section.copy()
    plane_axes = get_plane_axes(span_idx)

    # Use trailing edge (point 0) as scaling reference
    center = section[0].copy()

    radial = section[:, plane_axes] - center[plane_axes]
    max_radius = np.max(np.linalg.norm(radial, axis=1))

    if max_radius < 1e-12:
        scale = 1.0
    else:
        scale = CAP_CONTOUR_SIZE / max_radius

    cap[:, plane_axes] = center[plane_axes] + radial * scale
    cap[:, span_idx] = span_location

    return cap

def write_values_in_flightlab_format(f, values, values_per_line=3):
    """Write values in free format, column-wise style."""
    for i, value in enumerate(values):
        f.write(f"{value: .8f}")
        if (i + 1) % values_per_line == 0:
            f.write("\n")
        else:
            f.write(" ")
    if len(values) % values_per_line != 0:
        f.write("\n")


def write_flightlab_panel_tab(filename, slices):
    """Write structured panel grid to FLIGHTLAB .tab format."""
    n_slices = len(slices)
    n_points = slices[0].shape[0]

    npatch = 1
    npcol = n_slices - 1
    nprow = n_points

    pedge = []

    for section in slices:

        # Add extra closing point for structured panel connection
        section_closed = np.vstack([section, section[0]])

        for p in section_closed:
            pedge.append(p)

    pedge = np.asarray(pedge)

    pedge_columnwise = np.concatenate([
        pedge[:, 0],
        pedge[:, 1],
        pedge[:, 2]
    ])

    with open(filename, "w") as f:
        f.write(f"# NPATCH 1 1\n")
        f.write(f"# NPCOL 1 1\n")
        f.write(f"# NPROW 1 1\n")
        f.write(f"# PEDGE {pedge.shape[0]} 3\n")

        f.write(f"!B NPATCH 1 1\n")
        f.write(f"{npatch:.8f}\n")

        f.write(f"!B NPCOL 1 1\n")
        f.write(f"{npcol:.8f}\n")

        f.write(f"!B NPROW 1 1\n")
        f.write(f"{nprow:.8f}\n")

        f.write(f"!B PEDGE {pedge.shape[0]} 3\n")
        write_values_in_flightlab_format(f, pedge_columnwise, values_per_line=3)

    print(f"Saved FLIGHTLAB panel file to {filename}")
    print(f"NPATCH = {npatch}")
    print(f"NPCOL  = {npcol}")
    print(f"NPROW  = {nprow}")
    print(f"PEDGE  = {pedge.shape[0]} x 3")

# ==================================================
# Load STL
# ==================================================

mesh = trimesh.load(STL_FILE)

if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

# ==================================================
# Load STL
# ==================================================

mesh = trimesh.load(STL_FILE)

if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

# Find the vertex with minimum X-coordinate
vertices = mesh.vertices
min_x_idx = np.argmin(vertices[:, 0])
reference_point = vertices[min_x_idx]

# Calculate required X- and Y-translations
x_offset = -reference_point[0]
y_offset = -reference_point[1]

# Apply translation
mesh.apply_translation([x_offset, y_offset, 0])

# Target Z-coordinate at maximum X and Y = 0 [m]
TARGET_Z = -0.0111

# Find vertices closest to the symmetry plane Y = 0
vertices = mesh.vertices
y_tolerance = 1e-4
center_vertices = vertices[np.abs(vertices[:, 1]) < y_tolerance]

if len(center_vertices) == 0:
    raise ValueError("No vertices found near Y = 0.")

# Find the maximum X-coordinate on the symmetry plane
max_x_idx = np.argmax(center_vertices[:, 0])
reference_point = center_vertices[max_x_idx]

# Calculate required Z-translation
z_offset = TARGET_Z - reference_point[2]

# Apply Z-translation
mesh.apply_translation([0, 0, z_offset])

# Save translated geometry in metres
output_path = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "bemt_vpmx_input"
    / "EVO-body-translated.stl"
)
mesh.export(output_path)
print(f"Saved translated STL geometry in metres to {output_path.resolve()}")

# Convert geometry from original STL units to feet
mesh.apply_scale(M_TO_FT)

# Determine bounds of rotated geometry
bounds = mesh.bounds
span_min = bounds[0, span_idx]
span_max = bounds[1, span_idx]

# Keep original span limits for caps
original_span_min = span_min
original_span_max = span_max

# Real slices are slightly inside the original STL span,
# so the tiny caps close exactly at the original begin/end.
slice_locations = np.linspace(
    original_span_min + CAP_OFFSET,
    original_span_max - CAP_OFFSET,
    N_SLICES
)

structured_slices = []

# ==================================================
# Slice wing
# ==================================================

for i, loc in enumerate(slice_locations):
    origin = np.zeros(3)
    origin[span_idx] = loc

    section = mesh.section(
        plane_origin=origin,
        plane_normal=normal
    )

    if section is None:
        print(f"Warning: no section found at slice {i}, {SPAN_AXIS} = {loc:.4f}")
        continue

    curves = section.discrete

    if len(curves) == 0:
        print(f"Warning: empty section at slice {i}, {SPAN_AXIS} = {loc:.4f}")
        continue

    curve_lengths = [
        np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))
        for c in curves
    ]

    main_curve = curves[np.argmax(curve_lengths)]

    # Resample contour with point 0 at the trailing edge
    sampled = resample_curve(main_curve, N_POINTS_PER_SLICE)

    structured_slices.append(sampled)

# ==================================================
# Add tiny end-cap slices
# ==================================================

first_real_slice = structured_slices[0]
last_real_slice = structured_slices[-1]

# Caps are placed exactly at the original STL span limits
first_cap_location = original_span_min
last_cap_location = original_span_max

first_cap_slice = create_tiny_cap_slice(first_real_slice, first_cap_location)
last_cap_slice = create_tiny_cap_slice(last_real_slice, last_cap_location)

structured_slices = [first_cap_slice] + structured_slices + [last_cap_slice]

# ==================================================
# Verify trailing-edge point ordering
# ==================================================

for i, section in enumerate(structured_slices):

    # Identify trailing edge
    if TRAILING_EDGE_IS_MAX:
        te_idx = np.argmax(section[:, te_axis_idx])
    else:
        te_idx = np.argmin(section[:, te_axis_idx])

    # Verify that point 0 is at the trailing edge
    if te_idx != 0:
        raise ValueError(
            f"Section {i}: point 0 is not at the trailing edge. "
            f"Trailing edge found at point {te_idx}."
        )

print("Verified: point 0 is at the trailing edge of every section.")

# ==================================================
# Save points
# ==================================================

all_points = []

for i, section in enumerate(structured_slices):
    for j, p in enumerate(section):
        all_points.append([i, j, p[0], p[1], p[2]])

all_points = np.array(all_points)

# ==================================================
# Save FLIGHTLAB panel file
# ==================================================

write_flightlab_panel_tab(OUTPUT_FILE, structured_slices)

# ==================================================
# Static matplotlib plot
# ==================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# STL mesh
mesh_collection = Poly3DCollection(
    mesh.triangles,
    alpha=0.4
)

mesh_collection.set_facecolor("lightgray")
mesh_collection.set_edgecolor("none")
ax.add_collection3d(mesh_collection)

# Panel points
ax.scatter(
    all_points[:, 2],
    all_points[:, 3],
    all_points[:, 4],
    c="red",
    s=8
)

ax.set_xlabel("x [ft]")
ax.set_ylabel("y [ft]")
ax.set_zlabel("z [ft]")

# Equal axis scaling
vertices = mesh.vertices

mins = vertices.min(axis=0)
maxs = vertices.max(axis=0)

center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

plt.show()