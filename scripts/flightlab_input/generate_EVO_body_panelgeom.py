# create_panel_points_from_stl.py
import numpy as np
import trimesh
import plotly.graph_objects as go
from pathlib import Path

# ==================================================
# User settings
# ==================================================

STL_FILE = "simplified airframe.stl"

SPAN_AXIS = "z"       # Change to "x", "y", or "z"
N_SLICES = 25
N_POINTS_PER_SLICE = 40

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "flightlab_input"
    / "EVO-body-panelgeom.tab"
)

# FlightLab wants feet
STL_UNITS = "mm"      # Change to "ft" if your STL is already in feet

if STL_UNITS == "mm":
    UNIT_TO_FT = 0.003280839895
elif STL_UNITS == "m":
    UNIT_TO_FT = 3.280839895
elif STL_UNITS == "ft":
    UNIT_TO_FT = 1.0
else:
    raise ValueError("Unknown STL_UNITS. Use 'mm', 'm', or 'ft'.")

# Clockwise when looking from body-front / negative span side
CLOCKWISE_LOOKING_FROM_NEGATIVE_SPAN = False

# End-cap settings, in STL units
CAP_OFFSET = 0.001
CAP_CONTOUR_SIZE = 0.001

# Trailing edge is min x for aircraft geometry.
TRAILING_EDGE_AXIS = "x"
TRAILING_EDGE_IS_MAX = False

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
    """Resample a 3D curve to n equally spaced points along arc length."""
    points = np.asarray(points)

    if np.linalg.norm(points[0] - points[-1]) < 1e-9:
        points = points[:-1]

    points = enforce_clockwise(points, span_idx)

    closed_points = np.vstack([points, points[0]])

    ds = np.linalg.norm(np.diff(closed_points, axis=0), axis=1)
    s = np.insert(np.cumsum(ds), 0, 0.0)

    target_s = np.linspace(0, s[-1], n_points, endpoint=False)

    sampled = np.zeros((n_points, 3))
    for i in range(3):
        sampled[:, i] = np.interp(target_s, s, closed_points[:, i])

    return sampled


def create_tiny_cap_slice(section, span_location):
    """Create a tiny contour slice to close the panel body without collapsing to one point."""
    cap = section.copy()
    plane_axes = get_plane_axes(span_idx)

    center = np.mean(section, axis=0)

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
        section_ft = section * UNIT_TO_FT

        # Add extra closing point for structured panel connection
        section_closed = np.vstack([section_ft, section_ft[0]])

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

    sampled = resample_curve(main_curve, N_POINTS_PER_SLICE)

    # Ensure same point correspondence between neighbouring slices
    if len(structured_slices) > 0:
        sampled = circular_shift_to_match_previous(sampled, structured_slices[-1])

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


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

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

# ==================================================
# Interactive Plotly plot
# ==================================================
vertices = mesh.vertices
faces = mesh.faces

fig = go.Figure()

fig.add_trace(
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightgray",
        opacity=0.5,
        name="STL"
    )
)

fig.add_trace(
    go.Scatter3d(
        x=all_points[:, 2],
        y=all_points[:, 3],
        z=all_points[:, 4],
        mode="markers",
        marker=dict(
            size=3,
            color="red"
        ),
        name="Panel points"
    )
)

fig.update_layout(
    title="Wing Panel Geometry",
    scene=dict(
        aspectmode="data"
    )
)

fig.show()