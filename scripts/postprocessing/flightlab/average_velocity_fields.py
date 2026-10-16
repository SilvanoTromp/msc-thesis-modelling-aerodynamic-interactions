"""
Average FLIGHTLAB velocity-field snapshots and export mean-flow vorticity.

The script:
1. Searches a specified folder for velocityfield-step-[number].tab files.
2. Reads coordinates and total velocity components from each FLIGHTLAB file.
3. Checks that all snapshots contain the same structured Cartesian grid.
4. Averages Vx, Vy and Vz over the available snapshots.
5. Calculates vorticity from the time-averaged velocity field.
6. Exports a Tecplot ASCII .plt file readable by ParaView.

Requirements: numpy

Note: The .plt extension here denotes Tecplot ASCII data, not Tecplot's
binary PLT format. In ParaView, select the Tecplot reader if prompted.
"""

from pathlib import Path
import re

import numpy as np

#==============================================================#
# User settings
#==============================================================#

INPUT_FOLDER = Path(r"./results/V-10.30_alpha-0_RPMs-6700-5754-3705/velocityfield")
OUTPUT_FILE = INPUT_FOLDER / "velocityfield-averaged.plt"

# Set to True if VX/VY/VZ contain only induced velocity and the
# free-stream components WINDX/WINDY/WINDZ should be added.
ADD_FREESTREAM = False

#==============================================================#
# Read FLIGHTLAB TAB blocks
#==============================================================#

BLOCK_HEADER = re.compile(r"^!B\s+(\S+)\s+(\d+)\s+(\d+)")
REQUIRED = ("XPTS", "YPTS", "ZPTS", "VX", "VY", "VZ")


def read_tab(path):
    """Read selected numeric blocks from a FLIGHTLAB .tab file."""
    wanted = set(REQUIRED)
    if ADD_FREESTREAM:
        wanted.update(("WINDX", "WINDY", "WINDZ"))

    blocks = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        lines = iter(stream)
        for line in lines:
            match = BLOCK_HEADER.match(line.strip())
            if not match:
                continue
            name, rows, cols = match.group(1), int(match.group(2)), int(match.group(3))
            count = rows * cols
            values = []
            # Consume the entire block, including blocks not requested.
            while len(values) < count:
                try:
                    data_line = next(lines)
                except StopIteration as exc:
                    raise ValueError(f"Incomplete block {name} in {path.name}") from exc
                if data_line.lstrip().startswith(("!B", "#")):
                    raise ValueError(f"Unexpected header inside block {name} in {path.name}")
                values.extend(float(v.replace("D", "E").replace("d", "e"))
                              for v in data_line.split())
            if len(values) != count:
                raise ValueError(f"Incorrect number of values in block {name}: {path.name}")
            if name in wanted:
                blocks[name] = np.asarray(values, dtype=float)

    missing = wanted.difference(blocks)
    if missing:
        raise ValueError(f"Missing blocks {sorted(missing)} in {path.name}")
    n = len(blocks["XPTS"])
    if any(len(blocks[key]) != n for key in REQUIRED):
        raise ValueError(f"Coordinate/velocity lengths differ in {path.name}")
    return blocks


#==============================================================#
# Assemble structured grid and calculate vorticity
#==============================================================#


def structured_field(x, y, z, vx, vy, vz):
    """Map flattened points to a complete Cartesian grid."""
    axes = [np.unique(c) for c in (x, y, z)]
    shape = tuple(len(a) for a in axes)
    if np.prod(shape) != len(x):
        raise ValueError("Points do not form a complete Cartesian grid; "
                         "vorticity requires a structured grid.")
    indices = tuple(np.searchsorted(a, c) for a, c in zip(axes, (x, y, z)))
    flat = np.ravel_multi_index(indices, shape)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("Duplicate Cartesian grid points found.")
    fields = []
    for values in (vx, vy, vz):
        field = np.empty(shape)
        field[indices] = values
        fields.append(field)
    return axes, fields


def derivative(field, axis_values, axis):
    """Calculate spatial derivative; singleton dimensions give zero."""
    if len(axis_values) == 1:
        return np.zeros_like(field)
    return np.gradient(field, axis_values, axis=axis,
                       edge_order=2 if len(axis_values) >= 3 else 1)


def calculate_vorticity(axes, fields):
    """Calculate curl(V) in 1/s, assuming coordinates in metres."""
    vx, vy, vz = fields
    x, y, z = axes
    wx = derivative(vz, y, 1) - derivative(vy, z, 2)
    wy = derivative(vx, z, 2) - derivative(vz, x, 0)
    wz = derivative(vy, x, 0) - derivative(vx, y, 1)
    return wx, wy, wz


#==============================================================#
# Export Tecplot ASCII data
#==============================================================#


def write_tecplot(path, axes, fields, vort):
    """Write structured POINT-format Tecplot ASCII with a .plt extension."""
    x, y, z = axes
    vx, vy, vz = fields
    wx, wy, wz = vort
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    wmag = np.sqrt(wx**2 + wy**2 + wz**2)
    nx, ny, nz = vx.shape

    with path.open("w", encoding="utf-8") as out:
        out.write('TITLE = "Time-averaged FLIGHTLAB velocity field"\n')
        out.write('VARIABLES = "X" "Y" "Z" "Vx" "Vy" "Vz" "Vmag" '
                  '"VorticityX" "VorticityY" "VorticityZ" "VorticityMag"\n')
        out.write(f'ZONE I={nx}, J={ny}, K={nz}, DATAPACKING=POINT\n')
        # Tecplot structured zones require I to vary fastest.
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    values = (x[i], y[j], z[k], vx[i,j,k], vy[i,j,k], vz[i,j,k],
                              vmag[i,j,k], wx[i,j,k], wy[i,j,k], wz[i,j,k],
                              wmag[i,j,k])
                    out.write(" ".join(f"{v:.10e}" for v in values) + "\n")


#==============================================================#
# Main processing
#==============================================================#


def main():
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_FOLDER.resolve()}")

    pattern = re.compile(r"velocityfield-step-(\d+)\.tab$")
    files = [p for p in INPUT_FOLDER.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
    files.sort(key=lambda p: int(pattern.fullmatch(p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No velocityfield-step-[number].tab files in {INPUT_FOLDER}")

    reference = None
    sums = None
    for index, path in enumerate(files, start=1):
        blocks = read_tab(path)
        coords = tuple(blocks[key] for key in ("XPTS", "YPTS", "ZPTS"))
        if reference is None:
            reference = coords
            sums = [np.zeros_like(blocks[key]) for key in ("VX", "VY", "VZ")]
        elif any(a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9)
                 for a, b in zip(coords, reference)):
            raise ValueError(f"Grid mismatch in {path.name}")

        for component, key in enumerate(("VX", "VY", "VZ")):
            sums[component] += blocks[key]
            if ADD_FREESTREAM:
                wind = blocks[("WINDX", "WINDY", "WINDZ")[component]]
                if len(wind) not in (1, len(sums[component])):
                    raise ValueError(f"Unexpected wind block length in {path.name}")
                sums[component] += wind
        print(f"[{index}/{len(files)}] {path.name}")

    means = [total / len(files) for total in sums]
    axes, fields = structured_field(*reference, *means)
    vort = calculate_vorticity(axes, fields)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_tecplot(OUTPUT_FILE, axes, fields, vort)
    print(f"\nAveraged {len(files)} snapshots")
    print(f"Grid dimensions: {tuple(len(a) for a in axes)}")
    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()