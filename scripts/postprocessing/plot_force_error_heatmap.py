"""
Compare FLIGHTLAB and VPMchi force-vector errors against experimental loads.

The script:
1. Imports the readers and operating conditions from compare_loads.py.
2. Reads experimental, FLIGHTLAB, and VPMchi aircraft loads.
3. Matches operating conditions in the experimental test matrix.
4. Calculates the Euclidean error of the three force components.
5. Organizes RPM sweeps, reference airframe cases, and AoA sweeps.
6. Generates one figure with side-by-side FLIGHTLAB and VPMchi heatmaps.
7. Applies an identical color scale to both numerical methods.
8. Displays operating conditions and error values inside the table.
9. Saves the combined figure to results/force_error_heatmaps.

All force-vector errors are in N.
"""

from pathlib import Path
import importlib.util
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

# ==============================================================
# Settings
# ==============================================================

# Existing load-comparison script containing data readers
COMPARISON_SCRIPT = Path(__file__).resolve().with_name("compare_loads.py")

# Figure settings
FIGURE_SIZE = (23, 12)
FIGURE_DPI = 220

# Green indicates low error; red indicates high error
COLORMAP = "RdYlGn_r"

# None uses the maximum error across both numerical methods
COLOR_MAX = None

# Display force-vector error values inside the cells
SHOW_VALUES = True

# Output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_FILE = OUTPUT_DIR / "force_error_heatmaps.png"

# ==============================================================
# Import the existing readers and case definitions
# ==============================================================

def load_comparison_module():
    """Reuse the original readers, sign conventions and case definitions."""

    # ----------------------------------------------------------
    # Check that the existing comparison script is available
    # ----------------------------------------------------------

    if not COMPARISON_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Cannot find comparison script: {COMPARISON_SCRIPT}\n"
            "Set COMPARISON_SCRIPT to your existing comparison script filename."
        )

    # ----------------------------------------------------------
    # Import readers and operating-condition definitions
    # ----------------------------------------------------------

    spec = importlib.util.spec_from_file_location("load_comparison", COMPARISON_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================
# Build matrix rows in the same order as the test matrix
# ==============================================================

def make_cases(module):
    """Build test-matrix sections with fixed conditions in row labels."""
    sections = []

    # ----------------------------------------------------------
    # RPM sweep cases
    # ----------------------------------------------------------

    rpm_rows = []
    for spec in module.RPM_SWEEPS:
        varied = ("r1", "r4") if spec["name"] == "Hover" else (spec["xrotor"],)
        fixed = [(key, value) for key, value in zip(("r1", "r4", "r5"), spec["rpms"][0])
                 if key not in varied]
        conditions = [rf"V_\infty={spec['v']:g}\,\mathrm{{m/s}}",
                      rf"\alpha={spec['alpha']:g}^\circ"]
        conditions += [rf"\mathrm{{RPM}}_{{{key[1:]}}}={value:g}" for key, value in fixed]
        rpm_rows.append(dict(
            label=spec["name"],
            conditions=r",\; ".join(conditions),
            cases=[(spec["v"], spec["alpha"], rpms) for rpms in spec["rpms"]],
            sublabels=[", ".join(rf"\mathrm{{RPM}}_{{{key[1:]}}}={dict(zip(('r1','r4','r5'), rpms))[key]:g}"
                                 for key in varied) for rpms in spec["rpms"]],
        ))
    sections.append(("RPM Sweep", rpm_rows))

    # ----------------------------------------------------------
    # Reference airframe cases
    # ----------------------------------------------------------

    reference_rows = []
    for name, v, alpha in (("Transition 1", 5.1, 1.7),
                           ("Transition 2", 10.3, 3.3),
                           ("Cruise", 16.5, 5.3)):
        reference_rows.append(dict(
            label=name,
            conditions=(rf"V_\infty={v:g}\,\mathrm{{m/s}},\; "
                        rf"\alpha={alpha:g}^\circ,\; "
                        r"\mathrm{RPM}_{1,4,5}=0"),
            cases=[(v, alpha, (0, 0, 0))],
            sublabels=[r"\mathrm{RPM}_{1,4,5}=0"],
        ))
    sections.append(("Reference Airframe Cases", reference_rows))

    # ----------------------------------------------------------
    # Angle-of-attack sweep cases
    # ----------------------------------------------------------

    aoa_rows = []
    for spec in module.AOA_SWEEPS:
        r1, r4, r5 = spec["rpms"]
        aoa_rows.append(dict(
            label=spec["name"],
            conditions=(rf"V_\infty={spec['v']:g}\,\mathrm{{m/s}},\; "
                        rf"\mathrm{{RPM}}_1={r1},\; "
                        rf"\mathrm{{RPM}}_4={r4},\; "
                        rf"\mathrm{{RPM}}_5={r5}"),
            cases=[(spec["v"], angle, spec["rpms"]) for angle in module.ANGLES],
            sublabels=[rf"\alpha={angle:g}^\circ" for angle in module.ANGLES],
        ))
    sections.append(("Angle-of-Attack Sweep", aoa_rows))
    return sections

# ==============================================================
# Calculate Euclidean force-vector errors
# ==============================================================

def force_error(experiment, numerical):
    """Return ||F_num - F_exp||_2 in N, or NaN if unavailable."""
    if experiment is None or numerical is None:
        return np.nan

    # ----------------------------------------------------------
    # Extract force components and validate numerical values
    # ----------------------------------------------------------

    exp = np.asarray(experiment[:3], dtype=float)
    num = np.asarray(numerical[:3], dtype=float)
    if not np.all(np.isfinite(exp)) or not np.all(np.isfinite(num)):
        return np.nan

    # Euclidean norm of the force-vector difference [N]
    return float(np.linalg.norm(num - exp))

def collect_errors(module, sections, sources):
    """Compute errors using the original case matching function."""

    # ----------------------------------------------------------
    # Match experimental and numerical loads for every case
    # ----------------------------------------------------------

    for _, rows in sections:
        for row in rows:
            row["errors"] = []
            for v, alpha, rpms in row["cases"]:
                exp = module.lookup(sources["exp"], v, alpha, rpms)
                errors = {}
                for key in ("flightlab", "vpmx"):
                    numerical = module.lookup(sources[key], v, alpha, rpms)
                    errors[key] = force_error(exp, numerical)
                row["errors"].append(errors)

# ==============================================================
# Draw combined FLIGHTLAB and VPMchi heatmaps
# ==============================================================

def plot_heatmaps(sections, destination, vmax):
    """Draw both methods side by side with one shared color scale."""

    # ----------------------------------------------------------
    # Insert section headings as full-width table rows
    # ----------------------------------------------------------

    display_rows = []

    for title, group in sections:
        display_rows.append(("section", title, None))
        display_rows.extend(("case", None, row) for row in group)

    ncols = max(
        len(row["cases"])
        for _, group in sections
        for row in group
    )
    nrows = len(display_rows)

    # ----------------------------------------------------------
    # Configure shared color mapping and create figure
    # ----------------------------------------------------------

    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad("#dedede")
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(
        1, 2,
        figsize=FIGURE_SIZE,
        sharey=True,
        gridspec_kw={"wspace": 0.025},
    )

    from matplotlib.patches import Rectangle

    # ----------------------------------------------------------
    # Draw each numerical method using identical normalization
    # ----------------------------------------------------------

    for axis, method in zip(axes, ("flightlab", "vpmx")):

        matrix = np.full((nrows, ncols), np.nan)

        for i, (kind, _, row) in enumerate(display_rows):
            if kind == "case":
                for j, errors in enumerate(row["errors"]):
                    matrix[i, j] = errors[method]

        axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )

        # ------------------------------------------------------
        # Draw section headings and operating-condition cells
        # ------------------------------------------------------

        for i, (kind, title, row) in enumerate(display_rows):

            if kind == "section":
                axis.add_patch(Rectangle(
                    (-0.5, i - 0.5), ncols, 1,
                    facecolor="white", edgecolor="#333333",
                    linewidth=1.1, zorder=3,
                ))
                axis.text(
                    (ncols - 1) / 2, i, title,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", zorder=4,
                )
                continue

            # Unused cells are white; missing results are grey.
            for j in range(len(row["cases"]), ncols):
                axis.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="white", edgecolor="white", zorder=2,
                ))

            for j, (sublabel, errors) in enumerate(
                zip(row["sublabels"], row["errors"])
            ):
                value = errors[method]
                cell_text = rf"${sublabel}$"

                if SHOW_VALUES:
                    cell_text += (
                        f"\n{value:.1f} N"
                        if np.isfinite(value) else "\n—"
                    )

                rgba = (
                    cmap(norm(value))
                    if np.isfinite(value)
                    else (0.87, 0.87, 0.87, 1)
                )
                luminance = (
                    0.2126 * rgba[0]
                    + 0.7152 * rgba[1]
                    + 0.0722 * rgba[2]
                )

                axis.text(
                    j, i, cell_text,
                    ha="center", va="center", fontsize=8.3,
                    linespacing=1.5,
                    color="black" if luminance > 0.48 else "white",
                )

        # ------------------------------------------------------
        # Format grid and axis labels
        # ------------------------------------------------------

        axis.set_xticks(np.arange(-0.5, ncols + 0.5, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, nrows + 0.5, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xticks([])
        axis.set_yticks(np.arange(nrows))
        axis.tick_params(axis="y", length=0, pad=10)

        # Distinguish methods without adding a figure title.
        axis.set_xlabel(
            "FLIGHTLAB" if method == "flightlab" else r"VPM$\chi$",
            fontsize=12, fontweight="bold", labelpad=12,
        )

    # ----------------------------------------------------------
    # Show operating conditions only beside the left heatmap
    # ----------------------------------------------------------

    labels = [
        row["label"] + "\n" + rf"${row['conditions']}$"
        if kind == "case" else ""
        for kind, _, row in display_rows
    ]

    axes[0].set_yticklabels(labels, fontsize=8.5, linespacing=1.4)
    for label in axes[0].get_yticklabels():
        label.set_horizontalalignment("right")

    axes[1].tick_params(axis="y", labelleft=False)

    # ----------------------------------------------------------
    # Adjust figure layout and reserve space for colorbar
    # ----------------------------------------------------------

    fig.subplots_adjust(
        left=0.28,
        right=0.89,
        top=0.975,
        bottom=0.075,
        wspace=0.025,
    )

    # ----------------------------------------------------------
    # Add one common force-error colorbar
    # ----------------------------------------------------------

    cbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.50])

    mappable = ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(
        mappable,
        cax=cbar_ax,
    )

    cbar.set_label(
        r"Force-vector error $\|\mathbf{F}_{num}-\mathbf{F}_{exp}\|_2$ [N]"
    )

    # ----------------------------------------------------------
    # Save combined figure
    # ----------------------------------------------------------

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

def shared_color_max(sections):
    """Use identical error-to-color mapping for both methods."""

    # ----------------------------------------------------------
    # Collect finite errors from both numerical methods
    # ----------------------------------------------------------

    values = [error[method]
              for _, rows in sections for row in rows
              for error in row["errors"] for method in ("flightlab", "vpmx")]
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        warnings.warn("No matching experimental/numerical cases found.")

    # Apply user-defined limit or maximum observed error
    maximum = COLOR_MAX if COLOR_MAX is not None else (float(finite.max()) if finite.size else 1.0)
    return max(float(maximum), 1e-12)

# ==============================================================
# Main execution
# ==============================================================

def main():

    # ----------------------------------------------------------
    # Import comparison script and read all datasets
    # ----------------------------------------------------------

    module = load_comparison_module()
    sources = {
        "exp": module.read_experiments(),
        "flightlab": module.read_flightlab(),
        "vpmx": module.read_simulations(),
    }

    # ----------------------------------------------------------
    # Assemble test matrix and calculate force-vector errors
    # ----------------------------------------------------------

    sections = make_cases(module)
    collect_errors(module, sections, sources)
    vmax = shared_color_max(sections)

    # ----------------------------------------------------------
    # Generate separate heatmaps using the same color scale
    # ----------------------------------------------------------

    plot_heatmaps(sections, OUTPUT_FILE, vmax)
    print(f"Saved: {OUTPUT_FILE}")

    # ----------------------------------------------------------
    # Print final summary
    # ----------------------------------------------------------

    for title, rows in sections:
        n = sum(len(row["cases"]) for row in rows)
        print(f"{title}: {n} operating conditions")

if __name__ == "__main__":
    main()