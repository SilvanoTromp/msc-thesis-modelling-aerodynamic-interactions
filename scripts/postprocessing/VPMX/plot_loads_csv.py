
"""
Plot rotor, wing, and total aerodynamic loads from a CSV file.

Expected columns:
    revs
    rotor1_Fx_N, rotor1_Fy_N, rotor1_Fz_N
    rotor1_Mx_Nm, rotor1_My_Nm, rotor1_Mz_Nm
    ... same for rotor2, rotor3, and wing.

Total loads are calculated as the sum of the available components.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==============================================================
# Settings
# ==============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

CSV_FILE = Path("C:/Users/Silva/Documents/VPMX/DelftBlue output/V-10.30_alpha-0_RPMs-6700-5754-3705[v2]/out_cuda_run/V-10.30_alpha-0_RPMs-6700-5754-3705_loading_history.csv")

df = pd.read_csv(CSV_FILE, sep=None, engine="python")
df.columns = df.columns.str.strip()

# Number of final revolutions used for averaging
AVERAGE_LAST_REVS = 3.0

# Show figures interactively
SHOW_PLOTS = True

# Load components
COMPONENTS = ["rotor_0", "rotor_1", "rotor_2", "wing"]

# Quantities and units
QUANTITIES = {
    "Fx_N": ("Fx", "Force [N]"),
    "Fy_N": ("Fy", "Force [N]"),
    "Fz_N": ("Fz", "Force [N]"),
    "Mx_Nm": ("Mx", "Moment [N m]"),
    "My_Nm": ("My", "Moment [N m]"),
    "Mz_Nm": ("Mz", "Moment [N m]"),
}


# ==============================================================
# Read CSV
# ==============================================================

def read_loads(csv_file):
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    if "revs" not in df.columns:
        raise ValueError("Required column 'revs' not found.")

    df["revs"] = pd.to_numeric(df["revs"], errors="coerce")

    if df["revs"].isna().any():
        raise ValueError("Column 'revs' contains invalid values.")

    df = df.sort_values("revs").reset_index(drop=True)

    return df


# ==============================================================
# Calculate total loads
# ==============================================================

def calculate_totals(df):
    for quantity in QUANTITIES:
        columns = [
            f"{component}_{quantity}"
            for component in COMPONENTS
            if f"{component}_{quantity}" in df.columns
        ]

        if not columns:
            raise ValueError(
                f"No columns found for quantity '{quantity}'."
            )

        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        missing = [
            f"{component}_{quantity}"
            for component in COMPONENTS
            if f"{component}_{quantity}" not in df.columns
        ]

        if missing:
            print(f"Warning: Missing columns: {', '.join(missing)}")

        df[f"total_{quantity}"] = df[columns].sum(
            axis=1, min_count=len(columns)
        )

    return df




# ==============================================================
# Plot loads
# ==============================================================

def plot_loads(df):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 9),
        sharex=True
    )

    axes = axes.flatten()

    for ax, (quantity, (symbol, ylabel)) in zip(
        axes, QUANTITIES.items()
    ):

        for component in COMPONENTS:
            column = f"{component}_{quantity}"

            if column in df.columns:
                ax.plot(
                    df["revs"],
                    df[column],
                    label=component.capitalize(),
                    linewidth=1.2,
                )

        ax.plot(
            df["revs"],
            df[f"total_{quantity}"],
            label="Total",
            color="black",
            linewidth=2.2,
        )

        ax.set_xlabel("Revolutions [-]")
        ax.set_ylabel(ylabel)
        ax.set_title(symbol)
        ax.grid(True, alpha=0.3)

    # One shared legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02)
    )

    # Leave space for shared legend
    fig.tight_layout(rect=[0, 0.07, 1, 1])


# ==============================================================
# Calculate and print averaged loads
# ==============================================================

def calculate_averages(df):
    last_rev = df["revs"].max()
    start_rev = last_rev - AVERAGE_LAST_REVS

    averaging_data = df.loc[df["revs"] >= start_rev]

    if averaging_data.empty:
        raise ValueError("No data available for averaging.")

    print("\n" + "=" * 58)
    print("AVERAGED TOTAL AERODYNAMIC LOADS")
    print("=" * 58)
    print(f"Revolution range: {start_rev:.3f} to {last_rev:.3f}")
    print(f"Number of samples: {len(averaging_data)}")
    print("-" * 58)

    for quantity, (symbol, ylabel) in QUANTITIES.items():
        column = f"total_{quantity}"
        average = averaging_data[column].mean()

        unit = "N m" if "Moment" in ylabel else "N"

        print(f"{symbol:<4}: {average:>12.6f} {unit}")

    print("=" * 58)


# ==============================================================
# Main
# ==============================================================


def main():
    print(f"Reading: {CSV_FILE}")

    df = read_loads(CSV_FILE)
    df = calculate_totals(df)

    # Print averaged loads in terminal
    calculate_averages(df)

    # Create interactive plots
    if SHOW_PLOTS:
        plot_loads(df)
        plt.show()


if __name__ == "__main__":
    main()