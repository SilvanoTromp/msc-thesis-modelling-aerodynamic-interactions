"""Compare VPMχ, FLIGHTLAB LL/panel, and experimental aircraft loads.

Creates five RPM-sweep and six angle-of-attack-sweep figures. Each figure
contains three forces (top row) and three moments (bottom row).

VPMχ: average the final N_LAST samples of each loading-history CSV.
FLIGHTLAB: average total-aircraft loads over the sampling phase of loads.tab.
Experiment: match operating conditions against Loads.xlsx.

Run from anywhere: python plot_load_sweeps.py
Dependencies: numpy, pandas, matplotlib, openpyxl
"""
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================
# Project paths and general settings
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VPM_DIR = PROJECT_ROOT / 'results' / 'VPMx'
FLIGHTLAB_DIR = PROJECT_ROOT / 'results' / 'FLIGHTLAB'
EXPERIMENT_FILE = PROJECT_ROOT / 'results' / 'experimental' / 'Loads.xlsx'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'load_comparison_plots'
N_LAST = 250
EXPERIMENT_SHEET = 0
SIM_SIGNS = np.ones(6)
# Experimental Z force and Z moment use the opposite sign convention.
# Order: Fx, Fy, Fz, Mx, My, Mz.
EXP_SIGNS = np.array([1, 1, -1, 1, 1, -1])
FL_LL_SIGNS = np.ones(6)
FL_PANEL_SIGNS = np.ones(6)
MISSING_FLIGHTLAB_VALUE = 0.0  # Change to np.nan to avoid plotting missing cases as zero.
MATCH_ATOL_V = 0.11
MATCH_ATOL_ALPHA = 0.11
MATCH_ATOL_RPM = 1.1

# ==============================================================
# Load definitions and filename matching
# ==============================================================

LOADS = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
FL_VARIABLES = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']
CSV_COLS = ['total_Fx_N', 'total_Fy_N', 'total_Fz_N',
            'total_Mx_Nm', 'total_My_Nm', 'total_Mz_Nm']
LABELS = [r'$F_x$ [N]', r'$F_y$ [N]', r'$F_z$ [N]',
          r'$M_x$ [N m]', r'$M_y$ [N m]', r'$M_z$ [N m]']
PATTERN = re.compile(r'^V-(?P<v>-?\d+(?:\.\d+)?)_alpha-(?P<a>-?\d+(?:\.\d+)?)_RPMs-(?P<r1>\d+)-(?P<r4>\d+)-(?P<r5>\d+)$')

# ==============================================================
# Operating conditions: RPM sweeps
# ==============================================================

RPM_SWEEPS = [
    dict(name='Pusher only', v=0.0, alpha=0.0, rpms=[(0, 0, r) for r in (3000, 4000, 5000)], xrotor='r5'),
    dict(name='Hover', v=0.0, alpha=0.0, rpms=[(r, r, 0) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Transition 1', v=5.1, alpha=1.7, rpms=[(r, 6169, 2189) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Transition 2', v=10.3, alpha=3.3, rpms=[(r, 5754, 3705) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Cruise', v=16.5, alpha=5.3, rpms=[(0, 0, r) for r in (5500, 6000, 6500, 7000)], xrotor='r5'),
]
# ==============================================================
# Operating conditions: angle-of-attack sweeps
# ==============================================================

AOA_SWEEPS = [
    dict(name='Transition 1', v=5.1, rpms=(6700, 6169, 2189)),
    dict(name='Transition 1 (airframe)', v=5.1, rpms=(0, 0, 0)),
    dict(name='Transition 2', v=10.3, rpms=(6700, 5754, 3705)),
    dict(name='Transition 2 (airframe)', v=10.3, rpms=(0, 0, 0)),
    dict(name='Cruise', v=16.5, rpms=(0, 0, 5500)),
    dict(name='Cruise (airframe)', v=16.5, rpms=(0, 0, 0)),
]
ANGLES = [-6.0, 0.0, 6.0]


# ==============================================================
# Extract operating conditions from case folder names
# ==============================================================

def case_fields(folder):
    match = PATTERN.fullmatch(folder.name)
    if match is None:
        return None
    d = match.groupdict()
    return dict(case=folder.name, v=float(d['v']), alpha=float(d['a']),
                r1=int(d['r1']), r4=int(d['r4']), r5=int(d['r5']))


# ==============================================================
# Read and average VPMχ simulation results
# ==============================================================

def read_simulations():
    # Each matching folder represents one combination of V, alpha and RPMs.
    rows = []
    if not VPM_DIR.is_dir():
        warnings.warn(f'VPMχ directory missing: {VPM_DIR}')
        return pd.DataFrame(rows)
    for folder in sorted(VPM_DIR.iterdir()):
        if not folder.is_dir():
            continue
        fields = case_fields(folder)
        if fields is None:
            continue
        csv_file = folder / 'out_cuda_run' / (folder.name + '_loading_history.csv')
        if not csv_file.is_file():
            warnings.warn(f'Missing VPMχ CSV: {csv_file}')
            continue
        try:
            df = pd.read_csv(csv_file)
            missing = set(CSV_COLS) - set(df.columns)
            if missing:
                raise ValueError(f'Missing columns: {sorted(missing)}')
            # Use the final N_LAST records to approximate steady-state loads.
            last = df[CSV_COLS].apply(pd.to_numeric, errors='coerce').tail(N_LAST)
            if len(last) < N_LAST:
                warnings.warn(f'{folder.name}: only {len(last)} VPMχ samples')
            if last.isna().any().any():
                warnings.warn(f'{folder.name}: NaNs in last {N_LAST} VPMχ rows')
            values = last.mean().to_numpy() * SIM_SIGNS
            rows.append(dict(**fields, n_samples=len(last), **dict(zip(LOADS, values))))
        except Exception as exc:
            warnings.warn(f'Could not process {csv_file}: {exc}')
    return pd.DataFrame(rows)


# ==============================================================
# FLIGHTLAB TAB reader
# ==============================================================

def read_tab_file(file_path):
    """Read !B VARIABLE NROWS NCOLUMNS numerical blocks."""
    data = {}
    variable = None
    values = []
    # A !B line starts a new variable block; other ! metadata is ignored.
    with open(file_path) as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('!B'):
                if variable is not None:
                    data[variable] = np.asarray(values, dtype=float)
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f'Invalid !B header: {line}')
                variable = parts[1].upper()
                values = []
            elif line.startswith('!'):
                continue
            elif variable is not None:
                values.extend(float(v.replace('D', 'E').replace('d', 'e')) for v in line.split())
    if variable is not None:
        data[variable] = np.asarray(values, dtype=float)
    return data


# ==============================================================
# Extract scalar FLIGHTLAB simulation parameters
# ==============================================================

def read_scalar(data, name):
    values = data[name]
    if values.size != 1:
        raise ValueError(f'{name}: expected scalar, found {values.size} values')
    return float(values[0])


# ==============================================================
# Calculate FLIGHTLAB sampling-phase mean loads
# ==============================================================

def flightlab_total_means(file_path):
    """Mean total-aircraft (component 7, index 6) loads over sampling phase."""
    data = read_tab_file(file_path)
    dt = read_scalar(data, 'TS')
    if dt <= 0:
        raise ValueError(f'Invalid TS={dt}')
    # Reconstruct phase lengths using the same ceil convention as FLIGHTLAB.
    ninit = int(np.ceil(read_scalar(data, 'TINIT') / dt))
    nconv = int(np.ceil(read_scalar(data, 'TCONV') / dt))
    nsample = int(np.ceil(read_scalar(data, 'TSAMPLE') / dt))
    nsteps = ninit + nconv + nsample
    if nsample <= 0:
        raise ValueError('Sampling interval contains no steps')
    # Sampling begins only after initialization and convergence have ended.
    sample_start = ninit + nconv
    means = []
    for variable in FL_VARIABLES:
        values = data[variable]
        if values.size != 7 * nsteps:
            raise ValueError(f'{variable}: {values.size} values, expected {7 * nsteps} ({nsteps} steps)')
        # Seven component histories are stored consecutively; index 6 is total.
        history = values.reshape(7, nsteps)[6, sample_start:nsteps]
        means.append(np.mean(history))
    return np.asarray(means, dtype=float), nsample


# ==============================================================
# Read FLIGHTLAB cases for one wing representation
# ==============================================================

def read_flightlab(model):
    model_dir = FLIGHTLAB_DIR / model
    rows = []
    if not model_dir.is_dir():
        print(f'FLIGHTLAB {model}: 0 cases (directory missing: {model_dir})')
        return pd.DataFrame(rows)
    # Allow independent sign conventions for LL and panel results.
    signs = FL_LL_SIGNS if model == 'LL' else FL_PANEL_SIGNS
    for folder in sorted(model_dir.iterdir()):
        if not folder.is_dir():
            continue
        fields = case_fields(folder)
        if fields is None:
            continue
        file_path = folder / 'loads.tab'
        if not file_path.is_file():
            warnings.warn(f'Missing FLIGHTLAB {model} loads.tab: {file_path}')
            continue
        try:
            means, nsample = flightlab_total_means(file_path)
            rows.append(dict(**fields, n_samples=nsample,
                             **dict(zip(LOADS, means * signs))))
        except Exception as exc:
            warnings.warn(f'Could not process {file_path}: {exc}')
    print(f'FLIGHTLAB {model}: {len(rows)} cases')
    return pd.DataFrame(rows)


# ==============================================================
# Read experimental load measurements
# ==============================================================

def read_experiments():
    # Excel: C=V, D/E/F=rotor RPMs, G=alpha, K:P=Fx...Mz.
    raw = pd.read_excel(EXPERIMENT_FILE, sheet_name=EXPERIMENT_SHEET, header=None,
                        usecols='C:G,K:P', engine='openpyxl')
    raw.columns = ['v', 'r1', 'r4', 'r5', 'alpha'] + LOADS
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.dropna(subset=['v', 'r1', 'r4', 'r5', 'alpha'])
    raw.loc[:, LOADS] = raw[LOADS].to_numpy() * EXP_SIGNS
    return raw.reset_index(drop=True)


# ==============================================================
# Match a requested operating condition to a dataset
# ==============================================================

def lookup(df, v, alpha, rpms):
    if df.empty:
        return None
    # Small tolerances account for rounding in filenames and workbook cells.
    mask = np.isclose(df['v'], v, atol=MATCH_ATOL_V) & np.isclose(df['alpha'], alpha, atol=MATCH_ATOL_ALPHA)
    for col, rpm in zip(('r1', 'r4', 'r5'), rpms):
        mask &= np.isclose(df[col], rpm, atol=MATCH_ATOL_RPM)
    hits = df.loc[mask]
    if hits.empty:
        return None
    if len(hits) > 1:
        warnings.warn(f'Multiple matches for V={v}, alpha={alpha}, RPMs={rpms}; averaging')
    return hits[LOADS].mean().to_numpy(dtype=float)


# ==============================================================
# Build all RPM and AoA sweep datasets
# ==============================================================

def collect_series(sources):
    series = []
    # The horizontal axis is the rotor RPM selected in each sweep definition.
    for spec in RPM_SWEEPS:
        x = [dict(zip(('r1', 'r4', 'r5'), rpm))[spec['xrotor']] for rpm in spec['rpms']]
        series.append(dict(kind='RPM', name=spec['name'], x=np.asarray(x),
                           xlabel=f"RPM ({'R1 & R4' if spec['name'] == 'Hover' else spec['xrotor'].upper()})",
                           values={key: [lookup(df, spec['v'], spec['alpha'], rpm) for rpm in spec['rpms']]
                                   for key, df in sources.items()},
                           v=spec['v'], alpha=spec['alpha'], rpms=spec['rpms']))
    # AoA sweeps keep all three rotor speeds fixed.
    for spec in AOA_SWEEPS:
        series.append(dict(kind='AoA', name=spec['name'], x=np.asarray(ANGLES),
                           xlabel='Angle of attack [deg]',
                           values={key: [lookup(df, spec['v'], a, spec['rpms']) for a in ANGLES]
                                   for key, df in sources.items()},
                           v=spec['v'], rpms=[spec['rpms']] * len(ANGLES)))
    return series


# ==============================================================
# Convert missing points into plotting arrays
# ==============================================================

def array_for(values, source):
    # Missing FLIGHTLAB data is zero by default; other sources use NaN.
    fallback = MISSING_FLIGHTLAB_VALUE if source in ('ll', 'panel') else np.nan
    return np.asarray([np.full(6, fallback) if v is None else v for v in values], dtype=float)


# ==============================================================
# Calculate consistent vertical limits for all figures
# ==============================================================

def global_limits(series):
    """Use one common force range and one common moment range in every figure."""
    limits = []

    # Group the first three load components (forces) together, and the
    # last three load components (moments) together. Forces and moments
    # have different units, so their ranges must be calculated separately.
    for component_indices in (range(3), range(3, 6)):
        vals = np.concatenate([
            array_for(values, key)[:, j]
            for s in series
            for key, values in s['values'].items()
            for j in component_indices
        ])
        vals = vals[np.isfinite(vals)]

        if not len(vals):
            common_limit = (-1, 1)
        else:
            lo, hi = float(vals.min()), float(vals.max())
            margin = max(0.08 * (hi - lo),
                         0.05 * max(abs(lo), abs(hi)), 0.001)
            common_limit = (lo - margin, hi + margin)

        # Apply exactly the same vertical limits to all three axes in
        # this row, and reuse them for every RPM and AoA sweep figure.
        limits.extend([common_limit] * 3)

    return limits


# ==============================================================
# Distinct black-only styles: line pattern, marker shape and fill.
# A slightly thicker solid line highlights VPMχ; experiments use open squares.
# Custom dash patterns help distinguish the two FLIGHTLAB models in print.
# All styles remain readable in grayscale and without color.
# ==============================================================

STYLES = {
    'vpmx': dict(label='VPMχ', color='black', marker='o',
                 linestyle='-', linewidth=2.2, markersize=6.5,
                 markerfacecolor='black', markeredgecolor='black'),
    'll': dict(label='FLIGHTLAB LL', color='black', marker='^',
               linestyle=(0, (7, 3)), linewidth=1.7, markersize=7,
               markerfacecolor='white', markeredgecolor='black'),
    'panel': dict(label='FLIGHTLAB panel', color='black', marker='D',
                  linestyle=(0, (4, 2, 1, 2)), linewidth=1.7, markersize=6,
                  markerfacecolor='white', markeredgecolor='black'),
    'exp': dict(label='Experiment', color='black', marker='s',
                linestyle=(0, (1, 2)), linewidth=1.9, markersize=7,
                markerfacecolor='black', markeredgecolor='black'),
}

# ==============================================================
# Plot one sweep with six load-component subplots
# ==============================================================

def plot_one(s, limits, destination):
    # Top row: Fx, Fy, Fz. Bottom row: Mx, My, Mz.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for key, style in STYLES.items():
            ys = array_for(s['values'][key], key)
            ax.plot(s['x'], ys[:, j], **style)
        ax.set(xlabel=s['xlabel'], ylabel=LABELS[j], ylim=limits[j])
        ax.set_xticks(s['x'])
        ax.grid(True, alpha=0.3)
    rpm_text = ', '.join(str(r) for r in s['rpms'][0]) if s['kind'] == 'AoA' else None
    context = f"V∞ = {s['v']:g} m/s | " + (f"α = {s['alpha']:g}°" if s['kind'] == 'RPM' else f'RPMs = ({rpm_text})')
    fig.suptitle(f"{s['kind']} sweep — {s['name']}\n{context}", fontsize=15, fontweight='bold')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


# ==============================================================
# Main execution: load, plot and export
# ==============================================================

def main():
    if not EXPERIMENT_FILE.is_file():
        raise FileNotFoundError(f'Experimental workbook not found: {EXPERIMENT_FILE}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load all four datasets. Missing FLIGHTLAB directories yield empty tables.
    sources = dict(vpmx=read_simulations(), ll=read_flightlab('LL'),
                   panel=read_flightlab('panel'), exp=read_experiments())
    for key, df in sources.items():
        print(f'{STYLES[key]["label"]}: {len(df)} cases/rows')
        if key != 'exp' and not df.empty:
            df.to_csv(OUTPUT_DIR / f'{key}_averages.csv', index=False)
    series = collect_series(sources)
    limits = global_limits(series)
    # Export figures and keep a record of absent case/source combinations.
    missing = []
    for i, s in enumerate(series, 1):
        filename = f"{i:02d}_{s['kind']}_{re.sub(r'[^a-zA-Z0-9]+', '_', s['name']).strip('_')}.png"
        plot_one(s, limits, OUTPUT_DIR / filename)
        print(f'Saved {filename}')
        for k, x in enumerate(s['x']):
            for source, values in s['values'].items():
                if values[k] is None:
                    missing.append(dict(sweep=s['name'], type=s['kind'], x=x,
                                        source=STYLES[source]['label'], v=s['v'],
                                        alpha=s['alpha'] if s['kind'] == 'RPM' else x,
                                        rpms=str(s['rpms'][k])))
                    if source in ('ll', 'panel'):
                        print(f'  Missing {STYLES[source]["label"]}: {s["name"]}, x={x:g} -> {MISSING_FLIGHTLAB_VALUE}')
    pd.DataFrame(missing, columns=['sweep', 'type', 'x', 'source', 'v', 'alpha', 'rpms']).to_csv(
        OUTPUT_DIR / 'missing_cases.csv', index=False)
    print(f'Finished: {len(series)} figures in {OUTPUT_DIR}; {len(missing)} missing source/case combinations.')


if __name__ == '__main__':
    main()