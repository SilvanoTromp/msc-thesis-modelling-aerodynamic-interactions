"""Compare last-250-sample VPMχ loads against experimental Loads.xlsx.

Run from anywhere: python plot_vpmx_load_sweeps.py
Dependencies: numpy, pandas, matplotlib, openpyxl (for reading XLSX).
Adjust PROJECT_ROOT or column/coordinate settings below if necessary.
"""
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The script is expected to live three directories below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VPM_DIR = PROJECT_ROOT / 'results' / 'VPMx'
EXPERIMENT_FILE = PROJECT_ROOT / 'results' / 'experimental' / 'Loads.xlsx'
OUTPUT_DIR = VPM_DIR / 'load_comparison_plots'
N_LAST = 250
EXPERIMENT_SHEET = 0  # Change to the sheet name if necessary.
# Set to -1 if a force/moment axis differs between numerical and experimental conventions.
SIM_SIGNS = np.ones(6)
EXP_SIGNS = np.ones(6)
MATCH_ATOL_V = 0.11  # m/s, to accommodate rounding in filenames / spreadsheet
MATCH_ATOL_ALPHA = 0.11  # degrees
MATCH_ATOL_RPM = 1.1

LOADS = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
CSV_COLS = ['total_Fx_N', 'total_Fy_N', 'total_Fz_N',
            'total_Mx_Nm', 'total_My_Nm', 'total_Mz_Nm']
LABELS = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [N m]', 'My [N m]', 'Mz [N m]']
PATTERN = re.compile(r'^V-(?P<v>-?\d+(?:\.\d+)?)_alpha-(?P<a>-?\d+(?:\.\d+)?)_RPMs-(?P<r1>\d+)-(?P<r4>\d+)-(?P<r5>\d+)$')

# 5 RPM sweeps. For the hover sweep, R1 and R4 change together.
RPM_SWEEPS = [
    dict(name='Pusher only', v=0.0, alpha=0.0, rpms=[(0, 0, r) for r in (3000, 4000, 5000)], xrotor='r5'),
    dict(name='Hover', v=0.0, alpha=0.0, rpms=[(r, r, 0) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Transition 1', v=5.1, alpha=1.7, rpms=[(r, 6169, 2189) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Transition 2', v=10.3, alpha=3.3, rpms=[(r, 5754, 3705) for r in (3000, 4000, 5000, 6000)], xrotor='r1'),
    dict(name='Cruise', v=16.5, alpha=5.3, rpms=[(0, 0, r) for r in (5500, 6000, 6500, 7000)], xrotor='r5'),
]

# 6 AoA sweeps. The 3 angles in the supplied test matrix are -6, 0, +6 deg.
AOA_SWEEPS = [
    dict(name='Transition 1', v=5.1, rpms=(6700, 6169, 2189)),
    dict(name='Transition 1 (airframe)', v=5.1, rpms=(0, 0, 0)),
    dict(name='Transition 2', v=10.3, rpms=(6700, 5754, 3705)),
    dict(name='Transition 2 (airframe)', v=10.3, rpms=(0, 0, 0)),
    dict(name='Cruise', v=16.5, rpms=(0, 0, 5500)),
    dict(name='Cruise (airframe)', v=16.5, rpms=(0, 0, 0)),
]
ANGLES = [-6.0, 0.0, 6.0]


def read_simulations():
    rows = []
    for folder in sorted(VPM_DIR.iterdir()):
        if not folder.is_dir():
            continue
        match = PATTERN.fullmatch(folder.name)
        if not match:
            continue
        d = match.groupdict()
        csv_file = folder / 'out_cuda_run' / (folder.name + '_loading_history.csv')
        if not csv_file.is_file():
            warnings.warn(f'Missing CSV: {csv_file}')
            continue
        try:
            df = pd.read_csv(csv_file)
            missing = set(CSV_COLS) - set(df.columns)
            if missing:
                raise ValueError(f'Missing columns: {sorted(missing)}')
            last = df[CSV_COLS].apply(pd.to_numeric, errors='coerce').tail(N_LAST)
            if len(last) < N_LAST:
                warnings.warn(f'{folder.name}: only {len(last)} rows (expected {N_LAST})')
            if last.isna().any().any():
                warnings.warn(f'{folder.name}: NaNs in last {N_LAST} rows')
            values = last.mean().to_numpy() * SIM_SIGNS
            rows.append(dict(case=folder.name, v=float(d['v']), alpha=float(d['a']),
                             r1=int(d['r1']), r4=int(d['r4']), r5=int(d['r5']),
                             n_samples=len(last), **dict(zip(LOADS, values))))
        except Exception as exc:
            warnings.warn(f'Could not process {csv_file}: {exc}')
    return pd.DataFrame(rows)


def read_experiments():
    # Excel columns: C=V, D/E/F=RPM1/4/5, G=AoA, K:P=loads.
    # Read without a header to avoid assumptions about title rows.
    raw = pd.read_excel(EXPERIMENT_FILE, sheet_name=EXPERIMENT_SHEET, header=None,
                        usecols='C:G,K:P', engine='openpyxl')
    raw.columns = ['v', 'r1', 'r4', 'r5', 'alpha'] + LOADS
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.dropna(subset=['v', 'r1', 'r4', 'r5', 'alpha'])
    raw.loc[:, LOADS] = raw[LOADS].to_numpy() * EXP_SIGNS
    return raw.reset_index(drop=True)


def lookup(df, v, alpha, rpms):
    if df.empty:
        return None
    mask = np.isclose(df['v'], v, atol=MATCH_ATOL_V) & np.isclose(df['alpha'], alpha, atol=MATCH_ATOL_ALPHA)
    for c, rpm in zip(('r1', 'r4', 'r5'), rpms):
        mask &= np.isclose(df[c], rpm, atol=MATCH_ATOL_RPM)
    hits = df.loc[mask]
    if hits.empty:
        return None
    if len(hits) > 1:
        warnings.warn(f'Multiple matches for V={v}, alpha={alpha}, RPMs={rpms}; averaging duplicates')
    return hits[LOADS].mean().to_numpy(dtype=float)


def collect_series(sim, exp):
    series = []
    for spec in RPM_SWEEPS:
        x = [dict(zip(('r1', 'r4', 'r5'), rpm))[spec['xrotor']] for rpm in spec['rpms']]
        y_sim = [lookup(sim, spec['v'], spec['alpha'], rpm) for rpm in spec['rpms']]
        y_exp = [lookup(exp, spec['v'], spec['alpha'], rpm) for rpm in spec['rpms']]
        series.append(dict(kind='RPM', name=spec['name'], x=np.array(x),
                           xlabel=f"RPM ({'R1 & R4' if spec['name'] == 'Hover' else spec['xrotor'].upper()})",
                           sim=y_sim, exp=y_exp, v=spec['v'], alpha=spec['alpha'], rpms=spec['rpms']))
    for spec in AOA_SWEEPS:
        y_sim = [lookup(sim, spec['v'], a, spec['rpms']) for a in ANGLES]
        y_exp = [lookup(exp, spec['v'], a, spec['rpms']) for a in ANGLES]
        series.append(dict(kind='AoA', name=spec['name'], x=np.array(ANGLES),
                           xlabel='Angle of attack [deg]', sim=y_sim, exp=y_exp,
                           v=spec['v'], rpms=[spec['rpms']] * len(ANGLES)))
    return series


def array_for(values):
    return np.array([np.full(6, np.nan) if v is None else v for v in values], dtype=float)


def global_limits(series):
    # Shared limits PER COMPONENT across all 11 figures, not one limit for all forces.
    limits = []
    for j in range(6):
        vals = np.concatenate([np.r_[array_for(s['sim'])[:, j], array_for(s['exp'])[:, j]] for s in series])
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            limits.append((-1, 1))
        else:
            lo, hi = float(vals.min()), float(vals.max())
            margin = max(0.08 * (hi - lo), 0.05 * max(abs(lo), abs(hi)), 0.001)
            limits.append((lo - margin, hi + margin))
    return limits


def plot_one(s, limits, destination):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ys = array_for(s['sim'])
    ye = array_for(s['exp'])
    for j, ax in enumerate(axes.flat):
        ax.plot(s['x'], ys[:, j], 'o-', color='#1764aa', linewidth=1.8, markersize=6, label='VPMχ')
        ax.plot(s['x'], ye[:, j], 's--', color='#d36b24', linewidth=1.7, markersize=5.5, label='Experiment')
        ax.set(xlabel=s['xlabel'], ylabel=LABELS[j], ylim=limits[j])
        ax.set_xticks(s['x'])
        ax.grid(True, alpha=0.3)
    rpm_text = ', '.join(str(r) for r in s['rpms'][0]) if s['kind'] == 'AoA' else None
    context = f"V∞ = {s['v']:g} m/s | " + (f"α = {s['alpha']:g}°" if s['kind'] == 'RPM' else f"RPMs = ({rpm_text})")
    fig.suptitle(f"{s['kind']} sweep — {s['name']}\n{context}", fontsize=15, fontweight='bold')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def main():
    if not VPM_DIR.is_dir():
        raise FileNotFoundError(f'VPMx directory not found: {VPM_DIR}. Set PROJECT_ROOT above.')
    if not EXPERIMENT_FILE.is_file():
        raise FileNotFoundError(f'Experimental workbook not found: {EXPERIMENT_FILE}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim, exp = read_simulations(), read_experiments()
    print(f'Found {len(sim)} simulation cases and {len(exp)} experimental rows')
    if not sim.empty:
        sim.to_csv(OUTPUT_DIR / 'vpmx_last250_averages.csv', index=False)
    series = collect_series(sim, exp)
    limits = global_limits(series)
    missing = []
    for i, s in enumerate(series, 1):
        filename = f"{i:02d}_{s['kind']}_{re.sub(r'[^a-zA-Z0-9]+', '_', s['name']).strip('_')}.png"
        plot_one(s, limits, OUTPUT_DIR / filename)
        print(f'Saved {filename}')
        for k, x in enumerate(s['x']):
            for source in ('sim', 'exp'):
                if s[source][k] is None:
                    missing.append(dict(sweep=s['name'], type=s['kind'], x=x,
                                        source=source, v=s['v'], alpha=s.get('alpha', x) if s['kind']=='RPM' else x,
                                        rpms=str(s['rpms'][k])))
    pd.DataFrame(missing, columns=['sweep', 'type', 'x', 'source', 'v', 'alpha', 'rpms']).to_csv(
        OUTPUT_DIR / 'missing_cases.csv', index=False)
    print(f'Finished: 11 figures in {OUTPUT_DIR}; {len(missing)} missing source/case combinations.')


if __name__ == '__main__':
    main()
