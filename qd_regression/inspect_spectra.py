"""
Visualise spectra as a 2-D heatmap:
  x  = wavelength (nm)
  y  = sample (sorted by concentration)
  color = spectral intensity

Run:
  python qd_regression/inspect_spectra.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WL_MIN, WL_MAX = 380.0, 880.0
N_WAVELENGTHS = 1000
TARGET_WL = np.linspace(WL_MIN, WL_MAX, N_WAVELENGTHS)

COMPOUNDS = {
    'Fe':   'FeCl3',
    'FeCo': 'FeCl3 CoCl2',
}


def parse_concentration(s: str) -> float:
    s = s.strip()
    if s.endswith('uM'):
        return float(s[:-2]) * 1e-6
    elif s.endswith('mM'):
        return float(s[:-2]) * 1e-3
    elif s.endswith('M'):
        return float(s[:-1])
    raise ValueError(f"Cannot parse: {s!r}")


def build_matrix(df_compound: pd.DataFrame, one_per_concentration: bool = False):
    """Return (X, conc_molar, conc_str) sorted by concentration ascending."""
    if one_per_concentration:
        first_src = df_compound.groupby('Concentration')['Source File'].first()
        mask = df_compound.apply(
            lambda r: r['Source File'] == first_src[r['Concentration']], axis=1
        )
        df_compound = df_compound[mask]

    rows = []
    for (src, conc_str), grp in df_compound.groupby(['Source File', 'Concentration']):
        grp_crop = grp[
            (grp['Wavelength (nm)'] >= WL_MIN) &
            (grp['Wavelength (nm)'] <= WL_MAX)
        ].sort_values('Wavelength (nm)')
        if len(grp_crop) < 10:
            continue
        intensity = np.interp(
            TARGET_WL,
            grp_crop['Wavelength (nm)'].values,
            grp_crop['Intensity (a.u.)'].values,
        )
        conc_m = grp_crop['conc_molar'].iloc[0]
        rows.append({'intensity': intensity, 'conc_molar': conc_m,
                     'conc_str': conc_str, 'src': src})

    rows.sort(key=lambda r: r['conc_molar'])
    X = np.array([r['intensity'] for r in rows])
    conc_molar = np.array([r['conc_molar'] for r in rows])
    conc_str = [r['conc_str'] for r in rows]
    return X, conc_molar, conc_str


def plot_heatmap(X: np.ndarray, conc_molar: np.ndarray, conc_str: list,
                 label: str, ax: plt.Axes):
    n_samples = X.shape[0]

    # Normalize each spectrum to [0,1] so intensities are comparable across concentrations
    X_norm = X - X.min(axis=1, keepdims=True)
    row_max = X_norm.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    X_norm = X_norm / row_max

    im = ax.imshow(
        X_norm,
        aspect='auto',
        origin='lower',
        extent=[WL_MIN, WL_MAX, -0.5, n_samples - 0.5],
        cmap='inferno',
        vmin=0, vmax=1,
    )

    # y-ticks: one per sample, labelled with concentration
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(conc_str, fontsize=8)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Sample (sorted by concentration ↑)')
    ax.set_title(f'{label}  ({n_samples} spectra)', fontweight='bold')

    return im


def main():
    csv_path = Path(__file__).parent / 'final_cleaned_data.csv'
    out_dir = Path(__file__).parent

    df = pd.read_csv(csv_path)
    df_f = df[(df['Time'] == '0 hrs') & (df['Concentration'] != 'Unknown')].copy()
    df_f['conc_molar'] = df_f['Concentration'].apply(parse_concentration)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))

    for ax, (label, compound) in zip(axes, COMPOUNDS.items()):
        sub = df_f[df_f['Compound'] == compound]
        dedup = (label == 'FeCo')
        X, conc_molar, conc_str = build_matrix(sub, one_per_concentration=dedup)

        print(f"{label}: {X.shape[0]} spectra, "
              f"conc {conc_molar.min():.2e} – {conc_molar.max():.2e} M")

        im = plot_heatmap(X, conc_molar, conc_str, label, ax)
        plt.colorbar(im, ax=ax, label='Normalised intensity')

    plt.suptitle(
        'PL spectra heatmap (0 hrs, 380–880 nm, per-spectrum normalised)\n'
        'y = sample sorted by concentration ↑,  color = intensity',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = out_dir / 'spectra_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out}")
    plt.show()


if __name__ == '__main__':
    main()
