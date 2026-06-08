"""
Compare meent (Python RCWA, GPU) vs Reticolo (MATLAB RCWA) for a TiO2 circular
nanopillar on SiO2. Both solvers use the same 50-point wavelength grid.

Reticolo is called directly via RCWAWrapper (MATLAB engine, lazy start).
meent runs on CUDA via PyTorch backend (falls back to CPU if unavailable).

This script tests two meent Fourier factorization modes:
  fourier_type=0  DFS (standard)
  fourier_type=1  CFS (Li's rules — correct for TM at discontinuities)

Usage
-----
    python test_meent_vs_reticolo.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import meent
import numpy as np
import torch
from scipy.interpolate import interp1d

from rcwa_wrapper import RCWAWrapper

ROOT    = Path(__file__).parent.parent
MAT_DIR = ROOT / "data" / "material"
PLOTS   = ROOT / "data" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PERIOD_NM = 400.0
HEIGHT_NM = 600.0
DIAMETERS = [100, 200, 300]
WL_NM     = np.linspace(400, 1000, 50)   # shared grid for both solvers

FTO   = 7    # Fourier orders ±7 → (2×7+1)² = 225 harmonics — matches Reticolo nn=[7,7]
N_PTS = 128  # raster pixels per period

DEVICE  = 1 if torch.cuda.is_available() else 0   # meent: 1=CUDA, 0=CPU
BACKEND = 2                                        # PyTorch


# ---------------------------------------------------------------------------
# Material data (for meent — Reticolo loads its own via load_nk.m)
# ---------------------------------------------------------------------------

def load_nk(csv_path: Path, wl_query_nm: np.ndarray) -> np.ndarray:
    wl_tbl, n_tbl, k_tbl = [], [], []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#' or line[0].isalpha():
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                wl_tbl.append(float(parts[0]))
                n_tbl.append(float(parts[1]))
                k_tbl.append(float(parts[2]))
            except ValueError:
                continue
    wl_tbl = np.array(wl_tbl)
    n_real = interp1d(wl_tbl, n_tbl, kind='linear', fill_value='extrapolate')(wl_query_nm)
    k_imag = np.maximum(
        interp1d(wl_tbl, k_tbl, kind='linear', fill_value='extrapolate')(wl_query_nm), 0.0)
    return (n_real - 1j * k_imag).astype(np.complex128)


# ---------------------------------------------------------------------------
# meent
# ---------------------------------------------------------------------------

def build_ucell(diameter_nm: float, n_tio2: complex, n_pts: int = N_PTS) -> np.ndarray:
    ucell = np.ones((1, n_pts, n_pts), dtype=np.complex128)
    cy, cx = n_pts / 2, n_pts / 2
    r_px   = (diameter_nm / PERIOD_NM) * n_pts / 2
    jj, ii = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
    mask = (ii - cy + 0.5) ** 2 + (jj - cx + 0.5) ** 2 <= r_px ** 2
    ucell[0, mask] = n_tio2
    return ucell


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        v = x.cpu()
        return float(v.real) if v.is_complex() else float(v)
    return float(np.real(x))


def meent_spectrum(diameter_nm: float, n_tio2_arr: np.ndarray,
                   n_sio2_arr: np.ndarray,
                   fto: int = FTO,
                   n_pts: int = N_PTS,
                   fourier_type: int = 0,
                   device: int = DEVICE,
                   type_complex=torch.complex64,
                   backend: int = BACKEND) -> tuple[np.ndarray, np.ndarray]:
    T_out = np.zeros(len(WL_NM))
    R_out = np.zeros(len(WL_NM))

    # numpy backend (0) uses np.complex128; torch backend (2) uses torch types
    tc = np.complex128 if backend == 0 else type_complex

    mee = meent.call_mee(
        backend=backend, pol=0,
        n_top=1.0, n_bot=float(n_sio2_arr[0].real),
        theta=0, phi=0,
        fto=[fto, fto],
        wavelength=float(WL_NM[0]),
        period=[PERIOD_NM, PERIOD_NM],
        ucell=build_ucell(diameter_nm, n_tio2_arr[0], n_pts),
        thickness=[HEIGHT_NM],
        type_complex=tc,
        device=device,
        fourier_type=fourier_type,
    )

    for il, wl in enumerate(WL_NM):
        mee.wavelength = float(wl)
        mee.ucell      = build_ucell(diameter_nm, n_tio2_arr[il], n_pts)
        mee.n_bot      = float(n_sio2_arr[il].real)
        result = mee.conv_solve()
        T_out[il] = 0.5 * (_to_float(result.res_te_inc.de_ti[fto, fto]) +
                            _to_float(result.res_tm_inc.de_ti[fto, fto]))
        R_out[il] = 0.5 * (_to_float(result.res_te_inc.de_ri[fto, fto]) +
                            _to_float(result.res_tm_inc.de_ri[fto, fto]))

    return T_out, R_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_label = 'CUDA' if DEVICE == 1 else 'CPU'
    print(f"meent: {device_label}, PyTorch, fto={FTO}, N_PTS={N_PTS}")

    # --- Material data for meent ---
    n_tio2 = load_nk(MAT_DIR / "TiO2_ALD_nk.csv",         WL_NM)
    n_sio2 = load_nk(MAT_DIR / "SiO2_fused_silica_nk.csv", WL_NM)

    # --- Reticolo via MATLAB engine (same WL_NM grid) ---
    print("Запуск MATLAB engine для Reticolo...")
    rcwa = RCWAWrapper(wl_nm=WL_NM, period_nm=PERIOD_NM, height_nm=HEIGHT_NM)
    T_reticolo = {}
    for d in DIAMETERS:
        print(f"  Reticolo d={d} nm ...", end='', flush=True)
        t0 = time.time()
        T_reticolo[d] = rcwa.transmission(float(d)).astype(float)
        print(f"  {time.time()-t0:.1f}s")
    rcwa.flush_cache()
    rcwa.close()
    print()

    # -----------------------------------------------------------------------
    # Part 1: meent vs Reticolo for all diameters
    # -----------------------------------------------------------------------
    configs = [
        dict(label='meent', backend=BACKEND, fourier_type=0, device=DEVICE, type_complex=torch.complex64, ls='--'),
    ]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    header = f"{'D (nm)':>8}  {'T_ret_min':>10}"
    for cfg in configs:
        header += f"  {cfg['label'][:10]:>10}  {'MAE':>8}"
    print(header)
    print('-' * (len(header) + 10))

    results: dict[int, dict] = {d: {'Reticolo': T_reticolo[d]} for d in DIAMETERS}

    for cfg in configs:
        lbl = cfg['label']
        print(f"\n--- {lbl} ---")
        for k, d in enumerate(DIAMETERS):
            print(f"  d={d} nm ...", end='', flush=True)
            t0 = time.time()
            try:
                T_mee, _ = meent_spectrum(
                    float(d), n_tio2, n_sio2,
                    fourier_type=cfg['fourier_type'],
                    device=cfg['device'],
                    type_complex=cfg['type_complex'],
                    backend=cfg['backend'],
                )
                elapsed = time.time() - t0
                mae = float(np.mean(np.abs(T_mee - T_reticolo[d])))
                print(f"  MAE={mae:.4f}  {elapsed:.1f}s")
                results[d][lbl] = T_mee
            except Exception as e:
                print(f"  ERROR: {e}")
                results[d][lbl] = None

    # Plot
    ls_ret = '-'
    lbl_colors = {'meent': 'tab:blue'}
    for k, d in enumerate(DIAMETERS):
        ax = axes[k]
        ax.plot(WL_NM, T_reticolo[d], 'k-', linewidth=2, label='Reticolo', zorder=5)
        for cfg in configs:
            lbl = cfg['label']
            T = results[d].get(lbl)
            if T is not None:
                ax.plot(WL_NM, T, color=lbl_colors[lbl], linestyle=cfg['ls'],
                        linewidth=1.5, label=lbl)
        ax.set_title(f'd = {d} нм')
        ax.set_xlabel('Длина волны (нм)')
        if k == 0:
            ax.set_ylabel('Пропускание T₀')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)

    fig.suptitle(
        f'Reticolo vs meent DFS/CFS  |  fto={FTO}, N_PTS={N_PTS}, {device_label}\n'
        f'TiO₂ на SiO₂, высота={HEIGHT_NM:.0f} нм, период={PERIOD_NM:.0f} нм',
        fontsize=10)
    fig.tight_layout()
    out = PLOTS / "reticolo_vs_meent_dfs_cfs.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\nГрафик → {out}")

    # -----------------------------------------------------------------------
    # Part 2: MAE summary table
    # -----------------------------------------------------------------------
    print(f"\n{'D (nm)':>8}  {'Reticolo min':>12}", end='')
    for cfg in configs:
        print(f"  {cfg['label']:>16}  {'MAE':>8}", end='')
    print()
    print('-' * (12 + 14 * len(configs) + 10))

    for d in DIAMETERS:
        T_ret = T_reticolo[d]
        print(f"  d={d:3d} nm  {T_ret.min():>12.4f}", end='')
        for cfg in configs:
            lbl = cfg['label']
            T = results[d].get(lbl)
            if T is not None:
                mae = float(np.mean(np.abs(T - T_ret)))
                print(f"  {T.min():>16.4f}  {mae:>8.4f}", end='')
            else:
                print(f"  {'ERROR':>16}  {'N/A':>8}", end='')
        print()


if __name__ == "__main__":
    main()
