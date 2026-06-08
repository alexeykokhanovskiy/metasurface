"""
Diagnostic: print the full de_ti / de_ri arrays at a single wavelength
to verify we are reading the correct (0,0) zeroth order and that
energy conservation holds (sum_T + sum_R ≈ 1 for lossless).
"""
from __future__ import annotations

from pathlib import Path

import meent
import numpy as np
from scipy.interpolate import interp1d

ROOT    = Path(__file__).parent.parent
MAT_DIR = ROOT / "data" / "material"

PERIOD_NM = 400.0
HEIGHT_NM = 600.0
DIAMETER  = 300.0
WL_CHECK  = 600.0   # wavelength to examine
FTO       = 7
N_PTS     = 128


def load_nk(csv_path, wl_query_nm):
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
    return complex(n_real - 1j * k_imag)


def build_ucell(n_tio2, n_pts=N_PTS):
    ucell = np.ones((1, n_pts, n_pts), dtype=np.complex128)
    cy, cx = n_pts / 2, n_pts / 2
    r_px = (DIAMETER / PERIOD_NM) * n_pts / 2
    jj, ii = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
    mask = (ii - cy + 0.5) ** 2 + (jj - cx + 0.5) ** 2 <= r_px ** 2
    ucell[0, mask] = n_tio2
    return ucell


def main():
    n_tio2 = load_nk(MAT_DIR / "TiO2_ALD_nk.csv",         WL_CHECK)
    n_sio2 = load_nk(MAT_DIR / "SiO2_fused_silica_nk.csv", WL_CHECK)

    print(f"λ = {WL_CHECK} nm,  d = {DIAMETER} nm,  fto = {FTO}")
    print(f"n_tio2 = {n_tio2:.6f},  n_sio2 = {n_sio2:.6f}")
    print(f"fill factor = {np.pi*(DIAMETER/2)**2 / PERIOD_NM**2:.4f}")

    mee = meent.call_mee(
        backend=0,          # numpy, float64
        pol=0,
        n_top=1.0, n_bot=float(np.real(n_sio2)),
        theta=0, phi=0,
        fto=[FTO, FTO],
        wavelength=WL_CHECK,
        period=[PERIOD_NM, PERIOD_NM],
        ucell=build_ucell(n_tio2),
        thickness=[HEIGHT_NM],
        type_complex=np.complex128,
        device=0,
        fourier_type=0,
    )

    result = mee.conv_solve()

    de_ti_te = np.real(result.res_te_inc.de_ti)
    de_ri_te = np.real(result.res_te_inc.de_ri)
    de_ti_tm = np.real(result.res_tm_inc.de_ti)
    de_ri_tm = np.real(result.res_tm_inc.de_ri)

    print(f"\nde_ti shape: {de_ti_te.shape}   (expected {2*FTO+1}×{2*FTO+1} = 15×15)")

    print("\n--- TE incidence ---")
    print(f"  de_ti[{FTO},{FTO}] = {de_ti_te[FTO, FTO]:.6f}   ← (0,0) order we read as T")
    print(f"  de_ri[{FTO},{FTO}] = {de_ri_te[FTO, FTO]:.6f}   ← (0,0) order we read as R")
    print(f"  sum(de_ti)        = {de_ti_te.sum():.6f}   ← total transmitted power")
    print(f"  sum(de_ri)        = {de_ri_te.sum():.6f}   ← total reflected power")
    print(f"  T + R             = {de_ti_te.sum() + de_ri_te.sum():.6f}   ← should be ~1.0")

    idx_max_ti = np.unravel_index(np.argmax(de_ti_te), de_ti_te.shape)
    idx_max_ri = np.unravel_index(np.argmax(de_ri_te), de_ri_te.shape)
    print(f"  argmax(de_ti) = {idx_max_ti}  value = {de_ti_te[idx_max_ti]:.6f}  (order {idx_max_ti[0]-FTO},{idx_max_ti[1]-FTO})")
    print(f"  argmax(de_ri) = {idx_max_ri}  value = {de_ri_te[idx_max_ri]:.6f}  (order {idx_max_ri[0]-FTO},{idx_max_ri[1]-FTO})")

    print("\n--- TM incidence ---")
    print(f"  de_ti[{FTO},{FTO}] = {de_ti_tm[FTO, FTO]:.6f}")
    print(f"  de_ri[{FTO},{FTO}] = {de_ri_tm[FTO, FTO]:.6f}")
    print(f"  sum(de_ti)        = {de_ti_tm.sum():.6f}")
    print(f"  sum(de_ri)        = {de_ri_tm.sum():.6f}")
    print(f"  T + R             = {de_ti_tm.sum() + de_ri_tm.sum():.6f}")

    print("\n--- unpolarised T = 0.5*(TE + TM) ---")
    T_unpol = 0.5 * (de_ti_te[FTO, FTO] + de_ti_tm[FTO, FTO])
    R_unpol = 0.5 * (de_ri_te[FTO, FTO] + de_ri_tm[FTO, FTO])
    print(f"  T(0,0) = {T_unpol:.6f}")
    print(f"  R(0,0) = {R_unpol:.6f}")

    print("\n--- de_ti TE (rows=ky, cols=kx, center=[{0},{0}]) ---".format(FTO, FTO))
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(de_ti_te)


if __name__ == "__main__":
    main()
