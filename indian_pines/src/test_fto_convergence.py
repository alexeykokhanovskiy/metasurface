"""
Test whether increasing FTO makes meent converge to Reticolo for d=300nm.

Uses numpy backend (float64, CPU) to avoid GPU precision/stability issues.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import meent
import numpy as np
from scipy.interpolate import interp1d

from rcwa_wrapper import RCWAWrapper

ROOT    = Path(__file__).parent.parent
MAT_DIR = ROOT / "data" / "material"
PLOTS   = ROOT / "data" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PERIOD_NM = 400.0
HEIGHT_NM = 600.0
DIAMETER  = 300.0
WL_NM     = np.linspace(400, 1000, 50)
N_PTS     = 128


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


def build_ucell(n_tio2: complex, n_pts: int = N_PTS) -> np.ndarray:
    ucell = np.ones((1, n_pts, n_pts), dtype=np.complex128)
    cy, cx = n_pts / 2, n_pts / 2
    r_px   = (DIAMETER / PERIOD_NM) * n_pts / 2
    jj, ii = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
    mask = (ii - cy + 0.5) ** 2 + (jj - cx + 0.5) ** 2 <= r_px ** 2
    ucell[0, mask] = n_tio2
    return ucell


def meent_spectrum_numpy(n_tio2_arr: np.ndarray, n_sio2_arr: np.ndarray,
                         fto: int) -> np.ndarray:
    T_out = np.zeros(len(WL_NM))

    mee = meent.call_mee(
        backend=0, pol=0,
        n_top=1.0, n_bot=float(n_sio2_arr[0].real),
        theta=0, phi=0,
        fto=[fto, fto],
        wavelength=float(WL_NM[0]),
        period=[PERIOD_NM, PERIOD_NM],
        ucell=build_ucell(n_tio2_arr[0]),
        thickness=[HEIGHT_NM],
        type_complex=np.complex128,
        device=0,
        fourier_type=0,
    )

    for il, wl in enumerate(WL_NM):
        mee.wavelength = float(wl)
        mee.ucell      = build_ucell(n_tio2_arr[il])
        mee.n_bot      = float(n_sio2_arr[il].real)
        result = mee.conv_solve()
        T_out[il] = 0.5 * (float(np.real(result.res_te_inc.de_ti[fto, fto])) +
                            float(np.real(result.res_tm_inc.de_ti[fto, fto])))

    return T_out


def main():
    n_tio2 = load_nk(MAT_DIR / "TiO2_ALD_nk.csv",         WL_NM)
    n_sio2 = load_nk(MAT_DIR / "SiO2_fused_silica_nk.csv", WL_NM)

    print("Загрузка Reticolo (nn=7)...")
    rcwa = RCWAWrapper(wl_nm=WL_NM, period_nm=PERIOD_NM, height_nm=HEIGHT_NM, nn=7)
    T_ret7 = rcwa.transmission(DIAMETER).astype(float)

    print("Загрузка Reticolo (nn=11)...")
    rcwa11 = RCWAWrapper(wl_nm=WL_NM, period_nm=PERIOD_NM, height_nm=HEIGHT_NM, nn=11)
    T_ret11 = rcwa11.transmission(DIAMETER).astype(float)

    rcwa.flush_cache(); rcwa.close()
    rcwa11.flush_cache(); rcwa11.close()

    fto_list = [5, 7, 9, 11]
    T_meent  = {}
    maes     = {}

    print(f"\n{'FTO':>5}  {'MAE vs ret7':>12}  {'t (s)':>8}")
    print('-' * 32)

    for fto in fto_list:
        print(f"  fto={fto} ...", end='', flush=True)
        t0 = time.time()
        T = meent_spectrum_numpy(n_tio2, n_sio2, fto)
        elapsed = time.time() - t0
        mae = float(np.mean(np.abs(T - T_ret7)))
        T_meent[fto] = T
        maes[fto] = mae
        print(f"  MAE={mae:.4f}  {elapsed:.1f}s")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(WL_NM, T_ret7,  'k-',  lw=2, label='Reticolo nn=7')
    ax1.plot(WL_NM, T_ret11, 'k--', lw=2, label='Reticolo nn=11')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for i, fto in enumerate(fto_list):
        ax1.plot(WL_NM, T_meent[fto], color=colors[i], lw=1.5, label=f'meent fto={fto}')
    ax1.set_xlabel('Длина волны (нм)')
    ax1.set_ylabel('Пропускание T₀')
    ax1.set_title(f'd={DIAMETER:.0f} нм — спектры')
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)

    ax2.plot(fto_list, [maes[f] for f in fto_list], 'o-', color='steelblue', lw=2, ms=8)
    for fto in fto_list:
        ax2.annotate(f'{maes[fto]:.4f}', (fto, maes[fto]),
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    ax2.axhline(float(np.mean(np.abs(T_ret11 - T_ret7))), color='k', ls='--',
                label='Reticolo nn=11 vs nn=7')
    ax2.set_xlabel('FTO meent')
    ax2.set_ylabel('MAE vs Reticolo nn=7')
    ax2.set_title('Сходимость по FTO — d=300 нм')
    ax2.set_xticks(fto_list)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig.suptitle(f'FTO convergence: meent numpy float64 vs Reticolo, d={DIAMETER:.0f} нм', fontsize=11)
    fig.tight_layout()
    out = PLOTS / "fto_convergence_d300.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\nГрафик → {out}")


if __name__ == "__main__":
    main()
