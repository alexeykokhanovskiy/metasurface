"""
Drop-in replacement for RCWAWrapper using meent (GPU RCWA).

Same interface: transmission(diameter_nm), filter_matrix(diameters), flush_cache(), close().
In-memory cache keyed by (diameter, period, height, wl_grid, fto).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import meent
import numpy as np
import torch
from scipy.interpolate import interp1d

ROOT    = Path(__file__).parent.parent
MAT_DIR = ROOT / "data" / "material"

DEVICE  = 1 if torch.cuda.is_available() else 0
BACKEND = 2  # PyTorch


def _load_nk(csv_path: Path, wl_query_nm: np.ndarray) -> np.ndarray:
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


def _build_ucell(diameter_nm: float, period_nm: float, n_tio2: complex,
                 n_pts: int = 128) -> np.ndarray:
    ucell = np.ones((1, n_pts, n_pts), dtype=np.complex128)
    cy, cx = n_pts / 2, n_pts / 2
    r_px   = (diameter_nm / period_nm) * n_pts / 2
    jj, ii = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
    mask = (ii - cy + 0.5) ** 2 + (jj - cx + 0.5) ** 2 <= r_px ** 2
    ucell[0, mask] = n_tio2
    return ucell


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        v = x.cpu()
        return float(v.real) if v.is_complex() else float(v)
    return float(np.real(x))


class MeentWrapper:
    def __init__(
        self,
        wl_nm: np.ndarray,
        period_nm: float = 400.0,
        height_nm: float = 600.0,
        fto: int = 3,
        n_pts: int = 128,
    ):
        self.wl_nm     = np.asarray(wl_nm, dtype=float)
        self.period_nm = period_nm
        self.height_nm = height_nm
        self.fto       = fto
        self.n_pts     = n_pts
        self._cache: dict[str, np.ndarray] = {}

        device_label = 'CUDA' if DEVICE == 1 else 'CPU'
        print(f"  [Meent] {device_label}, PyTorch, fto={fto}, N_PTS={n_pts}, "
              f"wl=[{wl_nm[0]:.0f}, {wl_nm[-1]:.0f}] nm × {len(wl_nm)}")

        self._n_tio2 = _load_nk(MAT_DIR / "TiO2_ALD_nk.csv",         self.wl_nm)
        self._n_sio2 = _load_nk(MAT_DIR / "SiO2_fused_silica_nk.csv", self.wl_nm)

    def _cache_key(self, diameter_nm: float, height_nm: float) -> str:
        wl_sig = f"{self.wl_nm[0]:.1f}_{self.wl_nm[-1]:.1f}_{len(self.wl_nm)}"
        tag = (f"{diameter_nm:.4f}_{self.period_nm:.1f}_{height_nm:.1f}"
               f"_{wl_sig}_fto{self.fto}")
        return hashlib.md5(tag.encode()).hexdigest()

    def transmission(self, diameter_nm: float, height_nm: float | None = None) -> np.ndarray:
        if height_nm is None:
            height_nm = self.height_nm
        key = self._cache_key(diameter_nm, height_nm)
        if key in self._cache:
            return self._cache[key]

        fto = self.fto
        T_out = np.zeros(len(self.wl_nm), dtype=np.float32)

        mee = meent.call_mee(
            backend=BACKEND, pol=0,
            n_top=1.0, n_bot=float(self._n_sio2[0].real),
            theta=0, phi=0,
            fto=[fto, fto],
            wavelength=float(self.wl_nm[0]),
            period=[self.period_nm, self.period_nm],
            ucell=_build_ucell(diameter_nm, self.period_nm,
                               self._n_tio2[0], self.n_pts),
            thickness=[float(height_nm)],
            type_complex=torch.complex64,
            device=DEVICE,
            fourier_type=0,
        )

        for il, wl in enumerate(self.wl_nm):
            mee.wavelength = float(wl)
            mee.ucell      = _build_ucell(diameter_nm, self.period_nm,
                                          self._n_tio2[il], self.n_pts)
            mee.n_bot      = float(self._n_sio2[il].real)
            result = mee.conv_solve()
            T_out[il] = 0.5 * (_to_float(result.res_te_inc.de_ti[fto, fto]) +
                                _to_float(result.res_tm_inc.de_ti[fto, fto]))

        self._cache[key] = T_out
        return T_out

    def filter_matrix(self, diameters_nm: list[float],
                      heights_nm: list[float] | None = None) -> np.ndarray:
        if heights_nm is None:
            heights_nm = [self.height_nm] * len(diameters_nm)
        return np.stack([self.transmission(d, h)
                         for d, h in zip(diameters_nm, heights_nm)], axis=0)

    def flush_cache(self) -> None:
        pass  # in-memory only; meent is fast enough without disk cache

    def close(self) -> None:
        pass  # no external engine to shut down
