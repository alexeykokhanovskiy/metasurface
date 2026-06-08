"""
Python wrapper around rcwa_tio2_pillar.m with a disk cache.

Each unique (diameter_nm, period_nm, height_nm) → T(λ) result is saved to
data/rcwa_cache.npz so repeated CMA-ES calls are free after the first hit.

Usage
-----
    from rcwa_wrapper import RCWAWrapper
    rcwa = RCWAWrapper(wl_nm=np.linspace(400, 1000, 50))
    T = rcwa.transmission(diameter_nm=200.0)   # (50,) array
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matlab.engine
import numpy as np

ROOT     = Path(__file__).parent.parent
MAT_DIR  = ROOT / "data" / "material"
RCWA_DIR = Path(__file__).parent                    # where .m files live
V10_DIR  = ROOT.parent / "V10_2025"                 # Reticolo toolkit
CACHE_FILE = ROOT / "data" / "rcwa_cache.npz"


class RCWAWrapper:
    def __init__(
        self,
        wl_nm: np.ndarray,
        period_nm: float = 400.0,
        height_nm: float = 600.0,
        nn: int = 7,
    ):
        self.wl_nm     = np.asarray(wl_nm, dtype=float)
        self.period_nm = period_nm
        self.height_nm = height_nm
        self.nn        = nn
        self._cache: dict[str, np.ndarray] = {}
        self._eng: matlab.engine.MatlabEngine | None = None
        self._load_cache()

    # ------------------------------------------------------------------
    # MATLAB engine (lazy start)
    # ------------------------------------------------------------------

    def _start_engine(self) -> None:
        if self._eng is not None:
            return
        print("  [RCWA] Запуск MATLAB engine...")
        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(self._eng.genpath(str(V10_DIR)), nargout=0)
        self._eng.addpath(str(RCWA_DIR), nargout=0)
        self._eng.addpath(str(MAT_DIR), nargout=0)
        print("  [RCWA] MATLAB готов.")

    def close(self) -> None:
        if self._eng is not None:
            self._eng.quit()
            self._eng = None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, diameter_nm: float) -> str:
        wl_sig = f"{self.wl_nm[0]:.1f}_{self.wl_nm[-1]:.1f}_{len(self.wl_nm)}"
        tag = f"{diameter_nm:.4f}_{self.period_nm:.1f}_{self.height_nm:.1f}_{wl_sig}_nn{self.nn}"
        return hashlib.md5(tag.encode()).hexdigest()

    def _load_cache(self) -> None:
        if CACHE_FILE.exists():
            data = np.load(str(CACHE_FILE), allow_pickle=True)
            self._cache = {k: data[k] for k in data.files}
            print(f"  [Cache] Загружено {len(self._cache)} записей из {CACHE_FILE}")
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(CACHE_FILE), **self._cache)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def transmission(self, diameter_nm: float) -> np.ndarray:
        """Return T(λ) for a TiO2 pillar of given diameter. Uses cache."""
        key = self._cache_key(diameter_nm)
        if key in self._cache:
            return self._cache[key]

        self._start_engine()

        wl_ml  = matlab.double(self.wl_nm.tolist())
        T_ml, _, _ = self._eng.rcwa_tio2_pillar(
            float(diameter_nm),
            wl_ml,
            str(MAT_DIR),
            float(self.period_nm),
            float(self.height_nm),
            float(self.nn),
            nargout=3,
        )
        T = np.array(T_ml).ravel().astype(np.float32)

        self._cache[key] = T
        self._dirty = True
        return T

    def flush_cache(self) -> None:
        """Write in-memory cache to disk (call after each generation)."""
        if getattr(self, "_dirty", False):
            self._save_cache()
            self._dirty = False

    def filter_matrix(self, diameters_nm: list[float]) -> np.ndarray:
        """Return (N_filters, N_wl) transmission matrix for a list of diameters."""
        return np.stack([self.transmission(d) for d in diameters_nm], axis=0)
