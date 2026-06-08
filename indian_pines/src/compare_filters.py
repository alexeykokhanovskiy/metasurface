"""
Compare NMF spectral basis with CMA-ES optimised RCWA filters.

Loads:
  - NMF components from data/processed/nmf_filters_n3.npz
  - Best diameters from the CMA-ES log CSV
  - T(λ) for those diameters from the RCWA cache (no new MATLAB calls)

Produces one figure with two panels:
  Left  — NMF basis spectra (normalised to [0,1]), restricted to the same
          wavelength range as the RCWA result
  Right — Physical T(λ) from RCWA for the optimised pillar diameters

Also prints per-filter classification comparison: accuracy when using NMF
filters vs RCWA filters as the sensor response.

Usage
-----
    python compare_filters.py                        # auto-detect latest log
    python compare_filters.py --log cmaes_N3_vis.csv --nmf nmf_filters_n3.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from rcwa_wrapper import RCWAWrapper
from utils import load_mat, split_data

ROOT      = Path(__file__).parent.parent
OPT_DIR   = ROOT / "data" / "optimization"
PROC_DIR  = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP   = {10: 1, 11: 1, 12: 1, 14: 2, 16: 3}
LABEL_NAMES = {1: "Соя", 2: "Лес", 3: "Башни"}

RANGE_MAP = {"vis": (400, 1000), "nir": (1000, 1500),
             "swir": (1500, 2500), "full": (400, 2500)}


# ---------------------------------------------------------------------------

def load_log_best(log_path: Path):
    data = np.loadtxt(str(log_path), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    diameters = data[:, 1:-1]
    fitnesses = data[:, -1]
    best_idx = int(np.argmin(fitnesses))
    return diameters[best_idx], -fitnesses[best_idx]


def get_wl_range(tag: str):
    for key, (lo, hi) in RANGE_MAP.items():
        if key in tag:
            return lo, hi
    return 400, 1000


def prepare_data(wl_nm):
    X, y, wl_full = load_mat()
    keep = np.isin(y, list(LABEL_MAP.keys()))
    X = X[keep]
    y = np.array([LABEL_MAP[int(c)] for c in y[keep]])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    def interp(src):
        f = interp1d(wl_full, src, axis=1, bounds_error=False, fill_value=0.0)
        return np.maximum(f(wl_nm), 0.0).astype(np.float32)

    return (interp(X_train), interp(X_val), interp(X_test),
            y_train, y_val, y_test)


def rf_accuracy(F, X_train, X_val, y_train, y_val):
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 n_jobs=-1, random_state=42)
    clf.fit(X_train @ F.T, y_train)
    return accuracy_score(y_val, clf.predict(X_val @ F.T))


# ---------------------------------------------------------------------------

def run(log_name: str | None, nmf_name: str) -> None:
    # --- CMA-ES log ---
    if log_name:
        log_path = OPT_DIR / log_name
    else:
        csvs = sorted(OPT_DIR.glob("cmaes_*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise FileNotFoundError(f"Нет CSV-логов в {OPT_DIR}")
        log_path = csvs[-1]

    tag = log_path.stem
    wl_lo, wl_hi = get_wl_range(tag)
    wl_nm = np.linspace(wl_lo, wl_hi, 60).astype(np.float64)

    best_d, cma_val_acc = load_log_best(log_path)
    n_filters = len(best_d)
    print(f"\nЛог       : {log_path.name}")
    print(f"Диаметры  : {[f'{d:.1f}' for d in best_d]} нм")
    print(f"Val acc   : {cma_val_acc*100:.2f}%  (CMA-ES лог)")

    # --- RCWA filters ---
    print("\nЗагрузка T(λ) из RCWA кэша...")
    rcwa = RCWAWrapper(wl_nm=wl_nm)
    F_rcwa = rcwa.filter_matrix(list(best_d))   # (N, 60)
    rcwa.close()

    # --- NMF filters ---
    nmf_path = PROC_DIR / nmf_name
    nmf_data = np.load(str(nmf_path))
    wl_full  = nmf_data["wavelengths"].astype(float)
    comps    = nmf_data["components"].astype(float)   # (N, 200)

    # Interpolate NMF components to the same wl_nm grid
    f_interp = interp1d(wl_full, comps, axis=1, bounds_error=False, fill_value=0.0)
    F_nmf_raw = np.maximum(f_interp(wl_nm), 0.0)

    # Normalise each NMF component to [0, 1]
    F_nmf = F_nmf_raw / (F_nmf_raw.max(axis=1, keepdims=True) + 1e-12)

    # --- Classification comparison ---
    print("\nПодготовка данных для классификации...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(wl_nm)

    acc_rcwa = rf_accuracy(F_rcwa, X_train, X_val, y_train, y_val)
    acc_nmf  = rf_accuracy(F_nmf,  X_train, X_val, y_train, y_val)

    print(f"\n{'─'*40}")
    print(f"  RCWA (CMA-ES)  val accuracy : {acc_rcwa*100:.2f}%")
    print(f"  NMF            val accuracy : {acc_nmf*100:.2f}%")
    print(f"{'─'*40}")

    # --- Plot ---
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_filters))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = axes[0]
    for i, (comp, d) in enumerate(zip(F_nmf, range(n_filters))):
        ax.plot(wl_nm, comp, color=colors[i], linewidth=2,
                label=f"Компонента {d+1}")
    ax.set_title(f"NMF фильтры (нормир.)\nval acc = {acc_nmf*100:.2f}%")
    ax.set_xlabel("Длина волны (нм)")
    ax.set_ylabel("Нормированная интенсивность")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    for i, (T, d) in enumerate(zip(F_rcwa, best_d)):
        ax.plot(wl_nm, T, color=colors[i], linewidth=2,
                label=f"d = {d:.0f} нм")
    ax.set_title(f"RCWA фильтры (CMA-ES)\nval acc = {acc_rcwa*100:.2f}%")
    ax.set_xlabel("Длина волны (нм)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    fig.suptitle(
        f"Сравнение NMF и RCWA фильтров — {wl_lo}–{wl_hi} нм  |  N = {n_filters}",
        fontsize=12
    )
    fig.tight_layout()

    save_path = PLOTS_DIR / f"{tag}_nmf_vs_rcwa.png"
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"\n  График → {save_path.name}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--nmf", default="nmf_filters_n3.npz")
    args = parser.parse_args()
    run(args.log, args.nmf)
