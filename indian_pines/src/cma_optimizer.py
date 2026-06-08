"""
CMA-ES inverse design: find TiO2 pillar diameters whose transmission spectra
either maximise classification accuracy or match NMF target spectra.

Two fitness modes (--fitness flag):
  accuracy  — (default) minimise -val_accuracy of a Random Forest classifier
  nmf_mae   — minimise MAE between normalised RCWA T(λ) and NMF components
              (optimal pillar-to-component assignment via Hungarian algorithm)

Usage
-----
    python cma_optimizer.py --n 3 --range vis
    python cma_optimizer.py --n 3 --range vis --fitness nmf_mae
    python cma_optimizer.py --n 3 --range vis --resume
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cma
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from meent_wrapper import MeentWrapper
from utils import load_mat, split_data

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "optimization"
PROC_DIR    = ROOT / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANGE_PRESETS = {
    "vis":  (400,  1000),
    "nir":  (1000, 1500),
    "swir": (1500, 2500),
    "full": (400,  2500),
}

LABEL_MAP = {10: 1, 11: 1, 12: 1, 14: 2, 16: 3}

D_MIN, D_MAX = 80.0, 350.0
H_MIN, H_MAX = 200.0, 1200.0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(wl_nm: np.ndarray) -> dict:
    X, y, wl_full = load_mat()
    keep = np.isin(y, list(LABEL_MAP.keys()))
    X = X[keep]
    y = np.array([LABEL_MAP[int(c)] for c in y[keep]])

    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    def interp_to(src):
        f = interp1d(wl_full, src, axis=1, bounds_error=False, fill_value=0.0)
        return np.maximum(f(wl_nm), 0.0).astype(np.float32)

    return {"X_train": interp_to(X_train), "X_val": interp_to(X_val),
            "y_train": y_train, "y_val": y_val}


def load_nmf_targets(wl_nm: np.ndarray, nmf_file: str = "nmf_filters_n3.npz") -> np.ndarray:
    """
    Load NMF components and interpolate to wl_nm.
    Returns F_nmf (N, B) normalised to [0, 1] per component.
    """
    path = PROC_DIR / nmf_file
    d = np.load(str(path))
    wl_full = d["wavelengths"].astype(float)
    comps   = d["components"].astype(float)        # (N, 200)

    f = interp1d(wl_full, comps, axis=1, bounds_error=False, fill_value=0.0)
    F = np.maximum(f(wl_nm), 0.0)
    F /= F.max(axis=1, keepdims=True) + 1e-12      # normalise to [0, 1]
    return F.astype(np.float32)


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------

def _split_candidate(candidate: list[float], n: int) -> tuple[list[float], list[float]]:
    """Split flat [d1..dN, h1..hN] vector into (diameters, heights)."""
    return list(candidate[:n]), list(candidate[n:])


def fitness_accuracy(candidate: list[float], n_filters: int,
                     data: dict, rcwa: MeentWrapper) -> float:
    """Returns -val_accuracy (CMA-ES minimises)."""
    diameters, heights = _split_candidate(candidate, n_filters)
    T = rcwa.filter_matrix(diameters, heights)
    clf = RandomForestClassifier(n_estimators=50, max_depth=15,
                                 n_jobs=-1, random_state=42)
    clf.fit(data["X_train"] @ T.T, data["y_train"])
    acc = accuracy_score(data["y_val"], clf.predict(data["X_val"] @ T.T))
    return -acc


def fitness_nmf_mae(candidate: list[float], n_filters: int,
                    nmf_targets: np.ndarray, rcwa: MeentWrapper) -> float:
    """
    MAE between normalised RCWA T(λ) and NMF target components.

    Uses the Hungarian algorithm to find the optimal one-to-one assignment
    between RCWA spectra and NMF components before computing MAE, so the
    order of diameters in the candidate vector does not matter.

    Returns mean MAE over matched pairs (lower = better match).
    """
    diameters, heights = _split_candidate(candidate, n_filters)
    T = rcwa.filter_matrix(diameters, heights).astype(float)   # (N, B)

    # Normalise RCWA spectra to [0, 1] — same scale as NMF targets
    T_norm = T / (T.max(axis=1, keepdims=True) + 1e-12)

    # Cost matrix: MAE between each RCWA spectrum and each NMF component
    n = len(diameters)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.mean(np.abs(T_norm[i] - nmf_targets[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def run(n_filters: int, wl_lo: float, wl_hi: float, tag: str,
        fitness_mode: str, resume: bool = False) -> None:

    wl_nm    = np.linspace(wl_lo, wl_hi, 100).astype(np.float64)
    log_path = RESULTS_DIR / f"cmaes_N{n_filters}_{tag}_{fitness_mode}.csv"

    print(f"\nОптимизация: N={n_filters}, диапазон {wl_lo:.0f}–{wl_hi:.0f} нм, "
          f"fitness={fitness_mode}")

    rcwa = MeentWrapper(wl_nm=wl_nm, fto=3)

    if fitness_mode == "accuracy":
        data = prepare_data(wl_nm)
        print(f"  Train: {len(data['y_train'])}  Val: {len(data['y_val'])}")
        nmf_targets = None
    else:
        nmf_file = f"nmf_filters_n{n_filters}_{tag}.npz"
        nmf_targets = load_nmf_targets(wl_nm, nmf_file)
        print(f"  NMF цели загружены: {nmf_targets.shape}  из {nmf_file}")
        data = None

    n_params = 2 * n_filters   # [d1..dN, h1..hN]
    x0 = (np.linspace(D_MIN + 30, D_MAX - 30, n_filters).tolist()
          + [600.0] * n_filters)
    sigma = 40.0

    opts = {
        "bounds":   [[D_MIN] * n_filters + [H_MIN] * n_filters,
                     [D_MAX] * n_filters + [H_MAX] * n_filters],
        "maxiter":  500,
        "popsize":  max(8, 4 + int(3 * np.log(n_params))),
        "tolfun":   1e-4,
        "tolx":     1.0,
        "seed":     42,
        "verbose":  3,
        "CMA_stds": [sigma] * n_filters + [100.0] * n_filters,
    }

    if resume and log_path.exists():
        prev = np.loadtxt(str(log_path), delimiter=",", skiprows=1)
        if prev.ndim == 1:
            prev = prev[np.newaxis, :]
        best_idx = np.argmin(prev[:, -1])
        x0 = prev[best_idx, 1:1 + n_params].tolist()
        print(f"  Возобновление с {x0}")

    col_names = ([f"d{i+1}" for i in range(n_filters)]
                 + [f"h{i+1}" for i in range(n_filters)])
    with open(log_path, "w") as f:
        f.write("gen," + ",".join(col_names) + ",fitness\n")

    es  = cma.CMAEvolutionStrategy(x0, sigma, opts)

    gen = 0

    try:
        while not es.stop():
            gen += 1
            candidates = es.ask()
            t0 = time.time()

            if fitness_mode == "accuracy":
                fitnesses = [fitness_accuracy(list(c), n_filters, data, rcwa)
                             for c in candidates]
                metric_label = "best_val"
                metric_fmt   = lambda v: f"{-v*100:.2f}%"
            else:
                fitnesses = [fitness_nmf_mae(list(c), n_filters, nmf_targets, rcwa)
                             for c in candidates]
                metric_label = "best_MAE"
                metric_fmt   = lambda v: f"{v:.4f}"

            elapsed = time.time() - t0
            es.tell(candidates, fitnesses)

            best_f = min(fitnesses)
            best_c = candidates[np.argmin(fitnesses)]
            best_diams, best_heights = _split_candidate(list(best_c), n_filters)
            print(f"  Gen {gen:4d}  {metric_label}={metric_fmt(best_f)}  "
                  f"sigma={es.sigma:.1f}  t={elapsed:.1f}s")

            rcwa.flush_cache()

            with open(log_path, "a") as f:
                vals = ",".join(f"{v:.2f}" for v in list(best_c))
                f.write(f"{gen},{vals},{best_f:.6f}\n")

    except KeyboardInterrupt:
        print("\nПрервано пользователем.")

    finally:
        rcwa.close()

    best   = es.result.xbest
    best_f = es.result.fbest
    best_diams, best_heights = _split_candidate(list(best), n_filters)
    print(f"\n{'='*50}")
    if fitness_mode == "accuracy":
        print(f"  Лучший результат: {-best_f*100:.2f}% val accuracy")
    else:
        print(f"  Лучший результат: MAE = {best_f:.4f}")
    print(f"  Диаметры: {[f'{d:.1f}' for d in best_diams]} нм")
    print(f"  Высоты:   {[f'{h:.1f}' for h in best_heights]} нм")
    print(f"  Лог → {log_path}")
    print(f"{'='*50}")

    T_best = rcwa.filter_matrix(best_diams, best_heights)

    out = RESULTS_DIR / f"best_filters_N{n_filters}_{tag}_{fitness_mode}.npz"
    np.savez_compressed(str(out),
                        diameters_nm=np.array(best_diams),
                        heights_nm=np.array(best_heights),
                        wavelengths=wl_nm,
                        filters=T_best,
                        best_fitness=np.array(best_f))
    print(f"  Фильтры сохранены → {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, required=True)
    parser.add_argument("--range",   dest="range_name", default="vis",
                        choices=list(RANGE_PRESETS.keys()))
    parser.add_argument("--lo",      type=float, default=None)
    parser.add_argument("--hi",      type=float, default=None)
    parser.add_argument("--fitness", default="accuracy",
                        choices=["accuracy", "nmf_mae"])
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    if args.lo and args.hi:
        lo, hi, tag = args.lo, args.hi, f"{int(args.lo)}-{int(args.hi)}nm"
    else:
        lo, hi = RANGE_PRESETS[args.range_name]
        tag = args.range_name

    run(n_filters=args.n, wl_lo=lo, wl_hi=hi, tag=tag,
        fitness_mode=args.fitness, resume=args.resume)
