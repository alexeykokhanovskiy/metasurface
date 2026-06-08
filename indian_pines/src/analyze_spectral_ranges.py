"""
NMF classification accuracy across spectral sub-ranges.

Evaluates how well N non-negative filters perform when restricted to:
  VIS  : 400 – 1000 nm
  NIR  : 1000 – 1500 nm
  SWIR : 1500 – 2500 nm
  Full : 400 – 2500 nm  (all bands)

For each range: NMF sweep N=2..10, RF val accuracy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_mat, split_data, zscore

PLOTS = Path(__file__).parent.parent / "data" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

RANGES = {
    "VIS (400–1000 нм)":   (400,    1000),
    "NIR (1000–1500 нм)":  (1000,   1500),
    "SWIR (1500–2500 нм)": (1500,   2500),
    "Полный спектр":       (400,    2500),
}

N_RANGE = range(2, 11)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def band_mask(wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (wl >= lo) & (wl <= hi)


def shift_clip(X: np.ndarray, shift: np.ndarray | None = None):
    if shift is None:
        shift = X.min(axis=0)
    return np.maximum(X - shift, 0.0), shift


def fit_nmf(X: np.ndarray, n: int) -> NMF:
    model = NMF(n_components=n, init="nndsvda", max_iter=10000, tol=1e-4, random_state=42)
    model.fit(X)
    return model


def rf_accuracy(Xtr, ytr, Xval, yval) -> float:
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                 n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    return accuracy_score(yval, clf.predict(Xval))


# ---------------------------------------------------------------------------
# Sweep for one spectral range
# ---------------------------------------------------------------------------

def sweep_range(X_train, X_val, y_train, y_val,
                wl, lo, hi, n_range) -> list[float]:
    mask = band_mask(wl, lo, hi)
    n_bands = mask.sum()
    if n_bands == 0:
        return [0.0] * len(n_range)

    Xtr_sub  = X_train[:, mask]
    Xval_sub = X_val[:, mask]

    Xtr_nn,  shift = shift_clip(Xtr_sub)
    Xval_nn, _     = shift_clip(Xval_sub, shift)

    accs = []
    for n in n_range:
        if n > n_bands:
            accs.append(float("nan"))
            continue
        model = fit_nmf(Xtr_nn, n)
        acc = rf_accuracy(model.transform(Xtr_nn),  y_train,
                          model.transform(Xval_nn), y_val)
        accs.append(acc * 100)
    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("Загрузка данных...")
    X, y, wl = load_mat()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print()

    ns = list(N_RANGE)
    results: dict[str, list[float]] = {}

    for label, (lo, hi) in RANGES.items():
        mask = band_mask(wl, lo, hi)
        print(f"\n--- {label}  ({mask.sum()} полос) ---")
        accs = sweep_range(X_train, X_val, y_train, y_val, wl, lo, hi, N_RANGE)
        results[label] = accs
        for n, acc in zip(ns, accs):
            print(f"  N={n:2d}  {acc:.1f}%")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    styles = ["-o", "-s", "-^", "-D"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, accs), color, style in zip(results.items(), colors, styles):
        valid = [(n, a) for n, a in zip(ns, accs) if not np.isnan(a)]
        if valid:
            vns, vaccs = zip(*valid)
            ax.plot(vns, vaccs, style, lw=1.6, ms=6, color=color, label=label)

    ax.set_xlabel("Число NMF-фильтров N", fontsize=11)
    ax.set_ylabel("Точность классификации (val), %", fontsize=11)
    ax.set_title("Точность NMF по спектральным диапазонам", fontsize=12)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = PLOTS / "nmf_spectral_ranges.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"\nСохранено → {path}")


if __name__ == "__main__":
    run()
