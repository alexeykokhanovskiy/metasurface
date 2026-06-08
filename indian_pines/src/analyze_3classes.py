"""
Spectral range analysis for 3 classes only:
  Соя   — классы 10, 11, 12 объединены в один
  Лес   — класс 14
  Башни — класс 16

Повторяет analyze_spectral_ranges.py: NMF sweep N=2..10 по диапазонам:
  VIS  : 400 – 1000 нм
  NIR  : 1000 – 1500 нм
  SWIR : 1500 – 2500 нм
  Полный спектр
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

# Classes to keep; soybean variants merged into label 1
SOY_CLASSES    = {10, 11, 12}
FOREST_CLASS   = 14
TOWER_CLASS    = 16

LABEL_MAP = {10: 1, 11: 1, 12: 1, 14: 2, 16: 3}
LABEL_NAMES = {1: "Соя", 2: "Лес", 3: "Башни"}

RANGES = {
    "VIS (400–1000 нм)":   (400,  1000),
    "NIR (1000–1500 нм)":  (1000, 1500),
    "SWIR (1500–2500 нм)": (1500, 2500),
    "Полный спектр":       (400,  2500),
}

N_RANGE = range(2, 11)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(y, list(LABEL_MAP.keys()))
    X_f = X[keep]
    y_f = np.array([LABEL_MAP[int(c)] for c in y[keep]])
    return X_f, y_f


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
                wl, lo, hi) -> list[float]:
    mask = band_mask(wl, lo, hi)
    Xtr_sub,  shift = shift_clip(X_train[:, mask])
    Xval_sub, _     = shift_clip(X_val[:, mask], shift)

    accs = []
    for n in N_RANGE:
        model = fit_nmf(Xtr_sub, n)
        acc = rf_accuracy(model.transform(Xtr_sub),  y_train,
                          model.transform(Xval_sub), y_val)
        accs.append(acc * 100)
    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("Загрузка данных...")
    X, y, wl = load_mat()

    print("Фильтрация классов: Соя / Лес / Башни...")
    X, y = filter_classes(X, y)
    for lbl, name in LABEL_NAMES.items():
        print(f"  {name}: {(y == lbl).sum()} пикселей")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    ns = list(N_RANGE)
    results: dict[str, list[float]] = {}

    for label, (lo, hi) in RANGES.items():
        mask = band_mask(wl, lo, hi)
        print(f"\n--- {label}  ({mask.sum()} полос) ---")
        accs = sweep_range(X_train, X_val, y_train, y_val, wl, lo, hi)
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
        ax.plot(ns, accs, style, lw=1.6, ms=6, color=color, label=label)

    ax.set_xlabel("Число NMF-фильтров N", fontsize=11)
    ax.set_ylabel("Точность классификации (val), %", fontsize=11)
    ax.set_title("Соя / Лес / Башни — точность NMF по спектральным диапазонам",
                 fontsize=12)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = PLOTS / "nmf_3classes_spectral_ranges.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"\nСохранено → {path}")


if __name__ == "__main__":
    run()
