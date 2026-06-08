"""
Spectral basis analysis for Indian Pines.

Goal: find N non-negative transmission spectra that are sufficient
for classification. Pipeline:

  1. PCA variance curve  — shows how many linear dimensions the data needs
  2. NMF sweep N=2..6    — non-negative components + RF accuracy per N
  3. Plot NMF components for the best N as candidate filter spectra

Usage
-----
    python pca_analysis.py
    python pca_analysis.py --n 3     # fix N instead of sweeping
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_mat, split_data, zscore

ROOT  = Path(__file__).parent.parent
PLOTS = ROOT / "data" / "plots"
PROC  = ROOT / "data" / "processed"
PLOTS.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data() -> dict:
    X, y, wl = load_mat()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    # z-score for PCA; keep raw non-negative copy for NMF
    Xz_train, Xz_val, Xz_test, _, _ = zscore(X_train, X_val, X_test)
    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        Xz_train=Xz_train, Xz_val=Xz_val, Xz_test=Xz_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        wl=wl,
    )

# ---------------------------------------------------------------------------
# PCA variance
# ---------------------------------------------------------------------------

def plot_pca_variance(X_train: np.ndarray, wl: np.ndarray) -> None:
    pca = PCA().fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, lw=1.5)
    for thr, col in [(90, "orange"), (95, "tomato"), (99, "crimson")]:
        n = int(np.searchsorted(cumvar, thr)) + 1
        ax.axhline(thr, color=col, ls="--", lw=0.9, label=f"{thr}% → N={n}")
        ax.axvline(n,   color=col, ls=":",  lw=0.9)
    ax.axvline(3, color="steelblue", ls="-", lw=1.2, label="N=3 (цель)")
    ax.set_xlim(1, 40)
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Накопленная объяснённая дисперсия (%)")
    ax.set_title("PCA — дисперсия спектров Indian Pines")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS / "pca_variance.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Сохранено → {path}")

# ---------------------------------------------------------------------------
# NMF helpers
# ---------------------------------------------------------------------------

def _shift_nonneg(X: np.ndarray, shift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shift each band so min=0, clip any residual negatives. Returns (X_shifted, shift)."""
    if shift is None:
        shift = X.min(axis=0)
    return np.maximum(X - shift, 0), shift


def fit_nmf(X_train: np.ndarray, n: int, seed: int = 42) -> NMF:
    model = NMF(n_components=n, init="nndsvda", max_iter=10000, tol=1e-4, random_state=seed)
    model.fit(X_train)
    return model


def rf_accuracy(X_tr, y_tr, X_val, y_val) -> float:
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                 n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_val, clf.predict(X_val))

# ---------------------------------------------------------------------------
# NMF sweep
# ---------------------------------------------------------------------------

def nmf_sweep(data: dict, n_range: range) -> dict[int, float]:
    """Fit NMF for each N, evaluate RF val accuracy. Returns {n: accuracy}."""
    X_raw, shift = _shift_nonneg(data["X_train"])
    X_val_raw, _ = _shift_nonneg(data["X_val"], shift)

    results = {}
    print(f"\n  {'N':>3}  {'Точность (val)':>16}")
    print("  " + "-" * 22)
    for n in n_range:
        model = fit_nmf(X_raw, n)
        Xtr_proj  = model.transform(X_raw)
        Xval_proj = model.transform(X_val_raw)
        acc = rf_accuracy(Xtr_proj, data["y_train"], Xval_proj, data["y_val"])
        results[n] = acc
        print(f"  {n:>3}  {acc*100:>14.1f}%")
    return results


def plot_nmf_sweep(results: dict[int, float]) -> None:
    ns = list(results.keys())
    accs = [results[n] * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, accs, "o-", lw=1.5, ms=6)
    for n, acc in zip(ns, accs):
        ax.annotate(f"{acc:.1f}%", (n, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Число фильтров N")
    ax.set_ylabel("Точность классификации (val), %")
    ax.set_title("NMF: точность vs число неотрицательных фильтров")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS / "nmf_accuracy_sweep.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Сохранено → {path}")

# ---------------------------------------------------------------------------
# Plot NMF components
# ---------------------------------------------------------------------------

def plot_nmf_components(model: NMF, wl: np.ndarray, n: int) -> None:
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 3.5), squeeze=False)
    cmap = plt.cm.tab10

    for i in range(n):
        ax = axes[0][i]
        component = model.components_[i]
        component_norm = component / component.max()  # normalise to [0, 1]
        ax.plot(wl, component_norm, lw=1.5, color=cmap(i))
        ax.fill_between(wl, component_norm, alpha=0.15, color=cmap(i))
        ax.set_xlabel("Длина волны, нм", fontsize=9)
        ax.set_ylabel("Пропускание (норм.)", fontsize=9)
        ax.set_title(f"Фильтр {i + 1}", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"NMF — {n} неотрицательных спектральных фильтра (кандидаты)", fontsize=11)
    fig.tight_layout()
    path = PLOTS / f"nmf_components_n{n}.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Сохранено → {path}")


def save_nmf_components(model: NMF, wl: np.ndarray, n: int) -> None:
    components_norm = model.components_ / model.components_.max(axis=1, keepdims=True)
    np.savez_compressed(
        str(PROC / f"nmf_filters_n{n}.npz"),
        components=components_norm,
        wavelengths=wl,
    )
    print(f"  Компоненты сохранены → {PROC / f'nmf_filters_n{n}.npz'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_fixed: int | None = None) -> None:
    print("Загрузка данных...")
    data = prepare_data()
    wl = data["wl"]

    print("\n--- PCA: анализ дисперсии ---")
    plot_pca_variance(data["Xz_train"], wl)

    n_range = range(2, 11)
    print("\n--- NMF: перебор числа фильтров ---")
    sweep = nmf_sweep(data, n_range)
    plot_nmf_sweep(sweep)

    n = n_fixed if n_fixed is not None else 3
    print(f"\n--- NMF: компоненты для N={n} ---")
    X_raw, shift = _shift_nonneg(data["X_train"])
    model = fit_nmf(X_raw, n)
    plot_nmf_components(model, wl, n)
    save_nmf_components(model, wl, n)

    # Final accuracy on test set
    X_val_raw,  _ = _shift_nonneg(data["X_val"],  shift)
    X_test_raw, _ = _shift_nonneg(data["X_test"], shift)
    Xtr  = model.transform(X_raw)
    Xval = model.transform(X_val_raw)
    Xte  = model.transform(X_test_raw)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(Xtr, data["y_train"])
    val_acc  = accuracy_score(data["y_val"],  clf.predict(Xval))
    test_acc = accuracy_score(data["y_test"], clf.predict(Xte))

    # Full-band baseline
    clf_full = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    clf_full.fit(data["Xz_train"], data["y_train"])
    full_acc = accuracy_score(data["y_test"], clf_full.predict(data["Xz_test"]))

    print(f"\n{'='*45}")
    print(f"  Полный спектр (200 полос) test: {full_acc*100:.1f}%")
    print(f"  NMF N={n} val:                  {val_acc*100:.1f}%")
    print(f"  NMF N={n} test:                 {test_acc*100:.1f}%")
    print(f"{'='*45}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Число фильтров для финального графика (по умолчанию 3)")
    args = parser.parse_args()
    run(n_fixed=args.n)
