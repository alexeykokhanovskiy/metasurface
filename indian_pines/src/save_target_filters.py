"""
Fit NMF on a chosen spectral range and save the resulting filter spectra
for use as targets in the inverse design problem.

Usage
-----
    python save_target_filters.py --n 3 --range vis
    python save_target_filters.py --n 4 --range nir
    python save_target_filters.py --n 3 --range full
    python save_target_filters.py --n 3 --lo 800 --hi 1800   # custom range

Spectral range shortcuts:
    vis  : 400 – 1000 nm
    nir  : 1000 – 1500 nm
    swir : 1500 – 2500 nm
    full : 400 – 2500 nm

Output: data/filters/target_filters_N{n}_{range}.npz
    wavelengths : (B,)   float32  — wavelengths in nm for the selected range
    filters     : (N, B) float32  — filter spectra normalised to [0, 1]
    n_filters   : int
    wl_min      : float
    wl_max      : float
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from utils import load_mat, split_data

FILTERS_DIR = Path(__file__).parent.parent / "data" / "filters"
PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
PLOTS_DIR   = Path(__file__).parent.parent / "data" / "plots"
FILTERS_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANGE_PRESETS = {
    "vis":  (400,  1000),
    "nir":  (1000, 1500),
    "swir": (1500, 2500),
    "full": (400,  2500),
}

SOY_CLASSES = {10, 11, 12}
LABEL_MAP   = {10: 1, 11: 1, 12: 1, 14: 2, 16: 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_classes(X, y):
    keep = np.isin(y, list(LABEL_MAP.keys()))
    return X[keep], np.array([LABEL_MAP[int(c)] for c in y[keep]])


def band_mask(wl, lo, hi):
    return (wl >= lo) & (wl <= hi)


def shift_clip(X, shift=None):
    if shift is None:
        shift = X.min(axis=0)
    return np.maximum(X - shift, 0.0), shift


def fit_nmf(X, n):
    model = NMF(n_components=n, init="nndsvda", max_iter=10000, tol=1e-4, random_state=42)
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# Save + plot
# ---------------------------------------------------------------------------

def save_filters(wavelengths: np.ndarray, filters: np.ndarray,
                 n: int, lo: float, hi: float, tag: str) -> Path:
    fname = f"target_filters_N{n}_{tag}.npz"
    path = FILTERS_DIR / fname
    np.savez_compressed(
        str(path),
        wavelengths=wavelengths.astype(np.float32),
        filters=filters.astype(np.float32),
        n_filters=np.array(n),
        wl_min=np.array(lo),
        wl_max=np.array(hi),
    )
    print(f"Сохранено → {path}")

    # Also save in the format expected by cma_optimizer.py (nmf_mae fitness mode)
    proc_path = PROC_DIR / f"nmf_filters_n{n}_{tag}.npz"
    np.savez_compressed(
        str(proc_path),
        wavelengths=wavelengths.astype(np.float32),
        components=filters.astype(np.float32),
    )
    print(f"Сохранено (optimizer) → {proc_path}")
    return path


def plot_filters(wavelengths: np.ndarray, filters: np.ndarray,
                 n: int, tag: str) -> None:
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 3.5), squeeze=False)
    cmap = plt.cm.tab10

    for i in range(n):
        ax = axes[0][i]
        ax.plot(wavelengths, filters[i], lw=1.5, color=cmap(i))
        ax.fill_between(wavelengths, filters[i], alpha=0.15, color=cmap(i))
        ax.set_xlabel("Длина волны, нм", fontsize=9)
        ax.set_ylabel("Пропускание (норм.)", fontsize=9)
        ax.set_title(f"Фильтр {i + 1}", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Целевые фильтры N={n}  [{tag}]  (Соя / Лес / Башни)", fontsize=11)
    fig.tight_layout()

    plot_path = PLOTS_DIR / f"target_filters_N{n}_{tag}.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.show()
    print(f"График → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n: int, lo: float, hi: float, tag: str) -> None:
    print(f"\nПараметры: N={n}, диапазон {lo:.0f}–{hi:.0f} нм  [{tag}]")

    print("Загрузка данных...")
    X, y, wl = load_mat()
    X, y = filter_classes(X, y)
    print(f"  Всего пикселей: {len(y)}  (Соя: {(y==1).sum()}, "
          f"Лес: {(y==2).sum()}, Башни: {(y==3).sum()})")

    X_train, _, _, y_train, _, _ = split_data(X, y)

    mask = band_mask(wl, lo, hi)
    wl_sub = wl[mask]
    print(f"  Полос в диапазоне: {mask.sum()}")

    if mask.sum() < n:
        raise ValueError(f"Полос ({mask.sum()}) меньше числа фильтров ({n})")

    Xtr_sub, _ = shift_clip(X_train[:, mask])

    print("Подбор NMF...")
    model = fit_nmf(Xtr_sub, n)

    # Normalise each component to [0, 1]
    components = model.components_.copy()
    components /= components.max(axis=1, keepdims=True)

    print("\nФильтры (макс. нормированы к 1):")
    for i, comp in enumerate(components):
        peak_wl = wl_sub[np.argmax(comp)]
        print(f"  Фильтр {i+1}: пик на {peak_wl:.0f} нм, "
              f"мин={comp.min():.4f}, макс={comp.max():.4f}")

    save_filters(wl_sub, components, n, lo, hi, tag)
    plot_filters(wl_sub, components, n, tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True,
                        help="Число фильтров")
    parser.add_argument("--range", dest="range_name", default=None,
                        choices=list(RANGE_PRESETS.keys()),
                        help="Спектральный диапазон (vis/nir/swir/full)")
    parser.add_argument("--lo", type=float, default=None,
                        help="Нижняя граница диапазона, нм (для произвольного диапазона)")
    parser.add_argument("--hi", type=float, default=None,
                        help="Верхняя граница диапазона, нм")
    args = parser.parse_args()

    if args.range_name:
        lo, hi = RANGE_PRESETS[args.range_name]
        tag = args.range_name
    elif args.lo is not None and args.hi is not None:
        lo, hi = args.lo, args.hi
        tag = f"{int(lo)}-{int(hi)}nm"
    else:
        parser.error("Укажите --range или оба флага --lo и --hi")

    run(n=args.n, lo=lo, hi=hi, tag=tag)
