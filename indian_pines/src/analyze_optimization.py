"""
Analyse CMA-ES optimisation results.

Loads the saved best_filters_*.npz (no RCWA calls needed) and the log CSV,
then produces:
  1. Convergence curve   — fitness vs generation
  2. Filter spectra      — T(λ) for each optimised pillar
  3. Classification report + confusion matrices (val + test)

Works for both fitness modes: 'accuracy' and 'nmf_mae'.

Usage
-----
    python analyze_optimization.py                              # auto-detect latest
    python analyze_optimization.py --log cmaes_N3_vis_nmf_mae.csv
    python analyze_optimization.py --npz best_filters_N3_vis_nmf_mae.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from utils import load_mat, split_data

ROOT      = Path(__file__).parent.parent
OPT_DIR   = ROOT / "data" / "optimization"
PLOTS_DIR = ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP   = {10: 1, 11: 1, 12: 1, 14: 2, 16: 3}
LABEL_NAMES = {1: "Соя", 2: "Лес", 3: "Башни"}

RANGE_MAP = {
    "vis":  (400,  1000),
    "nir":  (1000, 1500),
    "swir": (1500, 2500),
    "full": (400,  2500),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_sensor_data(wl_nm: np.ndarray):
    X, y, wl_full = load_mat()
    keep = np.isin(y, list(LABEL_MAP.keys()))
    X = X[keep]
    y = np.array([LABEL_MAP[int(c)] for c in y[keep]])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    def interp(src):
        f = interp1d(wl_full, src, axis=1, bounds_error=False, fill_value=0.0)
        return np.maximum(f(wl_nm), 0.0).astype(np.float32)

    return interp(X_train), interp(X_val), interp(X_test), y_train, y_val, y_test


def classify(F, X_train, X_val, X_test, y_train, y_val, y_test):
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 n_jobs=-1, random_state=42)
    clf.fit(X_train @ F.T, y_train)
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    return {
        "val_acc":  accuracy_score(y_val,  clf.predict(X_val  @ F.T)),
        "test_acc": accuracy_score(y_test, clf.predict(X_test @ F.T)),
        "report":   classification_report(y_test, clf.predict(X_test @ F.T),
                                          target_names=target_names),
        "cm_val":   confusion_matrix(y_val,  clf.predict(X_val  @ F.T), labels=[1,2,3]),
        "cm_test":  confusion_matrix(y_test, clf.predict(X_test @ F.T), labels=[1,2,3]),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_convergence(gens, fitnesses, fitness_mode, tag, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    if fitness_mode == "accuracy":
        values = -fitnesses * 100
        ylabel = "Точность на валидации (%)"
        best_idx = int(np.argmax(values))
        best_label = f"Поколение {gens[best_idx]}  ({values[best_idx]:.2f}%)"
    else:
        values = fitnesses
        ylabel = "MAE (целевые спектры NMF)"
        best_idx = int(np.argmin(values))
        best_label = f"Поколение {gens[best_idx]}  (MAE={values[best_idx]:.4f})"

    ax.plot(gens, values, "o-", color="steelblue", linewidth=1.5,
            markersize=3, alpha=0.8)
    ax.axvline(gens[best_idx], color="crimson", linestyle="--",
               label=f"Лучшее: {best_label}")
    ax.set_xlabel("Поколение")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Сходимость CMA-ES — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Convergence  -> {save_path.name}")


def plot_filters(wl_nm, F, diameters, heights, tag, save_path):
    n = F.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n):
        label = f"Filter {i+1}:  d={diameters[i]:.0f} nm,  h={heights[i]:.0f} nm"
        ax.plot(wl_nm, F[i], color=colors[i], linewidth=2, label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance T₀")
    ax.set_title(f"Optimised filter spectra — {tag}")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Filter spectra -> {save_path.name}")


def plot_confusion(cm, title, save_path):
    labels = [LABEL_NAMES[i] for i in [1, 2, 3]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {save_path.name}")


# ---------------------------------------------------------------------------
# NMF vs RCWA comparison grid
# ---------------------------------------------------------------------------

def _load_nmf_for_range(wl_nm: np.ndarray) -> np.ndarray | None:
    """
    Try to find an NMF file and interpolate its components to wl_nm.
    Looks for range-tagged file first, then falls back to full-range file.
    Returns (N, B) normalised array or None if no file found.
    """
    proc_dir = ROOT / "data" / "processed"
    n = None
    candidates = sorted(proc_dir.glob("nmf_filters_n*.npz"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, None
    path = candidates[0]
    d = np.load(str(path))
    wl_full  = d["wavelengths"].astype(float)
    comps    = d["components"].astype(float)
    f        = interp1d(wl_full, comps, axis=1, bounds_error=False, fill_value=0.0)
    F        = np.maximum(f(wl_nm), 0.0)
    F       /= F.max(axis=1, keepdims=True) + 1e-12
    return F.astype(np.float32), path.name


def plot_nmf_vs_rcwa(wl_nm, F_rcwa, diameters, heights, tag, save_path):
    """
    3-column x 2-row grid.
    Row 0: NMF target components (matched to RCWA filters via Hungarian algorithm)
    Row 1: RCWA / meent filter spectra
    """
    n = F_rcwa.shape[0]
    F_nmf, nmf_fname = _load_nmf_for_range(wl_nm)

    if F_nmf is None:
        print("  (No NMF file found, skipping comparison plot)")
        return

    # Hungarian matching: assign NMF component j to RCWA filter i
    F_rcwa_norm = F_rcwa / (F_rcwa.max(axis=1, keepdims=True) + 1e-12)
    cost = np.array([[np.mean(np.abs(F_rcwa_norm[i] - F_nmf[j]))
                      for j in range(n)] for i in range(n)])
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = NMF component matched to RCWA filter i
    F_nmf_ordered = F_nmf[col_ind]

    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 6), sharey=True, sharex=True)

    for i in range(n):
        ax_nmf  = axes[0, i]
        ax_rcwa = axes[1, i]
        c = colors[i]

        # NMF row
        ax_nmf.plot(wl_nm, F_nmf_ordered[i], color=c, linewidth=2)
        ax_nmf.fill_between(wl_nm, F_nmf_ordered[i], alpha=0.15, color=c)
        ax_nmf.set_title(f"Фильтр {i+1} — цель NMF", fontsize=9)
        ax_nmf.set_ylim(-0.02, 1.15)
        ax_nmf.grid(True, alpha=0.35)

        # RCWA row
        ax_rcwa.plot(wl_nm, F_rcwa_norm[i], color=c, linewidth=2)
        ax_rcwa.fill_between(wl_nm, F_rcwa_norm[i], alpha=0.15, color=c)
        mae = float(np.mean(np.abs(F_rcwa_norm[i] - F_nmf_ordered[i])))
        ax_rcwa.set_title(
            f"d={diameters[i]:.0f} нм,  h={heights[i]:.0f} нм\nMAE={mae:.3f}",
            fontsize=9)
        ax_rcwa.set_xlabel("Длина волны (нм)", fontsize=9)
        ax_rcwa.grid(True, alpha=0.35)

    axes[0, 0].set_ylabel("Компонента NMF (норм.)", fontsize=9)
    axes[1, 0].set_ylabel("Пропускание RCWA (норм.)", fontsize=9)

    fig.suptitle(f"Целевые спектры NMF и оптимизированные фильтры RCWA — {tag}\n"
                 f"(NMF: {nmf_fname}, венгерское сопоставление)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  NMF vs RCWA  -> {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(log_name: str | None, npz_name: str | None) -> None:

    # --- Locate files ---
    if npz_name:
        npz_path = OPT_DIR / npz_name
    else:
        npzs = sorted(OPT_DIR.glob("best_filters_*.npz"), key=lambda p: p.stat().st_mtime)
        if not npzs:
            raise FileNotFoundError(f"No best_filters_*.npz in {OPT_DIR}")
        npz_path = npzs[-1]

    # Derive log path from npz name (best_filters_N3_vis_nmf_mae -> cmaes_N3_vis_nmf_mae)
    derived_log = npz_path.stem.replace("best_filters_", "cmaes_") + ".csv"
    if log_name:
        log_path = OPT_DIR / log_name
    else:
        log_path = OPT_DIR / derived_log

    print(f"\nNPZ : {npz_path.name}")
    print(f"Log : {log_path.name}")

    # Detect fitness mode from filename
    fitness_mode = "nmf_mae" if "nmf_mae" in npz_path.stem else "accuracy"
    tag = npz_path.stem.replace("best_filters_", "")
    print(f"Fitness mode: {fitness_mode}")

    # --- Load best filters ---
    d = np.load(str(npz_path))
    diameters = d["diameters_nm"]
    heights   = d["heights_nm"]
    wl_nm     = d["wavelengths"]
    F         = d["filters"].astype(np.float32)      # (N, B)
    best_fit  = float(d["best_fitness"])
    n_filters = len(diameters)

    print(f"\n{'='*54}")
    print(f"  Optimised geometries ({n_filters} filters):")
    print(f"  {'Filter':>8}  {'Diameter (nm)':>14}  {'Height (nm)':>12}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*12}")
    for i in range(n_filters):
        print(f"  {i+1:>8}  {diameters[i]:>14.1f}  {heights[i]:>12.1f}")
    if fitness_mode == "accuracy":
        print(f"\n  Best CMA-ES val accuracy: {-best_fit*100:.2f}%")
    else:
        print(f"\n  Best CMA-ES MAE: {best_fit:.4f}")
    print(f"{'='*54}")

    # --- Load log and plot convergence ---
    if log_path.exists():
        data = np.loadtxt(str(log_path), delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        gens      = data[:, 0].astype(int)
        fitnesses = data[:, -1]
        print(f"\n  Generations logged: {len(gens)}")
        plot_convergence(gens, fitnesses, fitness_mode, tag,
                         PLOTS_DIR / f"{tag}_convergence.png")
    else:
        print(f"  (Log not found — skipping convergence plot)")

    # --- Filter spectra ---
    plot_filters(wl_nm, F, diameters, heights, tag,
                 PLOTS_DIR / f"{tag}_filters.png")
    plot_nmf_vs_rcwa(wl_nm, F, diameters, heights, tag,
                     PLOTS_DIR / f"{tag}_nmf_vs_rcwa.png")

    # --- Classification ---
    print("\nRunning classification...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_sensor_data(wl_nm)
    metrics = classify(F, X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"\n{'='*54}")
    print(f"  Val  accuracy : {metrics['val_acc']*100:.2f}%")
    print(f"  Test accuracy : {metrics['test_acc']*100:.2f}%")
    print(f"\n  Classification report (test set):")
    print(metrics["report"])
    print("="*54)

    plot_confusion(metrics["cm_val"],
                   f"Матрица ошибок — валидация\n{tag}",
                   PLOTS_DIR / f"{tag}_cm_val.png")
    plot_confusion(metrics["cm_test"],
                   f"Матрица ошибок — тест\n{tag}",
                   PLOTS_DIR / f"{tag}_cm_test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None,
                        help="CSV log filename in data/optimization/")
    parser.add_argument("--npz", default=None,
                        help="NPZ results filename in data/optimization/")
    args = parser.parse_args()
    run(args.log, args.npz)
