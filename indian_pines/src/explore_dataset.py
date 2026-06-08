"""
Quick exploration of the Indian Pines dataset.

Prints dataset summary and plots:
  1. Class distribution bar chart
  2. Random sample spectra from each class
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from utils import load_mat, CLASS_NAMES


def print_summary(X: np.ndarray, y: np.ndarray, wl: np.ndarray) -> None:
    print("=" * 55)
    print("  Indian Pines — Dataset Summary")
    print("=" * 55)
    print(f"  Total labeled pixels : {len(y)}")
    print(f"  Spectral bands       : {X.shape[1]}")
    print(f"  Wavelength range     : {wl[0]:.0f} – {wl[-1]:.0f} nm")
    print(f"  Band spacing (avg)   : {(wl[-1]-wl[0])/(len(wl)-1):.1f} nm")
    print(f"  Classes              : {len(np.unique(y))}")
    print("-" * 55)
    print(f"  {'#':<4} {'Class':<35} {'Count':>6}")
    print("-" * 55)
    for cls in sorted(np.unique(y)):
        name = CLASS_NAMES.get(int(cls), f"Class {cls}")
        print(f"  {cls:<4} {name:<35} {(y == cls).sum():>6}")
    print("=" * 55)


def plot_class_distribution(y: np.ndarray) -> None:
    classes = sorted(np.unique(y))
    counts = [(y == c).sum() for c in classes]
    labels = [CLASS_NAMES.get(int(c), str(c)) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(classes)), counts, color=plt.cm.tab20.colors[:len(classes)])
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Number of pixels")
    ax.set_title("Indian Pines — class distribution")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    plt.show()


def plot_sample_spectra(X: np.ndarray, y: np.ndarray, wl: np.ndarray, n_samples: int = 3) -> None:
    classes = sorted(np.unique(y))
    cols = 4
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.5), squeeze=False)

    rng = np.random.default_rng(0)
    cmap = plt.cm.tab10

    for i, cls in enumerate(classes):
        ax = axes[i // cols][i % cols]
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
        for j, sample_idx in enumerate(chosen):
            ax.plot(wl, X[sample_idx], lw=0.9, alpha=0.85, color=cmap(j))
        name = CLASS_NAMES.get(int(cls), f"Class {cls}")
        ax.set_title(f"{cls}: {name}\n(n={len(idx)})", fontsize=8)
        ax.set_xlabel("λ (nm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"Indian Pines — {n_samples} sample spectra per class", fontsize=11)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading .mat files...")
    X, y, wl = load_mat()
    print_summary(X, y, wl)
    plot_class_distribution(y)
    plot_sample_spectra(X, y, wl, n_samples=3)
