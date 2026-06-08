"""
Plots sample spectra for the 3 most frequent classes in Indian Pines
and a spatial class-distribution map. All captions are in Russian.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from utils import load_mat, RAW_DIR

CLASS_NAMES_RU = {
    1:  "Люцерна",
    2:  "Кукуруза (без вспашки)",
    3:  "Кукуруза (мин. обработка)",
    4:  "Кукуруза",
    5:  "Трава / Пастбище",
    6:  "Трава / Деревья",
    7:  "Скошенное пастбище",
    8:  "Сено в валках",
    9:  "Овёс",
    10: "Соя (без вспашки)",
    11: "Соя (мин. обработка)",
    12: "Соя (чистая)",
    13: "Пшеница",
    14: "Лес",
    15: "Постройки / Трава / Деревья",
    16: "Металлические башни",
}

N_SAMPLES = 5


def top3_classes(y: np.ndarray) -> list[int]:
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [int(classes[i]) for i in order[:3]]


def plot_class(cls: int, X: np.ndarray, y: np.ndarray, wl: np.ndarray) -> None:
    idx = np.where(y == cls)[0]
    rng = np.random.default_rng(42)
    chosen = rng.choice(idx, size=min(N_SAMPLES, len(idx)), replace=False)

    name = CLASS_NAMES_RU.get(cls, f"Класс {cls}")
    count = len(idx)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.tab10
    for j, i in enumerate(chosen):
        ax.plot(wl, X[i], lw=1.2, alpha=0.85, color=cmap(j),
                label=f"Образец {j + 1}")

    ax.set_xlabel("Длина волны, нм", fontsize=11)
    ax.set_ylabel("Отражение", fontsize=11)
    ax.set_title(f"Класс {cls}: {name}  (всего пикселей: {count})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_spatial_map(raw_dir: Path = RAW_DIR) -> None:
    """Reconstruct and display the 145×145 class-distribution map."""
    gt_mat = scipy.io.loadmat(str(raw_dir / "Indian_pines_gt.mat"))
    gt_key = next(k for k in gt_mat if not k.startswith("_"))
    gt = gt_mat[gt_key].astype(np.int16)  # (145, 145), 0 = unlabeled

    # Build a discrete colormap: index 0 = unlabeled (white), 1-16 = tab20 colors
    colors = [(1.0, 1.0, 1.0)] + [plt.cm.tab20(i / 20) for i in range(16)]
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    ax.set_title("Карта распределения классов — Indian Pines", fontsize=12)
    ax.set_xlabel("Столбец", fontsize=10)
    ax.set_ylabel("Строка", fontsize=10)

    # Legend patches
    patches = [mpatches.Patch(color=colors[0], label="Без метки", ec="grey", lw=0.5)]
    for cls in range(1, 17):
        name = CLASS_NAMES_RU.get(cls, f"Класс {cls}")
        count = (gt == cls).sum()
        patches.append(mpatches.Patch(color=colors[cls],
                                      label=f"{cls}. {name} ({count})"))
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Загрузка данных...")
    X, y, wl = load_mat()

    top3 = top3_classes(y)
    print("Три наиболее частых класса:")
    for cls in top3:
        print(f"  Класс {cls:2d}: {CLASS_NAMES_RU[cls]}  —  {(y == cls).sum()} пикселей")

    print("\nКарта распределения объектов...")
    plot_spatial_map()

    for cls in top3:
        plot_class(cls, X, y, wl)
