"""
Class separability analysis for Indian Pines.

Summary produced:
  1. Per-class accuracy, precision, recall, F1 (RF on full spectrum)
  2. Confusion matrix heatmap
  3. Pairwise mean-spectrum distance matrix
  4. Ranked list: easiest and hardest class pairs to distinguish
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import normalize
from utils import load_mat, split_data, zscore, CLASS_NAMES

PLOTS = Path(__file__).parent.parent / "data" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_RU = {
    1: "Люцерна", 2: "Кукуруза (б/в)", 3: "Кукуруза (мин)",
    4: "Кукуруза", 5: "Трава/Пастбище", 6: "Трава/Деревья",
    7: "Паст. скошенное", 8: "Сено в валках", 9: "Овёс",
    10: "Соя (б/в)", 11: "Соя (мин)", 12: "Соя (чист.)",
    13: "Пшеница", 14: "Лес", 15: "Постройки", 16: "Башни",
}


# ---------------------------------------------------------------------------
# Train RF on full spectrum
# ---------------------------------------------------------------------------

def train_rf(Xtr, ytr):
    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    return clf


# ---------------------------------------------------------------------------
# 1. Per-class metrics
# ---------------------------------------------------------------------------

def print_per_class_metrics(y_true, y_pred, classes) -> None:
    print("\n" + "=" * 72)
    print("  Метрики по классам (тестовая выборка)")
    print("=" * 72)
    print(f"  {'Кл':>3}  {'Название':<28}  {'N':>5}  {'Точн':>6}  {'Полн':>6}  {'F1':>6}")
    print("-" * 72)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for cls in classes:
        key = str(cls)
        name = CLASS_NAMES_RU.get(cls, f"Класс {cls}")
        n = (y_true == cls).sum()
        p  = report[key]["precision"] * 100
        r  = report[key]["recall"]    * 100
        f1 = report[key]["f1-score"]  * 100
        flag = " ◄ сложный" if f1 < 50 else ""
        print(f"  {cls:>3}  {name:<28}  {n:>5}  {p:>5.1f}%  {r:>5.1f}%  {f1:>5.1f}%{flag}")
    print("-" * 72)
    oa = accuracy_score(y_true, y_pred) * 100
    print(f"  Общая точность (OA): {oa:.2f}%")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = [CLASS_NAMES_RU.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Доля предсказаний")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Предсказанный класс", fontsize=10)
    ax.set_ylabel("Истинный класс", fontsize=10)
    ax.set_title("Матрица ошибок (нормированная по строкам)", fontsize=12)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.5 else "black")

    fig.tight_layout()
    path = PLOTS / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Сохранено → {path}")


# ---------------------------------------------------------------------------
# 3. Pairwise mean-spectrum distance
# ---------------------------------------------------------------------------

def pairwise_distance_matrix(X, y, classes) -> np.ndarray:
    """Cosine distance between class mean spectra (0=identical, 1=orthogonal)."""
    means = np.array([X[y == c].mean(axis=0) for c in classes])
    means_norm = normalize(means, norm="l2")
    cosine_sim = means_norm @ means_norm.T
    return 1.0 - np.clip(cosine_sim, -1, 1)


def plot_distance_matrix(dist: np.ndarray, classes) -> None:
    labels = [CLASS_NAMES_RU.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(dist, cmap="RdYlGn", vmin=0, vmax=dist.max())
    plt.colorbar(im, ax=ax, fraction=0.046, label="Косинусное расстояние")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Попарное расстояние между средними спектрами классов\n"
                 "(0 = идентичные, больше = легче различить)", fontsize=11)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color="black")

    fig.tight_layout()
    path = PLOTS / "class_distance_matrix.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Сохранено → {path}")


# ---------------------------------------------------------------------------
# 4. Ranked pairs
# ---------------------------------------------------------------------------

def print_ranked_pairs(dist: np.ndarray, classes, top_n: int = 10) -> None:
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((dist[i, j], classes[i], classes[j]))
    pairs.sort(key=lambda x: x[0])

    print(f"\n{'─'*60}")
    print(f"  {top_n} наиболее ПОХОЖИХ пар (трудно различить):")
    print(f"{'─'*60}")
    for d, a, b in pairs[:top_n]:
        print(f"  {d:.4f}  {CLASS_NAMES_RU[a]:<28} ↔  {CLASS_NAMES_RU[b]}")

    print(f"\n{'─'*60}")
    print(f"  {top_n} наиболее РАЗЛИЧНЫХ пар (легко различить):")
    print(f"{'─'*60}")
    for d, a, b in reversed(pairs[-top_n:]):
        print(f"  {d:.4f}  {CLASS_NAMES_RU[a]:<28} ↔  {CLASS_NAMES_RU[b]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("Загрузка данных...")
    X, y, wl = load_mat()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    Xz_tr, _, Xz_te, _, _ = zscore(X_train, X_val, X_test)

    classes = sorted(np.unique(y).tolist())

    print("Обучение Random Forest (200 полос)...")
    clf = train_rf(Xz_tr, y_train)
    y_pred = clf.predict(Xz_te)

    print_per_class_metrics(y_test, y_pred, classes)

    print("\nМатрица ошибок...")
    plot_confusion_matrix(y_test, y_pred, classes)

    print("\nПопарные расстояния между классами...")
    dist = pairwise_distance_matrix(X, y, classes)
    plot_distance_matrix(dist, classes)
    print_ranked_pairs(dist, classes)


if __name__ == "__main__":
    run()
