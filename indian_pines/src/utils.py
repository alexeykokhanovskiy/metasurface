"""Shared loading utilities for Indian Pines .mat files."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

CLASS_NAMES = {
    1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
    5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
    8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
    11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
    14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers",
}


def _aviris_wavelengths() -> np.ndarray:
    """Centre wavelengths (nm) for the 200 corrected AVIRIS bands."""
    all_wl = np.linspace(399.6, 2499.6, 220)
    removed = list(range(103, 108)) + list(range(149, 163)) + [219]
    mask = np.ones(220, dtype=bool)
    mask[removed] = False
    return all_wl[mask].astype(np.float32)


def load_mat(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Indian Pines from .mat files.

    Returns
    -------
    X           : (n_labeled, 200) float32 — raw reflectance
    y           : (n_labeled,)     int16   — class labels 1-16
    wavelengths : (200,)           float32 — wavelength in nm
    """
    cube_path = raw_dir / "Indian_pines_corrected.mat"
    gt_path   = raw_dir / "Indian_pines_gt.mat"

    cube_mat = scipy.io.loadmat(str(cube_path))
    gt_mat   = scipy.io.loadmat(str(gt_path))

    cube_key = next(k for k in cube_mat if not k.startswith("_"))
    gt_key   = next(k for k in gt_mat   if not k.startswith("_"))

    cube = cube_mat[cube_key].astype(np.float32)  # (145,145,200)
    gt   = gt_mat[gt_key].astype(np.int16)        # (145,145)

    mask = gt > 0
    X = cube[mask]
    y = gt[mask]
    return X, y, _aviris_wavelengths()


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Stratified 60/20/20 split. Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def zscore(X_train, X_val, X_test):
    """Z-score per band, fit on train only. Returns normalised arrays + (mean, std)."""
    mean = X_train.mean(axis=0)
    std  = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std
