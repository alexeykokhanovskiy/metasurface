"""
SVD-based concentration regression — Fe (FeCl3), optimised.

Key findings from sweep:
  - Raw spectra (no normalisation) — intensity encodes concentration
  - n_components = 3 SVD components optimal by LOOCV
  - LOOCV is used (18 samples, most stable estimate)
  - Best models: Ridge (R2~0.94), CatBoost (R2~0.93), GPR (R2~0.94)
"""

import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from svd_analysis import SVDSpectralAnalyzer

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
RANDOM_STATE = 42
WL_MIN, WL_MAX = 380.0, 880.0
N_WAVELENGTHS = 1000
N_SVD_COMPONENTS = 3       # optimal from LOOCV sweep (minimises RMSE < 30 mM)
LOW_CONC_THRESHOLD = 0.030  # 30 mM
TARGET_WL = np.linspace(WL_MIN, WL_MAX, N_WAVELENGTHS)
COLOR_FE = '#e74c3c'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_concentration(s: str) -> float:
    s = s.strip()
    if s.endswith('uM'): return float(s[:-2]) * 1e-6
    if s.endswith('mM'): return float(s[:-2]) * 1e-3
    if s.endswith('M'):  return float(s[:-1])
    raise ValueError(f"Cannot parse: {s!r}")


def load_fe(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df_f = df[
        (df['Time'] == '0 hrs') &
        (df['Concentration'] != 'Unknown') &
        (df['Compound'] == 'FeCl3')
    ].copy()
    df_f['conc_molar'] = df_f['Concentration'].apply(parse_concentration)

    X_list, y_molar_list, conc_str_list = [], [], []
    for (_, conc_str), grp in df_f.groupby(['Source File', 'Concentration']):
        grp_c = grp[
            (grp['Wavelength (nm)'] >= WL_MIN) &
            (grp['Wavelength (nm)'] <= WL_MAX)
        ].sort_values('Wavelength (nm)')
        if len(grp_c) < 10:
            continue
        X_list.append(np.interp(TARGET_WL,
                                grp_c['Wavelength (nm)'].values,
                                grp_c['Intensity (a.u.)'].values))
        y_molar_list.append(grp_c['conc_molar'].iloc[0])
        conc_str_list.append(conc_str)

    X = np.array(X_list, dtype=float)
    y_molar = np.array(y_molar_list, dtype=float)
    y_log = np.log10(y_molar)

    print(f"Fe: {X.shape[0]} spectra, log10 {y_log.min():.2f} to {y_log.max():.2f}")
    return {'X': X, 'y_log': y_log, 'y_molar': y_molar, 'conc_str': conc_str_list}


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

def fit_svd(X: np.ndarray, n_components: int = N_SVD_COMPONENTS) -> tuple:
    analyzer = SVDSpectralAnalyzer(X, TARGET_WL)
    analyzer.fit(center=True)
    features = analyzer.U[:, :n_components] * analyzer.S[:n_components]
    var = np.cumsum(analyzer.explained_variance_ratio)[n_components - 1]
    print(f"  SVD: {n_components} components -> {var * 100:.2f}% variance")
    return features, analyzer


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models() -> dict:
    return {
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 30)),
        'SVR':   SVR(kernel='rbf', C=10.0, epsilon=0.05),
        'GPR':   GaussianProcessRegressor(
                     kernel=Matern(nu=1.5) + WhiteKernel(),
                     normalize_y=True, n_restarts_optimizer=5,
                     random_state=RANDOM_STATE),
        'CatBoost': CatBoostRegressor(
                     iterations=1000, learning_rate=0.05, depth=6,
                     loss_function='RMSE', random_seed=RANDOM_STATE,
                     verbose=False),
    }


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------

def run_loocv(X_feat: np.ndarray, y_log: np.ndarray, y_molar: np.ndarray) -> dict:
    loo = LeaveOneOut()
    mask_low = y_molar < LOW_CONC_THRESHOLD
    results = {}

    for name, tmpl in make_models().items():
        y_pred_log = np.zeros_like(y_log)

        for tr, val in loo.split(X_feat):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[tr])
            X_val = scaler.transform(X_feat[val])

            model = copy.deepcopy(tmpl)
            if name == 'CatBoost':
                model.fit(X_tr, y_log[tr], verbose=False)
            else:
                model.fit(X_tr, y_log[tr])
            y_pred_log[val] = model.predict(X_val)

        y_pred_molar = 10 ** y_pred_log
        r2        = r2_score(y_log, y_pred_log)
        rmse_all  = np.sqrt(mean_squared_error(y_molar, y_pred_molar))
        rmse_low  = np.sqrt(mean_squared_error(y_molar[mask_low], y_pred_molar[mask_low]))
        rmse_high = np.sqrt(mean_squared_error(y_molar[~mask_low], y_pred_molar[~mask_low]))
        mae       = mean_absolute_error(y_molar, y_pred_molar)
        mape      = np.mean(np.abs((y_pred_molar - y_molar) / y_molar)) * 100

        results[name] = {
            'r2': r2, 'rmse_all': rmse_all, 'rmse_low': rmse_low,
            'rmse_high': rmse_high, 'mae': mae, 'mape': mape,
            'y_pred_log': y_pred_log, 'y_pred_molar': y_pred_molar,
        }
        print(f"  {name:10s}: R2={r2:.4f}  "
              f"RMSE_all={rmse_all*1000:.2f} mM  "
              f"RMSE_<30mM={rmse_low*1000:.2f} mM  "
              f"RMSE_>=30mM={rmse_high*1000:.2f} mM")

    return results


# ---------------------------------------------------------------------------
# Component sweep plot
# ---------------------------------------------------------------------------

def plot_component_sweep(X: np.ndarray, y_log: np.ndarray, y_molar: np.ndarray,
                         out_dir: Path):
    """LOOCV R2 vs n_components for each model."""
    model_names = list(make_models().keys())
    n_range = list(range(1, min(18, X.shape[0])))
    sweep = {name: [] for name in model_names}
    loo = LeaveOneOut()

    for n in n_range:
        analyzer = SVDSpectralAnalyzer(X, TARGET_WL)
        analyzer.fit(center=True)
        feats = analyzer.U[:, :n] * analyzer.S[:n]

        for name, tmpl in make_models().items():
            y_pred = np.zeros_like(y_log)
            for tr, val in loo.split(feats):
                sc = StandardScaler()
                Xtr = sc.fit_transform(feats[tr])
                Xval = sc.transform(feats[val])
                mdl = copy.deepcopy(tmpl)
                if name == 'CatBoost':
                    mdl.fit(Xtr, y_log[tr], verbose=False)
                else:
                    mdl.fit(Xtr, y_log[tr])
                y_pred[val] = mdl.predict(Xval)
            sweep[name].append(r2_score(y_log, y_pred))

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, r2s), color in zip(sweep.items(), colors):
        ax.plot(n_range, r2s, 'o-', label=name, linewidth=2, markersize=6, color=color)
    ax.axvline(N_SVD_COMPONENTS, color='black', linestyle='--', linewidth=1.5,
               label=f'Selected n={N_SVD_COMPONENTS}')
    ax.set_xlabel('Number of SVD components')
    ax.set_ylabel('LOOCV R2 (log10 target)')
    ax.set_title('Fe: LOOCV R2 vs SVD components', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim([-0.5, 1.05])
    plt.tight_layout()
    p = out_dir / 'fe_component_sweep.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Prediction plots
# ---------------------------------------------------------------------------

def _conc_label(m: float) -> str:
    uM = m * 1e6
    return f"{uM:.0f}uM" if uM < 1000 else f"{uM/1000:.0f}mM"


def plot_predictions(loocv_results: dict, y_molar: np.ndarray,
                     conc_str: list, out_dir: Path):
    from matplotlib.ticker import FuncFormatter

    model_names = list(make_models().keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    def log_fmt(x, _):
        m = 10 ** x
        uM = m * 1e6
        return f"{uM:.0f}uM" if uM < 1000 else f"{uM/1000:.0f}mM"

    y_true_log = np.log10(y_molar)
    order = np.argsort(y_true_log)

    for col, name in enumerate(model_names):
        res = loocv_results[name]
        y_pred_log = res['y_pred_log']
        y_pred_m   = res['y_pred_molar']

        # --- scatter ---
        ax = axes[0, col]
        ax.scatter(y_true_log, y_pred_log,
                   color=COLOR_FE, s=70, edgecolors='black', linewidths=0.6,
                   alpha=0.9, zorder=3)
        lo = min(y_true_log.min(), y_pred_log.min()) - 0.2
        hi = max(y_true_log.max(), y_pred_log.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5)
        ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])

        ticks = np.arange(np.ceil(lo * 2) / 2, hi + 0.1, 0.5)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(log_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(log_fmt))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('True concentration')
        ax.set_ylabel('Predicted concentration' if col == 0 else '')
        ax.set_title(f'{name}\nR2={res["r2"]:.4f}  MAPE={res["mape"]:.1f}%',
                     fontweight='bold')
        ax.grid(alpha=0.25)

        # --- calibration strip ---
        ax2 = axes[1, col]
        xs = np.arange(len(order))
        ax2.plot(xs, y_true_log[order], 'o-', color='steelblue',
                 linewidth=1.5, markersize=6, label='True', zorder=3)
        ax2.scatter(xs, y_pred_log[order], color=COLOR_FE, marker='D',
                    s=55, edgecolors='black', linewidths=0.5,
                    label='Predicted', zorder=4)
        for xi, (yt, yp) in enumerate(zip(y_true_log[order], y_pred_log[order])):
            ax2.plot([xi, xi], [yt, yp], color='gray', linewidth=0.8, alpha=0.6)

        ax2.set_xticks(xs)
        ax2.set_xticklabels([_conc_label(y_molar[i]) for i in order],
                             rotation=60, ha='right', fontsize=7)
        ax2.yaxis.set_major_formatter(FuncFormatter(log_fmt))
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_xlabel('Sample (sorted by true conc.)')
        ax2.set_ylabel('Concentration' if col == 0 else '')
        ax2.set_title(f'{name}: True vs Predicted per sample', fontweight='bold')
        # 30 mM threshold
        thr_log = np.log10(LOW_CONC_THRESHOLD)
        ax2.axhline(thr_log, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax2.text(len(xs) - 0.5, thr_log + 0.06, '30 mM',
                 fontsize=7, ha='right', color='black', alpha=0.7)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    plt.suptitle(f'Fe (FeCl3) — LOOCV predictions, SVD n={N_SVD_COMPONENTS}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'fe_predicted_vs_actual.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {p.name}")


def plot_catboost_fit_curve(loocv_results: dict, y_molar: np.ndarray, out_dir: Path):
    from matplotlib.ticker import FuncFormatter

    res = loocv_results['CatBoost']
    y_true_log  = np.log10(y_molar)
    y_pred_log  = res['y_pred_log']

    def log_fmt(x, _):
        m = 10 ** x
        uM = m * 1e6
        return f"{uM:.0f} µM" if uM < 1000 else f"{uM/1000:.0f} mM"

    fig, ax = plt.subplots(figsize=(6, 6))

    # scatter
    ax.scatter(y_true_log, y_pred_log,
               color='#78323c', s=100, edgecolors='black', linewidths=0.7,
               zorder=4, label='LOOCV samples')

    # perfect prediction line
    lo = min(y_true_log.min(), y_pred_log.min()) - 0.15
    hi = max(y_true_log.max(), y_pred_log.max()) + 0.15
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ticks = np.arange(np.ceil(lo * 2) / 2, hi + 0.1, 0.5)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(log_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(log_fmt))
    ax.tick_params(axis='x', rotation=45, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel('True concentration', fontsize=14)
    ax.set_ylabel('Predicted concentration', fontsize=14)
    ax.set_title('CatBoost: LOOCV Predicted vs. Actual\n'
                 f'Fe (FeCl₃), SVD n={N_SVD_COMPONENTS}',
                 fontweight='bold', fontsize=14)

    r2   = res['r2']
    rmse = res['rmse_all'] * 1000
    rmse_low = res['rmse_low'] * 1000
    ax.text(0.04, 0.96,
            f'R² = {r2:.4f}\nRMSE = {rmse:.2f} mM\nRMSE (<30 mM) = {rmse_low:.2f} mM',
            transform=ax.transAxes, va='top', ha='left', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.legend(fontsize=12, loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    p = out_dir / 'fe_catboost_fit_curve.png'
    plt.savefig(p, dpi=600, bbox_inches='tight'); plt.close()
    print(f"  Saved {p.name}")


def plot_svd_analysis(analyzer: SVDSpectralAnalyzer, out_dir: Path):
    evr = analyzer.explained_variance_ratio
    cumvar = np.cumsum(evr)
    n_show = min(18, len(evr))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.bar(range(1, n_show + 1), evr[:n_show] * 100, color=COLOR_FE, alpha=0.8)
    ax.axvline(N_SVD_COMPONENTS, color='black', linestyle='--', linewidth=1.5,
               label=f'n={N_SVD_COMPONENTS}')
    ax.set_xlabel('Component'); ax.set_ylabel('Variance (%)')
    ax.set_title('Individual Variance', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, n_show + 1), cumvar[:n_show] * 100,
            'o-', color=COLOR_FE, linewidth=2, markersize=5)
    for thr, ls in [(0.90, ':'), (0.95, '--'), (0.99, '-')]:
        n_thr = analyzer.find_n_components_for_variance(thr)
        ax.axhline(thr * 100, color='gray', linestyle=ls, linewidth=1)
        if n_thr <= n_show:
            ax.plot(n_thr, thr * 100, 'r^', markersize=7)
    ax.axvline(N_SVD_COMPONENTS, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Components'); ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Cumulative Variance', fontweight='bold')
    ax.set_ylim([0, 105]); ax.grid(alpha=0.3)

    ax = axes[2]
    for k in range(N_SVD_COMPONENTS):
        ax.plot(TARGET_WL, analyzer.Vt[k, :],
                label=f'SV{k+1} ({evr[k]*100:.1f}%)', linewidth=1.5)
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Amplitude')
    ax.set_title(f'Top-{N_SVD_COMPONENTS} Singular Vectors', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle('Fe (FeCl3) — SVD Analysis (0 hrs, 380-880 nm)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'fe_svd_analysis.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {p.name}")


def print_summary(loocv_results: dict, out_dir: Path):
    rows = []
    for name, res in loocv_results.items():
        rows.append({
            'Model':           name,
            'LOOCV R2':        f"{res['r2']:.4f}",
            'RMSE all (mM)':   f"{res['rmse_all']*1000:.3f}",
            'RMSE <30mM (mM)': f"{res['rmse_low']*1000:.3f}",
            'RMSE >=30mM (mM)':f"{res['rmse_high']*1000:.3f}",
            'MAE (mM)':        f"{res['mae']*1000:.3f}",
            'MAPE (%)':        f"{res['mape']:.2f}",
        })
    df = pd.DataFrame(rows)
    p = out_dir / 'fe_regression_summary.csv'
    df.to_csv(p, index=False)
    print('\nLOOCV Summary:')
    print(df.to_string(index=False))
    print(f'\nSaved to {p}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent

    print('=== Loading Fe data ===')
    data = load_fe(out_dir / 'final_cleaned_data.csv')

    print('\n=== SVD feature extraction ===')
    X_feat, analyzer = fit_svd(data['X'])

    print('\n=== Component sweep (LOOCV) ===')
    plot_component_sweep(data['X'], data['y_log'], data['y_molar'], out_dir)

    print('\n=== LOOCV regression ===')
    loocv_results = run_loocv(X_feat, data['y_log'], data['y_molar'])

    print('\n=== Saving plots ===')
    plot_svd_analysis(analyzer, out_dir)
    plot_predictions(loocv_results, data['y_molar'], data['conc_str'], out_dir)
    plot_catboost_fit_curve(loocv_results, data['y_molar'], out_dir)

    print_summary(loocv_results, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
