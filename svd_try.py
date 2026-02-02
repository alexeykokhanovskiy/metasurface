import numpy as np
import h5py
from cmaes import CMA
import matplotlib.pyplot as plt

from data_processing import (
    load_experimental_curve,
    calc_mae_model_experimental,
    calc_rmse_model_experimental
)
from metasurface import Metasurface, Geometry, Material, Spectrum
from svd_analysis import SVDSpectralAnalyzer




WL_MIN = 1.0
WL_MAX = 4.0
N_wl_points = 100
wl_model = np.linspace(WL_MIN, WL_MAX, N_wl_points)
EXP_FILE = './LIPPS_sio2_polarized.csv'
EXP_COL = 6



wl_exp, signal_exp = load_experimental_curve(
    wl_min=WL_MIN, wl_max=WL_MAX,
    experimental_file_path=EXP_FILE,
    col_index=EXP_COL
)

# Convert signal to fraction (from percentage)
signal_exp = signal_exp / 100.0

# Create multiple noisy versions for SVD analysis
n_replicas = 20
noise_level = 0.01  # 1% noise
spectra_matrix = []

for i in range(n_replicas):
    noisy_signal = signal_exp + np.random.randn(len(signal_exp)) * noise_level
    spectra_matrix.append(noisy_signal)

spectra_matrix = np.array(spectra_matrix)

# Apply SVD analysis
print("Applying SVD to experimental data...")
analyzer = SVDSpectralAnalyzer(spectra_matrix, wl_exp)
analyzer.fit(center=True)

# Find components for 95% variance
n_components = analyzer.find_n_components_for_variance(0.95)
print(f"Components needed for 95% variance: {n_components}")
print(f"Actual variance: {analyzer.get_explained_variance(n_components)*100:.2f}%")

# Reconstruct with optimal components
reconstructed = analyzer.reconstruct(n_components)
denoised_signal = np.mean(reconstructed, axis=0)

# Plot original vs denoised - single figure
plt.figure(figsize=(12, 6))
plt.plot(wl_exp, signal_exp, 'o-', label='Original Experimental', linewidth=2.5, markersize=6, color='black')
plt.plot(wl_exp, denoised_signal, 's-', label=f'SVD Reconstructed ({n_components} comp., 95% variance)',
         linewidth=2, markersize=5, color='red')
plt.xlabel('Wavelength (μm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title(f'Experimental Spectrum vs SVD Reconstruction ({n_components} components)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
