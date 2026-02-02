"""
Singular Value Decomposition (SVD) analysis module for optical spectra.

This module provides tools for dimensionality reduction and noise filtering
of optical spectral data using SVD decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SVDSpectralAnalyzer:
    """
    SVD-based analyzer for optical spectra.

    Performs singular value decomposition on spectral data to identify
    principal components, reduce dimensionality, and filter noise.

    Attributes:
        spectra_matrix: Original spectra data (n_spectra x n_wavelengths)
        wavelengths: Wavelength array
        U: Left singular vectors (n_spectra x n_spectra)
        S: Singular values (min(n_spectra, n_wavelengths))
        Vt: Right singular vectors (n_wavelengths x n_wavelengths)
        explained_variance_ratio: Variance explained by each component
    """

    def __init__(self, spectra_matrix: np.ndarray, wavelengths: np.ndarray = None):
        """
        Initialize SVD analyzer with spectral data.

        Args:
            spectra_matrix: 2D array of shape (n_spectra, n_wavelengths)
                           Each row is a spectrum
            wavelengths: 1D array of wavelength values (optional)
        """
        self.spectra_matrix = np.array(spectra_matrix)
        if self.spectra_matrix.ndim != 2:
            raise ValueError("spectra_matrix must be 2D array (n_spectra x n_wavelengths)")

        self.n_spectra, self.n_wavelengths = self.spectra_matrix.shape

        if wavelengths is not None:
            self.wavelengths = np.array(wavelengths)
            if len(self.wavelengths) != self.n_wavelengths:
                raise ValueError("wavelengths length must match spectra_matrix columns")
        else:
            self.wavelengths = np.arange(self.n_wavelengths)

        # SVD components (computed on demand)
        self.U = None
        self.S = None
        self.Vt = None
        self.explained_variance_ratio = None
        self._mean_spectrum = None

    def fit(self, center: bool = True):
        """
        Perform SVD decomposition on the spectral data.

        Args:
            center: If True, center data by subtracting mean spectrum before SVD

        Returns:
            self (for method chaining)
        """
        # Center the data if requested
        if center:
            self._mean_spectrum = np.mean(self.spectra_matrix, axis=0)
            centered_data = self.spectra_matrix - self._mean_spectrum
        else:
            self._mean_spectrum = np.zeros(self.n_wavelengths)
            centered_data = self.spectra_matrix.copy()

        # Perform SVD
        self.U, self.S, self.Vt = np.linalg.svd(centered_data, full_matrices=False)

        # Calculate explained variance ratio
        total_variance = np.sum(self.S**2)
        self.explained_variance_ratio = (self.S**2) / total_variance

        return self

    def reconstruct(self, n_components: int) -> np.ndarray:
        """
        Reconstruct spectra using specified number of SVD components.

        Args:
            n_components: Number of principal components to use

        Returns:
            Reconstructed spectra matrix (n_spectra x n_wavelengths)
        """
        if self.U is None:
            raise RuntimeError("Must call fit() before reconstruct()")

        if n_components < 1 or n_components > min(self.n_spectra, self.n_wavelengths):
            raise ValueError(f"n_components must be between 1 and {min(self.n_spectra, self.n_wavelengths)}")

        # Reconstruct using truncated SVD
        U_truncated = self.U[:, :n_components]
        S_truncated = np.diag(self.S[:n_components])
        Vt_truncated = self.Vt[:n_components, :]

        reconstructed = U_truncated @ S_truncated @ Vt_truncated

        # Add back the mean
        reconstructed += self._mean_spectrum

        return reconstructed

    def find_n_components_for_variance(self, variance_threshold: float = 0.95) -> int:
        """
        Find minimum number of components to explain given variance.

        Args:
            variance_threshold: Target cumulative variance (0 to 1)

        Returns:
            Number of components needed
        """
        if self.explained_variance_ratio is None:
            raise RuntimeError("Must call fit() before find_n_components_for_variance()")

        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")

        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return n_components

    def get_explained_variance(self, n_components: int) -> float:
        """
        Get cumulative explained variance for given number of components.

        Args:
            n_components: Number of components

        Returns:
            Cumulative explained variance ratio (0 to 1)
        """
        if self.explained_variance_ratio is None:
            raise RuntimeError("Must call fit() before get_explained_variance()")

        return np.sum(self.explained_variance_ratio[:n_components])

    def get_reconstruction_error(self, n_components: int, metric: str = 'rmse') -> float:
        """
        Calculate reconstruction error for given number of components.

        Args:
            n_components: Number of components to use
            metric: Error metric ('rmse', 'mae', 'max')

        Returns:
            Reconstruction error value
        """
        reconstructed = self.reconstruct(n_components)
        error = self.spectra_matrix - reconstructed

        if metric == 'rmse':
            return np.sqrt(np.mean(error**2))
        elif metric == 'mae':
            return np.mean(np.abs(error))
        elif metric == 'max':
            return np.max(np.abs(error))
        else:
            raise ValueError(f"Unknown metric: {metric}")


def perform_svd_decomposition(spectra_matrix: np.ndarray,
                              n_components: int,
                              wavelengths: np.ndarray = None,
                              center: bool = True) -> Tuple[np.ndarray, SVDSpectralAnalyzer]:
    """
    Perform SVD decomposition and reconstruction with given number of components.

    Args:
        spectra_matrix: 2D array of shape (n_spectra, n_wavelengths)
        n_components: Number of principal components to use for reconstruction
        wavelengths: Optional wavelength array
        center: If True, center data by subtracting mean

    Returns:
        Tuple of (reconstructed_spectra, analyzer_object)
    """
    analyzer = SVDSpectralAnalyzer(spectra_matrix, wavelengths)
    analyzer.fit(center=center)
    reconstructed = analyzer.reconstruct(n_components)

    return reconstructed, analyzer


def find_components_for_variance(spectra_matrix: np.ndarray,
                                 variance_threshold: float = 0.95,
                                 wavelengths: np.ndarray = None,
                                 center: bool = True) -> Tuple[int, SVDSpectralAnalyzer]:
    """
    Find the number of components needed to explain given variance.

    Args:
        spectra_matrix: 2D array of shape (n_spectra, n_wavelengths)
        variance_threshold: Target cumulative variance (0 to 1)
        wavelengths: Optional wavelength array
        center: If True, center data by subtracting mean

    Returns:
        Tuple of (n_components, analyzer_object)
    """
    analyzer = SVDSpectralAnalyzer(spectra_matrix, wavelengths)
    analyzer.fit(center=center)
    n_components = analyzer.find_n_components_for_variance(variance_threshold)

    return n_components, analyzer


def plot_svd_reconstruction(analyzer: SVDSpectralAnalyzer,
                            n_components: int,
                            spectrum_indices: list = None,
                            figsize: Tuple[int, int] = (14, 10),
                            save_path: str = None):
    """
    Plot original spectra vs SVD-reconstructed spectra.

    Creates a multi-panel figure showing:
    - Original vs reconstructed spectra
    - Explained variance by component
    - Cumulative variance
    - Reconstruction error

    Args:
        analyzer: Fitted SVDSpectralAnalyzer object
        n_components: Number of components used for reconstruction
        spectrum_indices: List of spectrum indices to plot (None = plot all)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
    """
    if analyzer.U is None:
        raise RuntimeError("Analyzer must be fitted before plotting")

    # Reconstruct spectra
    reconstructed = analyzer.reconstruct(n_components)

    # Select spectra to plot
    if spectrum_indices is None:
        spectrum_indices = range(min(analyzer.n_spectra, 5))  # Max 5 spectra

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Original vs Reconstructed Spectra
    ax1 = fig.add_subplot(gs[0:2, 0])
    for idx in spectrum_indices:
        ax1.plot(analyzer.wavelengths, analyzer.spectra_matrix[idx, :],
                'o-', alpha=0.7, label=f'Original #{idx}', markersize=4)
        ax1.plot(analyzer.wavelengths, reconstructed[idx, :],
                '--', linewidth=2, label=f'Reconstructed #{idx}')

    ax1.set_xlabel('Wavelength (μm)', fontsize=11)
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.set_title(f'Original vs SVD Reconstruction ({n_components} components)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Explained Variance by Component
    ax2 = fig.add_subplot(gs[0, 1])
    n_plot_components = min(20, len(analyzer.explained_variance_ratio))
    ax2.bar(range(1, n_plot_components + 1),
            analyzer.explained_variance_ratio[:n_plot_components] * 100,
            color='steelblue', alpha=0.7)
    ax2.axvline(n_components, color='red', linestyle='--', linewidth=2,
                label=f'Selected: {n_components}')
    ax2.set_xlabel('Component Number', fontsize=11)
    ax2.set_ylabel('Explained Variance (%)', fontsize=11)
    ax2.set_title('Variance Explained by Each Component', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cumulative Explained Variance
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative_variance = np.cumsum(analyzer.explained_variance_ratio) * 100
    ax3.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
            'o-', color='green', linewidth=2, markersize=5)
    ax3.axvline(n_components, color='red', linestyle='--', linewidth=2,
                label=f'Selected: {n_components}')
    ax3.axhline(analyzer.get_explained_variance(n_components) * 100,
               color='red', linestyle=':', linewidth=1.5,
               label=f'Variance: {analyzer.get_explained_variance(n_components)*100:.2f}%')
    ax3.set_xlabel('Number of Components', fontsize=11)
    ax3.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax3.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])

    # 4. Reconstruction Error
    ax4 = fig.add_subplot(gs[2, :])
    for idx in spectrum_indices:
        error = analyzer.spectra_matrix[idx, :] - reconstructed[idx, :]
        ax4.plot(analyzer.wavelengths, error, 'o-', alpha=0.7,
                label=f'Error #{idx}', markersize=3)

    rmse = analyzer.get_reconstruction_error(n_components, 'rmse')
    mae = analyzer.get_reconstruction_error(n_components, 'mae')

    ax4.set_xlabel('Wavelength (μm)', fontsize=11)
    ax4.set_ylabel('Reconstruction Error', fontsize=11)
    ax4.set_title(f'Reconstruction Error (RMSE={rmse:.6f}, MAE={mae:.6f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_variance_analysis(analyzer: SVDSpectralAnalyzer,
                          max_components: int = None,
                          variance_thresholds: list = [0.90, 0.95, 0.99],
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: str = None):
    """
    Plot detailed variance analysis to help choose number of components.

    Args:
        analyzer: Fitted SVDSpectralAnalyzer object
        max_components: Maximum number of components to show (None = all)
        variance_thresholds: List of variance thresholds to highlight
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
    """
    if analyzer.U is None:
        raise RuntimeError("Analyzer must be fitted before plotting")

    if max_components is None:
        max_components = len(analyzer.explained_variance_ratio)
    else:
        max_components = min(max_components, len(analyzer.explained_variance_ratio))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Individual variance explained
    ax = axes[0, 0]
    ax.plot(range(1, max_components + 1),
           analyzer.explained_variance_ratio[:max_components] * 100,
           'o-', color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel('Component Number', fontsize=11)
    ax.set_ylabel('Explained Variance (%)', fontsize=11)
    ax.set_title('Variance by Component', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Cumulative variance
    ax = axes[0, 1]
    cumulative = np.cumsum(analyzer.explained_variance_ratio[:max_components]) * 100
    ax.plot(range(1, max_components + 1), cumulative,
           'o-', color='green', linewidth=2, markersize=6)

    # Mark thresholds
    for threshold in variance_thresholds:
        n_comp = analyzer.find_n_components_for_variance(threshold)
        if n_comp <= max_components:
            ax.axhline(threshold * 100, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(n_comp, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.plot(n_comp, threshold * 100, 'ro', markersize=8)
            ax.text(n_comp + 0.5, threshold * 100, f'{n_comp} comp.\n({threshold*100:.0f}%)',
                   fontsize=9, verticalalignment='center')

    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # 3. Singular values
    ax = axes[1, 0]
    ax.semilogy(range(1, max_components + 1),
               analyzer.S[:max_components],
               'o-', color='purple', linewidth=2, markersize=6)
    ax.set_xlabel('Component Number', fontsize=11)
    ax.set_ylabel('Singular Value (log scale)', fontsize=11)
    ax.set_title('Singular Values', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # 4. Reconstruction error vs components
    ax = axes[1, 1]
    n_range = range(1, min(max_components + 1, 21))
    errors_rmse = [analyzer.get_reconstruction_error(n, 'rmse') for n in n_range]
    errors_mae = [analyzer.get_reconstruction_error(n, 'mae') for n in n_range]

    ax.plot(n_range, errors_rmse, 'o-', label='RMSE', linewidth=2, markersize=6)
    ax.plot(n_range, errors_mae, 's-', label='MAE', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Reconstruction Error', fontsize=11)
    ax.set_title('Reconstruction Error vs Components', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic spectral data for demonstration
    print("SVD Spectral Analysis - Example Usage\n" + "="*60)

    # Create synthetic spectra (e.g., Gaussian peaks with noise)
    n_spectra = 10
    n_wavelengths = 100
    wavelengths = np.linspace(2.5, 4.0, n_wavelengths)

    # Generate base spectra with Gaussian peaks
    spectra = []
    for i in range(n_spectra):
        # Random Gaussian peak
        center = 3.0 + np.random.randn() * 0.3
        width = 0.2 + np.random.randn() * 0.05
        amplitude = 0.8 + np.random.randn() * 0.1

        spectrum = amplitude * np.exp(-((wavelengths - center) / width)**2)
        # Add noise
        spectrum += np.random.randn(n_wavelengths) * 0.02
        spectra.append(spectrum)

    spectra_matrix = np.array(spectra)

    print(f"Generated {n_spectra} synthetic spectra with {n_wavelengths} wavelength points")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} μm\n")

    # Example 1: SVD with fixed number of components
    print("Example 1: SVD reconstruction with 3 components")
    print("-" * 60)
    reconstructed, analyzer = perform_svd_decomposition(spectra_matrix,
                                                        n_components=3,
                                                        wavelengths=wavelengths)
    explained_var = analyzer.get_explained_variance(3)
    print(f"Explained variance with 3 components: {explained_var*100:.2f}%")
    print(f"RMSE: {analyzer.get_reconstruction_error(3, 'rmse'):.6f}\n")

    # Example 2: Find components for target variance
    print("Example 2: Find components for 95% variance")
    print("-" * 60)
    n_comp, analyzer2 = find_components_for_variance(spectra_matrix,
                                                     variance_threshold=0.95,
                                                     wavelengths=wavelengths)
    print(f"Number of components needed for 95% variance: {n_comp}")
    actual_var = analyzer2.get_explained_variance(n_comp)
    print(f"Actual explained variance: {actual_var*100:.2f}%\n")

    # Example 3: Generate plots
    print("Example 3: Generating visualization plots")
    print("-" * 60)
    print("Generating SVD reconstruction plot...")
    plot_svd_reconstruction(analyzer, n_components=3, spectrum_indices=[0, 1, 2])

    print("Generating variance analysis plot...")
    plot_variance_analysis(analyzer, max_components=10,
                          variance_thresholds=[0.90, 0.95, 0.99])

    print("\n" + "="*60)
    print("Example complete!")
