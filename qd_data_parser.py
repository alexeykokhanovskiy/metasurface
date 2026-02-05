"""
Quantum Dot Fluorescence Data Parser for Machine Learning

Parses QD fluorescence spectra CSV files and prepares data for ML applications.
Structure: Concentration as labels, spectral intensities as features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import re


@dataclass
class QDSpectralDataset:
    """
    Container for quantum dot spectral data prepared for ML.

    Attributes:
        X: Feature matrix (n_samples, n_wavelengths) - intensity values only
        y: Label vector (n_samples,) - concentration values in molar
        y_categorical: Categorical labels (n_samples,) - concentration as strings
        wavelengths: Single wavelength array (n_wavelengths,) in nm - shared by all samples
        concentration_labels: Original concentration strings
        replicate_numbers: Replicate/measurement number for each sample
        metadata: Dictionary with additional information
    """
    X: np.ndarray  # Features: (n_samples, n_wavelengths) - intensities only
    y: np.ndarray  # Labels: (n_samples,) in molar units
    y_categorical: np.ndarray  # Categorical labels
    wavelengths: np.ndarray  # (n_wavelengths,) - ONE array for all samples
    concentration_labels: List[str]
    replicate_numbers: np.ndarray  # (n_samples,) - replicate/measurement numbers
    metadata: dict

    def __post_init__(self):
        """Validate dimensions after initialization."""
        assert self.X.shape[0] == len(self.y), "X and y must have same number of samples"
        assert self.X.shape[1] == len(self.wavelengths), "X columns must match wavelength points"
        assert len(self.y) == len(self.concentration_labels), "Labels must match concentrations"
        assert len(self.y) == len(self.replicate_numbers), "Labels must match replicate numbers"

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split data into train/test sets.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(self.X, self.y, test_size=test_size,
                               random_state=random_state, stratify=None)

    def normalize_features(self, method: str = 'standard'):
        """
        Normalize feature matrix in-place.

        Args:
            method: 'standard' (z-score) or 'minmax' (0-1 range)
        """
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.X = scaler.fit_transform(self.X)
        self.metadata['normalization'] = method
        return scaler

    def get_log_labels(self):
        """
        Get logarithm of concentration labels (useful for regression).

        Returns:
            Log10 of concentrations
        """
        # Avoid log(0) by adding small epsilon
        return np.log10(self.y + 1e-12)


def parse_concentration_string(conc_str: str) -> Tuple[float, str, int]:
    """
    Parse concentration string to numerical value in molar units.

    Args:
        conc_str: String like "Fe + 50 mM_0 (0 hrs)", "Control_1 (0 hrs)", or old format "50 mM"

    Returns:
        Tuple of (concentration_in_molar, original_string, replicate_number)
    """
    conc_str = conc_str.strip()
    replicate_num = 0  # Default replicate number

    # Extract replicate number if present (e.g., "_0", "_1", "_2")
    replicate_match = re.search(r'_(\d+)\s*(?:\(|$)', conc_str)
    if replicate_match:
        replicate_num = int(replicate_match.group(1))
        # Remove the replicate number and everything after it for concentration parsing
        conc_str_for_parsing = conc_str[:replicate_match.start()]
    else:
        conc_str_for_parsing = conc_str

    # Extract number and unit using regex
    match = re.search(r'([\d.]+)\s*(mM|uM|nM|M)', conc_str_for_parsing, re.IGNORECASE)

    if not match:
        # Handle special cases (e.g., control samples)
        return 0.0, conc_str, replicate_num

    value = float(match.group(1))
    unit = match.group(2).lower()

    # Convert to molar
    conversion = {
        'm': 1.0,
        'mm': 1e-3,
        'um': 1e-6,
        'μm': 1e-6,
        'nm': 1e-9
    }

    molar_value = value * conversion.get(unit, 1.0)

    return molar_value, conc_str, replicate_num


def parse_qd_fluorescence_csv(filepath: str,
                              wavelength_range: Optional[Tuple[float, float]] = None,
                              normalize: bool = False) -> QDSpectralDataset:
    """
    Parse quantum dot fluorescence CSV file into ML-ready format.

    Args:
        filepath: Path to CSV file
        wavelength_range: Optional (min_wl, max_wl) to filter wavelengths
        normalize: If True, apply standard normalization to features

    Returns:
        QDSpectralDataset object ready for ML
    """
    # Read CSV file (semicolon separated)
    df = pd.read_csv(filepath, sep=';', header=None, nrows=2)  # Read first 2 rows

    # Parse header row (row 0) to extract concentrations
    header_row = df.iloc[0].values

    # Extract concentration information from header
    concentrations = []
    concentration_strings = []
    replicate_nums = []

    for i in range(0, len(header_row), 2):  # Every other column is a new sample
        header_text = str(header_row[i])
        if pd.notna(header_text) and header_text != 'nan':
            conc_molar, conc_str, replicate_num = parse_concentration_string(header_text)
            concentrations.append(conc_molar)
            concentration_strings.append(conc_str)
            replicate_nums.append(replicate_num)

    n_samples = len(concentrations)
    print(f"Found {n_samples} samples with different concentrations")

    # Read full data (semicolon separated)
    df_full = pd.read_csv(filepath, sep=';', skiprows=2, header=None)

    # Remove empty columns (from double semicolons ;;)
    df_full = df_full.dropna(axis=1, how='all')

    # Remove rows with metadata (non-numeric data in first column)
    # Try to convert first column to numeric, keep only numeric rows
    numeric_mask = pd.to_numeric(df_full.iloc[:, 0], errors='coerce').notna()
    df_full = df_full[numeric_mask].reset_index(drop=True)

    print(f"Data rows after filtering metadata: {len(df_full)}")

    # Extract wavelengths (only once) and intensities for all samples
    wavelengths = df_full.iloc[:, 0].values.astype(float)  # First wavelength column, convert to float
    X_list = []

    for i in range(n_samples):
        intensity_col = i * 2 + 1  # Only intensity columns
        if intensity_col >= df_full.shape[1]:
            print(f"Warning: Not enough columns for sample {i}, stopping")
            break
        intensity = df_full.iloc[:, intensity_col].values.astype(float)  # Convert to float
        X_list.append(intensity)

    # Update n_samples to actual loaded samples
    n_samples = len(X_list)
    concentrations = concentrations[:n_samples]
    concentration_strings = concentration_strings[:n_samples]
    replicate_nums = replicate_nums[:n_samples]

    # Convert to numpy arrays
    X = np.array(X_list, dtype=float)  # Shape: (n_samples, n_wavelengths) - only intensities
    y = np.array(concentrations, dtype=float)  # Shape: (n_samples,)
    y_categorical = np.array(concentration_strings)
    replicate_numbers = np.array(replicate_nums, dtype=int)  # Shape: (n_samples,)

    # Filter wavelength range if specified
    if wavelength_range is not None:
        min_wl, max_wl = wavelength_range
        mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
        wavelengths = wavelengths[mask]
        X = X[:, mask]
        print(f"Filtered to wavelength range {min_wl}-{max_wl} nm: {len(wavelengths)} points")

    # Create dataset
    metadata = {
        'filepath': filepath,
        'n_samples': n_samples,
        'n_wavelengths': len(wavelengths),
        'wavelength_range': (wavelengths.min(), wavelengths.max()),
        'concentration_range_molar': (y.min(), y.max())
    }

    dataset = QDSpectralDataset(
        X=X,
        y=y,
        y_categorical=y_categorical,
        wavelengths=wavelengths,  # Single array for all samples
        concentration_labels=concentration_strings,
        replicate_numbers=replicate_numbers,
        metadata=metadata
    )

    # Normalize if requested
    if normalize:
        dataset.normalize_features(method='standard')

    return dataset


def plot_dataset_overview(dataset: QDSpectralDataset, save_path: str = None):
    """
    Plot overview of the dataset for visualization.

    Args:
        dataset: QDSpectralDataset object
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. All spectra overlaid
    ax = axes[0, 0]
    for i in range(dataset.X.shape[0]):
        ax.plot(dataset.wavelengths, dataset.X[i, :],
               label=dataset.concentration_labels[i], alpha=0.7)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('All Fluorescence Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Concentration vs Peak Intensity
    ax = axes[0, 1]
    peak_intensities = np.max(dataset.X, axis=1)
    ax.scatter(dataset.y * 1000, peak_intensities, s=100, alpha=0.6, c=range(len(dataset.y)))
    ax.set_xlabel('Concentration (mM)', fontsize=11)
    ax.set_ylabel('Peak Intensity (a.u.)', fontsize=11)
    ax.set_title('Concentration vs Peak Intensity', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Feature matrix heatmap
    ax = axes[1, 0]
    im = ax.imshow(dataset.X, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Wavelength Index', fontsize=11)
    ax.set_ylabel('Sample Index', fontsize=11)
    ax.set_title('Feature Matrix (X) Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Intensity')

    # 4. Concentration distribution
    ax = axes[1, 1]
    log_conc = np.log10(dataset.y + 1e-12)
    ax.bar(range(len(dataset.y)), log_conc, color='steelblue', alpha=0.7)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Log10(Concentration [M])', fontsize=11)
    ax.set_title('Concentration Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(dataset.y)))
    ax.set_xticklabels([f"{i}" for i in range(len(dataset.y))], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def export_to_sklearn_format(dataset: QDSpectralDataset, output_prefix: str = 'qd_data'):
    """
    Export dataset to numpy files compatible with scikit-learn.

    Args:
        dataset: QDSpectralDataset object
        output_prefix: Prefix for output files
    """
    np.save(f'{output_prefix}_X.npy', dataset.X)
    np.save(f'{output_prefix}_y.npy', dataset.y)
    np.save(f'{output_prefix}_wavelengths.npy', dataset.wavelengths)  # Saved only once
    np.save(f'{output_prefix}_replicate_numbers.npy', dataset.replicate_numbers)

    # Save metadata as text
    with open(f'{output_prefix}_metadata.txt', 'w') as f:
        f.write("QD Fluorescence Dataset Metadata\n")
        f.write("="*50 + "\n\n")
        for key, value in dataset.metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\nConcentration Labels:\n")
        for i, label in enumerate(dataset.concentration_labels):
            f.write(f"  Sample {i}: {label} ({dataset.y[i]:.2e} M) [replicate {dataset.replicate_numbers[i]}]\n")

    print(f"Exported dataset to {output_prefix}_*.npy and metadata.txt")
    print(f"  - X shape: {dataset.X.shape}")
    print(f"  - y shape: {dataset.y.shape}")
    print(f"  - wavelengths shape: {dataset.wavelengths.shape} (single array for all samples)")
    print(f"  - replicate_numbers shape: {dataset.replicate_numbers.shape}")


# Example usage
if __name__ == "__main__":
    import sys
    import os

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python qd_data_parser.py <path_to_csv> [output_path]")
        print("\nArguments:")
        print("  path_to_csv  - Path to the CSV file to parse")
        print("  output_path  - (Optional) Path prefix for saving output files")
        print("\nExamples:")
        print("  python qd_data_parser.py data.csv")
        print("  python qd_data_parser.py data.csv ./output/parsed_data")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)

    print("\n" + "="*70)
    print("QUANTUM DOT FLUORESCENCE DATA PARSER")
    print("="*70)
    print(f"Loading: {csv_file}\n")

    # Parse the dataset
    dataset = parse_qd_fluorescence_csv(
        csv_file,
        wavelength_range=None,  # Use full wavelength range from data
        normalize=False
    )

    # Print summary
    print("\nDataset Summary:")
    print("-" * 70)
    print(f"Feature matrix X shape: {dataset.X.shape}")
    print(f"Label vector y shape: {dataset.y.shape}")
    print(f"Wavelength array shape: {dataset.wavelengths.shape} (ONE array for all samples)")
    print(f"Wavelength range: {dataset.wavelengths.min():.1f} - {dataset.wavelengths.max():.1f} nm")
    print(f"Number of wavelength points: {len(dataset.wavelengths)}")
    print(f"\nConcentrations (in molar):")
    for i, (label, conc, rep) in enumerate(zip(dataset.concentration_labels, dataset.y, dataset.replicate_numbers)):
        print(f"  Sample {i}: {label:40s} -> {conc:.2e} M [rep {rep}]")

    # If output path provided, save automatically
    if output_path:
        print("\n" + "="*70)
        print("Saving parsed data...")
        print("="*70)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        export_to_sklearn_format(dataset, output_prefix=output_path)

        print("\n" + "="*70)
        print("Done!")
    else:
        # Interactive mode - show detailed info and visualize
        print("\n" + "="*70)
        print("ML-Ready Format Demonstration")
        print("="*70)
        print(f"\nFeature matrix X (sklearn format):")
        print(f"  Shape: {dataset.X.shape}")
        print(f"  Type: {dataset.X.dtype}")
        print(f"  Sample data point: X[0, 0:5] = {dataset.X[0, 0:5]}")

        print(f"\nLabel vector y (regression target):")
        print(f"  Shape: {dataset.y.shape}")
        print(f"  Type: {dataset.y.dtype}")
        print(f"  Values: {dataset.y}")

        print(f"\nWavelength array (shared by all samples):")
        print(f"  Shape: {dataset.wavelengths.shape}")
        print(f"  First 5 values: {dataset.wavelengths[0:5]}")

        print(f"\nLog-transformed labels (for regression):")
        y_log = dataset.get_log_labels()
        print(f"  Log10(y): {y_log}")

        # Visualize
        print("\nGenerating visualization...")
        plot_dataset_overview(dataset)

        # Export option
        export = input("\nExport to numpy files? (y/n): ").strip().lower()
        if export == 'y':
            export_to_sklearn_format(dataset, output_prefix='qd_ml_data')

        print("\n" + "="*70)
        print("Done!")
