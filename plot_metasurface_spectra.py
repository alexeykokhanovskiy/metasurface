import numpy as np
import matplotlib.pyplot as plt
import h5py
from metasurface import Metasurface
from surface_optimization import (
    load_metasurface_from_results,
    get_best_metasurface,
    analyze_optimization_results
)
from data_processing import load_experimental_curve


def load_experimental_data_from_file(filename: str, wl_min: float = 2.5, wl_max: float = 4.0,
                                      col_index: int = 6) -> tuple:
    """
    Load experimental data from CSV file.

    Args:
        filename: Path to experimental data CSV file
        wl_min: Minimum wavelength (μm)
        wl_max: Maximum wavelength (μm)
        col_index: Column index in CSV file

    Returns:
        Tuple of (wavelength array, signal array)
    """
    wl_exp, signal_exp = load_experimental_curve(
        wl_min=wl_min, wl_max=wl_max,
        experimental_file_path=filename,
        col_index=col_index
    )
    signal_exp /= 100.0  # Convert percentage to fraction
    return wl_exp, signal_exp


def load_experimental_data_from_hdf5(filename: str) -> tuple:
    """
    Load experimental data parameters from HDF5 optimization results file.

    Args:
        filename: Path to optimization results HDF5 file

    Returns:
        Tuple of (experimental_file_path, wl_min, wl_max) or None if not found
    """
    with h5py.File(filename, 'r') as f:
        if 'optimization_summary' in f:
            summary = f['optimization_summary']
            exp_file = summary.attrs.get('experimental_file', None)
            wl_range = summary.attrs.get('wavelength_range', [2.5, 4.0])
            if exp_file:
                return exp_file, wl_range[0], wl_range[1]
    return None


def plot_metasurface_spectrum(metasurface: Metasurface, title: str = None, save_path: str = None,
                             experimental_data: tuple = None):
    """
    Plot all spectra from a metasurface.

    Args:
        metasurface: Metasurface object to plot
        title: Optional title for the plot
        save_path: Optional path to save the figure
        experimental_data: Optional tuple of (wavelength, signal) for experimental data
    """
    n_spectra = len(metasurface.spectra)

    if n_spectra == 0:
        print("No spectra found in metasurface")
        return

    # Create figure with extra space on the right for text
    fig = plt.figure(figsize=(13, 6))
    ax = plt.subplot2grid((1, 4), (0, 0), colspan=3, fig=fig)

    # Plot experimental data first if provided
    if experimental_data is not None:
        wl_exp, signal_exp = experimental_data
        ax.plot(wl_exp, signal_exp, 'o-', label='Experimental (Target)',
                linewidth=2.5, markersize=6, color='black', zorder=10)

    # Plot each spectrum
    for name, spectrum in metasurface.spectra.items():
        label = f"{name} ({spectrum.type}, {spectrum.polarization})"
        ax.plot(spectrum.wavelength, spectrum.value, marker='s', label=label,
                linewidth=2, markersize=5, alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Metasurface Spectrum', fontsize=14, fontweight='bold')

    # Add geometry and materials info on the right side
    info_text = metasurface.geometry.summary_detailed()
    info_text += f"\n\nMaterials:\n"
    for key, val in metasurface.materials.materials.items():
        info_text += f"  {key}: {val}\n"

    # Place text on the right side of the figure
    fig.text(0.75, 0.5, info_text,
             fontsize=10,
             verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1))

    plt.tight_layout(rect=[0, 0, 0.72, 1])  # Leave space for text on right

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Don't show the plot when saving - just close it
    else:
        plt.show()


def plot_multiple_metasurfaces(filename: str, indices: list, spectrum_name: str = "R_TM",
                               experimental_data: tuple = None):
    """
    Plot spectra from multiple metasurfaces for comparison.

    Args:
        filename: Path to the optimization results HDF5 file
        indices: List of candidate indices to plot
        spectrum_name: Name of the spectrum to plot (default: "R_TM")
        experimental_data: Optional tuple of (wavelength, signal) for experimental data
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot experimental data first if provided
    if experimental_data is not None:
        wl_exp, signal_exp = experimental_data
        ax.plot(wl_exp, signal_exp, 'o-', label='Experimental (Target)',
                linewidth=3, markersize=7, color='black', zorder=10)

    with h5py.File(filename, 'r') as f:
        for idx in indices:
            candidate_name = f'candidate_{idx:05d}'
            if candidate_name not in f['metasurfaces']:
                print(f"Warning: Candidate {idx} not found, skipping...")
                continue

            ms_group = f['metasurfaces'][candidate_name]
            mae = ms_group.attrs['mae']

            # Load metasurface
            ms = Metasurface._read_from_group(ms_group)

            if spectrum_name in ms.spectra:
                spectrum = ms.spectra[spectrum_name]
                label = f"Candidate {idx} (MAE={mae:.6e})"
                ax.plot(spectrum.wavelength, spectrum.value, marker='s', label=label,
                        linewidth=2, markersize=4, alpha=0.8)
            else:
                print(f"Warning: Spectrum '{spectrum_name}' not found in candidate {idx}")

    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title(f'Comparison of Multiple Metasurfaces ({spectrum_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_best_vs_experimental(filename: str, experimental_file: str = './LIPPS_sio2_polarized.csv',
                               col_index: int = 6, wl_min: float = 2.5, wl_max: float = 4.0):
    """
    Plot the best metasurface spectrum against experimental data.

    Args:
        filename: Path to the optimization results HDF5 file
        experimental_file: Path to experimental data CSV
        col_index: Column index in experimental file
        wl_min: Minimum wavelength
        wl_max: Maximum wavelength
    """
    from data_processing import load_experimental_curve

    # Load experimental data
    wl_exp, signal_exp = load_experimental_curve(
        wl_min=wl_min, wl_max=wl_max,
        experimental_file_path=experimental_file,
        col_index=col_index
    )
    signal_exp /= 100.0

    # Load best metasurface
    best_ms, best_idx, best_mae = get_best_metasurface(filename)

    # Plot with extra space on the right
    fig = plt.figure(figsize=(14, 7))
    ax = plt.subplot2grid((1, 4), (0, 0), colspan=3, fig=fig)

    # Experimental data
    ax.plot(wl_exp, signal_exp, 'o-', label='Experimental', linewidth=2, markersize=6, color='black')

    # Best model
    spectrum = best_ms.spectra["R_TM"]
    ax.plot(spectrum.wavelength, spectrum.value, 's-', label=f'Best Model (MAE={best_mae:.6e})',
            linewidth=2, markersize=5, color='red')

    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Best Metasurface vs Experimental Data', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    # Add geometry info on the right side
    info_text = f"Best Candidate: {best_idx}\n"
    info_text += f"MAE: {best_mae:.6e}\n\n"
    info_text += best_ms.geometry.summary_detailed()
    info_text += f"\n\nMaterials:\n"
    for key, val in best_ms.materials.materials.items():
        info_text += f"  {key}: {val}\n"

    fig.text(0.75, 0.5, info_text,
             fontsize=10,
             verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, pad=1))

    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.show()


def interactive_plotter(filename: str):
    """
    Interactive plotting tool with menu.

    Args:
        filename: Path to the optimization results HDF5 file
    """
    with h5py.File(filename, 'r') as f:
        total_count = f['metasurfaces'].attrs['total_count']

    print("\n" + "="*70)
    print("METASURFACE SPECTRUM PLOTTER")
    print("="*70)
    print(f"File: {filename}")
    print(f"Total metasurfaces: {total_count}")
    print("="*70)

    # Try to load experimental data from HDF5
    exp_data = None
    exp_info = load_experimental_data_from_hdf5(filename)
    if exp_info:
        exp_file, wl_min, wl_max = exp_info
        print(f"\nExperimental data found: {exp_file}")
        load_exp = input("Load experimental data for overlay? (y/n): ").strip().lower()
        if load_exp == 'y':
            try:
                # Try with default column
                exp_data = load_experimental_data_from_file(exp_file, wl_min, wl_max, col_index=6)
                print(f"✓ Experimental data loaded: {len(exp_data[0])} points")
            except Exception as e:
                print(f"✗ Could not load experimental data: {e}")
                exp_data = None
    else:
        print("\nNo experimental data info in HDF5 file.")
        manual_load = input("Manually load experimental data? (y/n): ").strip().lower()
        if manual_load == 'y':
            exp_file = input("Enter path to experimental CSV: ").strip()
            wl_min = float(input("Enter min wavelength (μm) [2.5]: ").strip() or "2.5")
            wl_max = float(input("Enter max wavelength (μm) [4.0]: ").strip() or "4.0")
            col_index = int(input("Enter column index [6]: ").strip() or "6")
            try:
                exp_data = load_experimental_data_from_file(exp_file, wl_min, wl_max, col_index)
                print(f"✓ Experimental data loaded: {len(exp_data[0])} points")
            except Exception as e:
                print(f"✗ Could not load experimental data: {e}")
                exp_data = None

    while True:
        print("\nOptions:")
        print("  1. Plot specific metasurface by index")
        print("  2. Plot best metasurface")
        print("  3. Plot top N metasurfaces")
        print("  4. Compare multiple metasurfaces")
        print("  5. Plot best vs experimental")
        print("  6. Show optimization summary")
        print("  7. Toggle experimental data overlay")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == '0':
            print("Exiting...")
            break

        elif choice == '1':
            idx = int(input(f"Enter metasurface index (0-{total_count-1}): "))
            if 0 <= idx < total_count:
                ms = load_metasurface_from_results(filename, idx)
                with h5py.File(filename, 'r') as f:
                    mae = f['metasurfaces'][f'candidate_{idx:05d}'].attrs['mae']
                plot_metasurface_spectrum(ms, title=f"Candidate {idx} (MAE={mae:.6e})",
                                         experimental_data=exp_data)
            else:
                print(f"Invalid index! Must be between 0 and {total_count-1}")

        elif choice == '2':
            best_ms, best_idx, best_mae = get_best_metasurface(filename)
            print(f"\nBest metasurface: candidate {best_idx}, MAE = {best_mae:.6e}")
            plot_metasurface_spectrum(best_ms, title=f"Best Candidate {best_idx} (MAE={best_mae:.6e})",
                                     experimental_data=exp_data)

        elif choice == '3':
            n = int(input("How many top metasurfaces to plot? "))
            # Get all MAEs and sort
            with h5py.File(filename, 'r') as f:
                maes = [(f['metasurfaces'][name].attrs['index'],
                         f['metasurfaces'][name].attrs['mae'])
                        for name in f['metasurfaces'].keys()]
            maes.sort(key=lambda x: x[1])
            top_indices = [idx for idx, _ in maes[:n]]
            plot_multiple_metasurfaces(filename, top_indices, experimental_data=exp_data)

        elif choice == '4':
            indices_str = input("Enter indices separated by commas (e.g., 0,5,10): ")
            indices = [int(x.strip()) for x in indices_str.split(',')]
            plot_multiple_metasurfaces(filename, indices, experimental_data=exp_data)

        elif choice == '5':
            plot_best_vs_experimental(filename)

        elif choice == '6':
            analyze_optimization_results(filename)

        elif choice == '7':
            if exp_data is not None:
                # Turn off experimental data
                exp_data = None
                print("✓ Experimental data overlay disabled")
            else:
                # Try to load experimental data
                exp_info = load_experimental_data_from_hdf5(filename)
                if exp_info:
                    exp_file, wl_min, wl_max = exp_info
                    col_index = 6
                else:
                    exp_file = input("Enter path to experimental CSV: ").strip()
                    wl_min = float(input("Enter min wavelength (μm) [2.5]: ").strip() or "2.5")
                    wl_max = float(input("Enter max wavelength (μm) [4.0]: ").strip() or "4.0")
                    col_index = int(input("Enter column index [6]: ").strip() or "6")

                try:
                    exp_data = load_experimental_data_from_file(exp_file, wl_min, wl_max, col_index)
                    print(f"✓ Experimental data loaded and overlay enabled: {len(exp_data[0])} points")
                except Exception as e:
                    print(f"✗ Could not load experimental data: {e}")
                    exp_data = None

        else:
            print("Invalid choice!")


def quick_plot_best_with_exp(results_file: str, experimental_file: str = None,
                              wl_min: float = 2.5, wl_max: float = 4.0, col_index: int = 6):
    """
    Quick plot of best metasurface with experimental overlay.

    Args:
        results_file: Path to optimization results HDF5 file
        experimental_file: Path to experimental CSV file (optional, reads from HDF5 if None)
        wl_min: Minimum wavelength
        wl_max: Maximum wavelength
        col_index: Column index in experimental CSV
    """
    # Load experimental data
    if experimental_file is None:
        exp_info = load_experimental_data_from_hdf5(results_file)
        if exp_info:
            experimental_file, wl_min, wl_max = exp_info

    if experimental_file:
        exp_data = load_experimental_data_from_file(experimental_file, wl_min, wl_max, col_index)
    else:
        exp_data = None
        print("Warning: No experimental data found")

    # Load and plot best metasurface
    best_ms, best_idx, best_mae = get_best_metasurface(results_file)
    plot_metasurface_spectrum(best_ms, title=f"Best Candidate {best_idx} (MAE={best_mae:.6e})",
                              experimental_data=exp_data)


def quick_compare_top_with_exp(results_file: str, n: int = 5, experimental_file: str = None,
                                wl_min: float = 2.5, wl_max: float = 4.0, col_index: int = 6):
    """
    Quick comparison plot of top N metasurfaces with experimental overlay.

    Args:
        results_file: Path to optimization results HDF5 file
        n: Number of top candidates to plot
        experimental_file: Path to experimental CSV file (optional, reads from HDF5 if None)
        wl_min: Minimum wavelength
        wl_max: Maximum wavelength
        col_index: Column index in experimental CSV
    """
    # Load experimental data
    if experimental_file is None:
        exp_info = load_experimental_data_from_hdf5(results_file)
        if exp_info:
            experimental_file, wl_min, wl_max = exp_info

    if experimental_file:
        exp_data = load_experimental_data_from_file(experimental_file, wl_min, wl_max, col_index)
    else:
        exp_data = None
        print("Warning: No experimental data found")

    # Get top indices
    with h5py.File(results_file, 'r') as f:
        maes = [(f['metasurfaces'][name].attrs['index'],
                 f['metasurfaces'][name].attrs['mae'])
                for name in f['metasurfaces'].keys()]
    maes.sort(key=lambda x: x[1])
    top_indices = [idx for idx, _ in maes[:n]]

    # Plot
    plot_multiple_metasurfaces(results_file, top_indices, experimental_data=exp_data)


def export_all_spectra_to_png(filename: str, output_folder: str = None,
                              experimental_data: tuple = None, export_top_n: int = None):
    """
    Export all metasurface spectra from HDF5 file to PNG images.

    Args:
        filename: Path to the optimization results HDF5 file
        output_folder: Folder to save PNG files (auto-generated if None)
        experimental_data: Optional tuple of (wavelength, signal) for experimental data overlay
        export_top_n: If specified, only export top N candidates by MAE (None = export all)
    """
    import os
    from datetime import datetime

    # Use non-interactive backend to prevent plot windows from opening
    import matplotlib
    matplotlib.use('Agg')

    # Create output folder if not specified
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        # Place output folder in the same directory as the HDF5 file
        hdf5_dir = os.path.dirname(os.path.abspath(filename))
        output_folder = os.path.join(hdf5_dir, f"{base_name}_spectra_{timestamp}")

    # Create folder
    os.makedirs(output_folder, exist_ok=True)
    abs_output_folder = os.path.abspath(output_folder)
    print(f"\nExporting spectra to folder: {abs_output_folder}")

    # Get all metasurfaces with MAE
    with h5py.File(filename, 'r') as f:
        metasurface_data = []
        for name in f['metasurfaces'].keys():
            idx = f['metasurfaces'][name].attrs['index']
            mae = f['metasurfaces'][name].attrs['mae']
            metasurface_data.append((idx, mae, name))

    # Sort by MAE
    metasurface_data.sort(key=lambda x: x[1])

    # Limit to top N if specified
    if export_top_n is not None:
        metasurface_data = metasurface_data[:export_top_n]
        print(f"Exporting top {export_top_n} metasurfaces by MAE")
    else:
        print(f"Exporting all {len(metasurface_data)} metasurfaces")

    # Load each metasurface and save plot
    for rank, (idx, mae, name) in enumerate(metasurface_data, start=1):
        ms = load_metasurface_from_results(filename, idx)

        # Generate filename
        png_filename = os.path.join(output_folder, f"candidate_{idx:05d}_mae_{mae:.6e}_rank_{rank:03d}.png")

        # Create plot
        title = f"Candidate {idx} (Rank {rank}, MAE={mae:.6e})"
        plot_metasurface_spectrum(ms, title=title, save_path=png_filename,
                                 experimental_data=experimental_data)

        # Close the plot to free memory
        plt.close('all')

        # Progress
        if rank % 10 == 0:
            print(f"  Exported {rank}/{len(metasurface_data)} spectra...")

    print(f"\n✓ Export complete! {len(metasurface_data)} PNG files saved to '{abs_output_folder}'")

    # Create summary text file
    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("METASURFACE SPECTRA EXPORT SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Source file: {filename}\n")
        f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total exported: {len(metasurface_data)}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Ranking by MAE:\n")
        f.write("="*70 + "\n")
        for rank, (idx, mae, name) in enumerate(metasurface_data, start=1):
            f.write(f"Rank {rank:3d} | Candidate {idx:5d} | MAE = {mae:.6e}\n")

    print(f"✓ Summary saved to '{summary_file}'")

    return abs_output_folder


def export_comparison_plots(filename: str, output_folder: str = None,
                            n_per_plot: int = 5, experimental_data: tuple = None):
    """
    Create comparison plots with multiple metasurfaces per plot.

    Args:
        filename: Path to the optimization results HDF5 file
        output_folder: Folder to save PNG files (auto-generated if None)
        n_per_plot: Number of metasurfaces per comparison plot
        experimental_data: Optional tuple of (wavelength, signal) for experimental data
    """
    import os
    from datetime import datetime

    # Use non-interactive backend to prevent plot windows from opening
    import matplotlib
    matplotlib.use('Agg')

    # Create output folder if not specified
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        # Place output folder in the same directory as the HDF5 file
        hdf5_dir = os.path.dirname(os.path.abspath(filename))
        output_folder = os.path.join(hdf5_dir, f"{base_name}_comparisons_{timestamp}")

    # Create folder
    os.makedirs(output_folder, exist_ok=True)
    abs_output_folder = os.path.abspath(output_folder)
    print(f"\nExporting comparison plots to folder: {abs_output_folder}")

    # Get all metasurfaces sorted by MAE
    with h5py.File(filename, 'r') as f:
        metasurface_data = []
        for name in f['metasurfaces'].keys():
            idx = f['metasurfaces'][name].attrs['index']
            mae = f['metasurfaces'][name].attrs['mae']
            metasurface_data.append((idx, mae))

    metasurface_data.sort(key=lambda x: x[1])

    # Create comparison plots
    n_plots = (len(metasurface_data) + n_per_plot - 1) // n_per_plot
    print(f"Creating {n_plots} comparison plots with {n_per_plot} metasurfaces each")

    for plot_num in range(n_plots):
        start_idx = plot_num * n_per_plot
        end_idx = min(start_idx + n_per_plot, len(metasurface_data))

        indices_to_plot = [idx for idx, _ in metasurface_data[start_idx:end_idx]]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot experimental data if provided
        if experimental_data is not None:
            wl_exp, signal_exp = experimental_data
            ax.plot(wl_exp, signal_exp, 'o-', label='Experimental (Target)',
                    linewidth=3, markersize=7, color='black', zorder=10)

        # Plot metasurfaces
        with h5py.File(filename, 'r') as f:
            for rank, idx in enumerate(indices_to_plot, start=start_idx+1):
                candidate_name = f'candidate_{idx:05d}'
                ms_group = f['metasurfaces'][candidate_name]
                mae = ms_group.attrs['mae']

                ms = Metasurface._read_from_group(ms_group)

                if "R_TM" in ms.spectra:
                    spectrum = ms.spectra["R_TM"]
                    label = f"Rank {rank} (Cand {idx}, MAE={mae:.6e})"
                    ax.plot(spectrum.wavelength, spectrum.value, marker='s', label=label,
                            linewidth=2, markersize=4, alpha=0.8)

        ax.set_xlabel('Wavelength (μm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'Comparison: Ranks {start_idx+1}-{end_idx}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()

        # Save
        png_filename = os.path.join(output_folder, f"comparison_ranks_{start_idx+1:03d}_to_{end_idx:03d}.png")
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close('all')

        print(f"  Saved comparison plot {plot_num+1}/{n_plots}")

    print(f"\n✓ Export complete! {n_plots} comparison plots saved to '{abs_output_folder}'")

    return abs_output_folder


if __name__ == "__main__":
    # Default file (change this to your results file)н
    default_file = 'optimization_results_4params_PxWyfixed_20260124_112903.h5'

    import sys
    import os

    if len(sys.argv) > 1:
        results_file = sys.argv[1]

        # Check for command-line export mode
        if len(sys.argv) > 2 and sys.argv[2] == '--export-all':
            if not os.path.exists(results_file):
                print(f"Error: File '{results_file}' not found!")
                sys.exit(1)

            # Load experimental data if available
            exp_data = None
            exp_info = load_experimental_data_from_hdf5(results_file)
            if exp_info:
                exp_file, wl_min, wl_max = exp_info
                try:
                    exp_data = load_experimental_data_from_file(exp_file, wl_min, wl_max, col_index=6)
                    print(f"✓ Experimental data loaded for overlay")
                except:
                    print("✗ Could not load experimental data")

            # Export all spectra
            export_all_spectra_to_png(results_file, experimental_data=exp_data)

            # Also create comparison plots
            export_comparison_plots(results_file, n_per_plot=5, experimental_data=exp_data)

            sys.exit(0)

        elif len(sys.argv) > 2 and sys.argv[2].startswith('--export-top='):
            n = int(sys.argv[2].split('=')[1])
            if not os.path.exists(results_file):
                print(f"Error: File '{results_file}' not found!")
                sys.exit(1)

            # Load experimental data if available
            exp_data = None
            exp_info = load_experimental_data_from_hdf5(results_file)
            if exp_info:
                exp_file, wl_min, wl_max = exp_info
                try:
                    exp_data = load_experimental_data_from_file(exp_file, wl_min, wl_max, col_index=6)
                except:
                    pass

            # Export top N spectra
            export_all_spectra_to_png(results_file, experimental_data=exp_data, export_top_n=n)
            sys.exit(0)
    else:
        results_file = default_file

    print(f"Using results file: {results_file}")

    # Check if file exists
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' not found!")
        print(f"Usage: python {sys.argv[0]} <path_to_results.h5> [--export-all | --export-top=N]")
        sys.exit(1)

    # Start interactive plotter
    interactive_plotter(results_file)
