import numpy as np
import matlab.engine
import h5py
from cmaes import CMA

from data_processing import (
    load_experimental_curve,
    calc_mae_model_experimental,
    calc_rmse_model_experimental
)
from metasurface import Metasurface, Geometry, Material, Spectrum


# ===============================
# UTILITY FUNCTIONS (safe to import)
# ===============================
def load_metasurface_from_results(filename: str, candidate_idx: int) -> Metasurface:
    """
    Load a specific metasurface from the optimization results file.

    Args:
        filename: Path to the optimization results HDF5 file
        candidate_idx: Index of the candidate to load (0-based)

    Returns:
        Metasurface object
    """
    with h5py.File(filename, 'r') as f:
        candidate_name = f'candidate_{candidate_idx:05d}'
        if candidate_name not in f['metasurfaces']:
            raise ValueError(f"Candidate {candidate_idx} not found in {filename}")

        ms_group = f['metasurfaces'][candidate_name]
        ms = Metasurface._read_from_group(ms_group)

    return ms


def get_best_metasurface(filename: str) -> tuple[Metasurface, int, float]:
    """
    Find and load the best metasurface from optimization results.

    Args:
        filename: Path to the optimization results HDF5 file

    Returns:
        Tuple of (Metasurface object, candidate index, MAE value)
    """
    with h5py.File(filename, 'r') as f:
        grp_ms = f['metasurfaces']

        # Find candidate with minimum MAE
        best_mae = float('inf')
        best_idx = None

        for name in grp_ms.keys():
            mae = grp_ms[name].attrs['mae']
            if mae < best_mae:
                best_mae = mae
                best_idx = grp_ms[name].attrs['index']

        # Load the best metasurface
        best_ms = Metasurface._read_from_group(grp_ms[f'candidate_{best_idx:05d}'])

    return best_ms, best_idx, best_mae


def analyze_optimization_results(filename: str):
    """
    Print summary statistics from optimization results.

    Args:
        filename: Path to the optimization results HDF5 file
    """
    with h5py.File(filename, 'r') as f:
        summary = f['optimization_summary']

        print(f"\n{'='*70}")
        print(f"Optimization Results: {filename}")
        print(f"{'='*70}")
        print(f"Parameters optimized: {summary.attrs['n_parameters']}")
        print(f"Parameter names: {', '.join(summary.attrs['parameter_names'])}")
        print(f"Iterations: {summary.attrs['n_iterations']}")
        print(f"Population size: {summary.attrs['population_size']}")
        print(f"Total evaluations: {summary.attrs['total_evaluations']}")
        print(f"Experimental file: {summary.attrs['experimental_file']}")
        print(f"Wavelength range: {summary.attrs['wavelength_range']} μm")

        # Statistics on losses
        losses = np.array(summary['all_losses'])
        print(f"\nLoss (MAE) statistics:")
        print(f"  Best: {losses.min():.6e}")
        print(f"  Worst: {losses.max():.6e}")
        print(f"  Mean: {losses.mean():.6e}")
        print(f"  Median: {np.median(losses):.6e}")
        print(f"  Std: {losses.std():.6e}")

        print(f"\nTotal metasurfaces saved: {f['metasurfaces'].attrs['total_count']}")
        print(f"{'='*70}\n")


def analyze_metasurface(ms: Metasurface):
    """
    Print detailed information about a metasurface.

    Args:
        ms: Metasurface object to analyze
    """
    print(f"\n{'='*60}")
    print(f"Metasurface Analysis")
    print(f"{'='*60}")
    print(ms.geometry.summary_detailed())
    print(f"\nMaterials:")
    for key, val in ms.materials.materials.items():
        print(f"  {key}: {val}")
    print(f"\nSpectra available: {ms.spectra_summary()}")
    if ms.metadata:
        print(f"\nMetadata: {ms.metadata}")

    # Print spectrum details
    for name, spectrum in ms.spectra.items():
        print(f"\nSpectrum '{name}':")
        print(f"  Type: {spectrum.type}")
        print(f"  Polarization: {spectrum.polarization}")
        print(f"  Incidence angle: {spectrum.incidence_angle}°")
        print(f"  Wavelength range: {spectrum.wavelength.min():.4f} - {spectrum.wavelength.max():.4f} μm")
        print(f"  Value range: {spectrum.value.min():.4f} - {spectrum.value.max():.4f}")

    print(f"{'='*60}\n")


# ===============================
# MAIN OPTIMIZATION SCRIPT
# ===============================
if __name__ == "__main__":
    from datetime import datetime

    # ===============================
    # НАСТРОЙКИ КОНСТАНТ
    # ===============================
    WL_MIN = 1.0
    WL_MAX = 4.0
    N_wl_points = 100
    wl_model = np.linspace(WL_MIN, WL_MAX, N_wl_points)
    EXP_FILE = './LIPPS_sio2_polarized.csv'
    EXP_COL = 6
    # path = 'C:/Users/alexey/YandexDisk/Projects/metasurface_spectral_filters/signal_processing_PCM_metasurfaces/V10_2025/results/new_antennas/'
    path = './V10_2025' # path to the folder with solver functions

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_results = f'optimization_results_4params_PxWyfixed_{timestamp}.h5'
    CACHE_PRECISION = 4


    # -----------------------------
    # ОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ
    # -----------------------------
    parameters_config = [
        ("h_cross",       0.09,  0.01,  0.20),   # толщина креста GST
        ("Px",            1.00,  0.50,  1.50),   # период по x
        ("Wy",            1.00,  0.50,  1.50),   # ширина плеча креста
        ("Gst_thickness", 0.23,  0.205, 0.40),   # общая толщина GST-слоя (подложка + крест)
    ]

    n_dim = len(parameters_config)

    # Формируем массивы для CMA-ES
    initial_mean = np.array([p[1] for p in parameters_config])
    lower_bounds = np.array([p[2] for p in parameters_config])
    upper_bounds = np.array([p[3] for p in parameters_config])
    bounds = np.stack((lower_bounds, upper_bounds)).T  # shape (n_dim, 2)

    # Индивидуальные sigma: 20% от ширины диапазона каждого параметра
    individual_sigmas = 0.2 * (upper_bounds - lower_bounds)

    param_names = [p[0] for p in parameters_config]

    # Параметры CMA-ES
    POPULATION_SIZE = max(12, 4 + int(3 * np.log(n_dim)))  # ~15 для 4 параметров
    N_ITER = 30
    SEED = 42

    # ===============================
    # ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
    # ===============================
    wl_exp, signal_exp = load_experimental_curve(
        wl_min=WL_MIN, wl_max=WL_MAX,
        experimental_file_path=EXP_FILE,
        col_index=EXP_COL
    )
    signal_exp /= 100.0

    # ===============================
    # MATLAB ENGINE
    # ===============================
    print("Запуск MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.pwd(), nargout=0)
    # eng.addpath(path, nargout=0)
    eng.addpath(eng.genpath(path), nargout=0)

    # ===============================
    # CMA-ES ОПТИМИЗАТОР
    # ===============================
    optimizer = CMA(
        mean=initial_mean,
        sigma=0.2,                # массив индивидуальных sigma
        bounds=bounds,
        seed=SEED,
        population_size=POPULATION_SIZE
    )

    all_solutions = []

    print(f"\nЗапуск оптимизации по {n_dim} параметрам:")
    for name, start, low, high in parameters_config:
        sigma_val = individual_sigmas[param_names.index(name)]
        print(f"  {name}: [{low:.3f}, {high:.3f}], start = {start:.3f}, sigma = {sigma_val:.3f}")
    # ===============================
    # ЦИКЛ ОПТИМИЗАЦИИ
    # ===============================
    all_metasurfaces = []

    for iteration in range(N_ITER):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()  # np.array shape (4,)

            # Передаём параметры в MATLAB в нужном порядке:
            # h_cross, Px, Wy, Gst_thickness
            h_cross = float(x[0])
            Px      = float(x[1])
            Wy      = float(x[2])
            Gst_thickness = float(x[3])

            signal_model_matlab, _ = eng.r_t_calc_multi_param(
                WL_MIN, WL_MAX, N_wl_points,
                h_cross, Px, Wy, Gst_thickness,
                nargout=2
            )

            signal_model = np.array(signal_model_matlab).flatten()
            value = calc_mae_model_experimental(wl_exp, wl_model, signal_exp, signal_model)

            # Create Metasurface object for this candidate
            geometry = Geometry(
                name="cross_antenna",
                parameters={
                    "h_cross": h_cross,
                    "Px": Px,
                    "Wy": Wy,
                    "Gst_thickness": Gst_thickness
                }
            )

            materials = Material(materials={
                "substrate": "SiO2",
                "cross_layer": "GST",
                "background": "air"
            })

            spectrum = Spectrum(
                wavelength=wl_model,
                value=signal_model,
                type="R",  # Reflection
                polarization="TM",
                incidence_angle=0.0
            )

            metasurface = Metasurface(
                geometry=geometry,
                materials=materials,
                spectra={"R_TM": spectrum},
                metadata=f"Optimization iteration {iteration+1}, MAE={value:.6e}"
            )

            solutions.append((x.copy(), value, metasurface))
            all_metasurfaces.append((metasurface, value))

        # Tell optimizer (only need x and value)
        optimizer.tell([(x, val) for x, val, _ in solutions])

        # Лучшее решение в текущей популяции
        best_x, best_loss, best_metasurface = min(solutions, key=lambda item: item[1])
        best_params = dict(zip(param_names, best_x))

        print(f"Iter {iteration+1:02d}/{N_ITER} | "
              f"best MAE = {best_loss:.6e} | "
              f"params = {best_params}")

        all_solutions.extend([(x, val) for x, val, _ in solutions])

    # ===============================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В HDF5
    # ===============================
    # Save all data in a single HDF5 file
    print(f"\nСохранение результатов в '{filename_results}'...")

    with h5py.File(filename_results, 'w') as f:
        # ---------- Optimization summary ----------
        grp_summary = f.create_group('optimization_summary')
        grp_summary.attrs['n_parameters'] = n_dim
        grp_summary.attrs['parameter_names'] = param_names
        grp_summary.attrs['n_iterations'] = N_ITER
        grp_summary.attrs['population_size'] = POPULATION_SIZE
        grp_summary.attrs['total_evaluations'] = len(all_solutions)
        grp_summary.attrs['experimental_file'] = EXP_FILE
        grp_summary.attrs['wavelength_range'] = [WL_MIN, WL_MAX]

        # Save loss values and parameters for all candidates
        losses = np.array([loss for _, loss in all_solutions])
        grp_summary.create_dataset('all_losses', data=losses)

        params_array = np.array([params for params, _ in all_solutions])
        grp_summary.create_dataset('all_parameters', data=params_array)

        # ---------- Save all metasurfaces ----------
        grp_metasurfaces = f.create_group('metasurfaces')
        grp_metasurfaces.attrs['total_count'] = len(all_metasurfaces)

        print(f"Сохранение {len(all_metasurfaces)} метаповерхностей...")
        for i, (metasurface, mae) in enumerate(all_metasurfaces):
            # Create a group for each metasurface
            ms_group = grp_metasurfaces.create_group(f'candidate_{i:05d}')
            ms_group.attrs['mae'] = mae
            ms_group.attrs['index'] = i

            # Write metasurface data to this group
            metasurface._write_to_group(ms_group)

            # Progress indicator every 10 candidates
            if (i + 1) % 10 == 0:
                print(f"  Сохранено {i + 1}/{len(all_metasurfaces)} метаповерхностей...")

    # Find and report best result
    best_metasurface, best_mae = min(all_metasurfaces, key=lambda item: item[1])
    best_idx = all_metasurfaces.index((best_metasurface, best_mae))

    print(f"\nОптимизация завершена!")
    print(f"Всего кандидатов: {len(all_solutions)}")
    print(f"Лучшая метаповерхность: candidate_{best_idx:05d}, MAE = {best_mae:.6e}")
    print(f"\nЛучшие параметры:")
    for name, value in best_metasurface.geometry.parameters.items():
        print(f"  {name} = {value:.4f}")
    print(f"\nВсе результаты сохранены в '{filename_results}'")

    # ===============================
    # ЗАВЕРШЕНИЕ
    # ===============================
    eng.quit()
