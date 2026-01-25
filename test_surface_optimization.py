import numpy as np
import matlab.engine
import h5py
from cmaes import CMA
from datetime import datetime

from data_processing import (
    load_experimental_curve,
    calc_mae_model_experimental,
    calc_rmse_model_experimental
)
from metasurface import Metasurface, Geometry, Material, Spectrum


# ===============================
# TEST CONFIGURATION
# ===============================
print("="*70)
print("TEST MODE: Running optimization for 1 iteration only")
print("="*70)

# ===============================
# НАСТРОЙКИ КОНСТАНТ
# ===============================
WL_MIN = 2.5
WL_MAX = 4.0
N_wl_points = 50
wl_model = np.linspace(WL_MIN, WL_MAX, N_wl_points)
EXP_FILE = './LIPPS_sio2_polarized.csv'
EXP_COL = 6
path = './V10_2025' # path to the folder with solver functions

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_results = f'test_optimization_results_{timestamp}.h5'
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

# TEST PARAMETERS - Reduced for quick testing
POPULATION_SIZE = 5  # Reduced from ~15 to 5 for testing
N_ITER = 1  # Single iteration for testing
SEED = 42

print(f"\nTest configuration:")
print(f"  Population size: {POPULATION_SIZE}")
print(f"  Iterations: {N_ITER}")
print(f"  Total evaluations: {POPULATION_SIZE * N_ITER}")
print(f"  Output file: {filename_results}")

# ===============================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
# ===============================
print(f"\nLoading experimental data from: {EXP_FILE}")
wl_exp, signal_exp = load_experimental_curve(
    wl_min=WL_MIN, wl_max=WL_MAX,
    experimental_file_path=EXP_FILE,
    col_index=EXP_COL
)
signal_exp /= 100.0
print(f"Experimental data loaded: {len(wl_exp)} wavelength points")

# ===============================
# MATLAB ENGINE
# ===============================
print("\nЗапуск MATLAB engine...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.pwd(), nargout=0)
eng.addpath(eng.genpath(path), nargout=0)
print("MATLAB engine started successfully")

# ===============================
# CMA-ES ОПТИМИЗАТОР
# ===============================
optimizer = CMA(
    mean=initial_mean,
    sigma=0.2,
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
    print(f"\n{'='*70}")
    print(f"Iteration {iteration+1}/{N_ITER}")
    print(f"{'='*70}")

    solutions = []
    for candidate_num in range(optimizer.population_size):
        print(f"\nEvaluating candidate {candidate_num+1}/{optimizer.population_size}...")

        x = optimizer.ask()  # np.array shape (4,)

        # Передаём параметры в MATLAB в нужном порядке:
        # h_cross, Px, Wy, Gst_thickness
        h_cross = float(x[0])
        Px      = float(x[1])
        Wy      = float(x[2])
        Gst_thickness = float(x[3])

        print(f"  Parameters: h_cross={h_cross:.4f}, Px={Px:.4f}, Wy={Wy:.4f}, Gst_thickness={Gst_thickness:.4f}")

        signal_model_matlab, _ = eng.r_t_calc_multi_param(
            WL_MIN, WL_MAX, N_wl_points,
            h_cross, Px, Wy, Gst_thickness,
            nargout=2
        )

        signal_model = np.array(signal_model_matlab).flatten()
        value = calc_mae_model_experimental(wl_exp, wl_model, signal_exp, signal_model)

        print(f"  MAE: {value:.6e}")

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
            metadata=f"Test optimization iteration {iteration+1}, candidate {candidate_num+1}, MAE={value:.6e}"
        )

        solutions.append((x.copy(), value, metasurface))
        all_metasurfaces.append((metasurface, value))

    # Tell optimizer (only need x and value)
    optimizer.tell([(x, val) for x, val, _ in solutions])

    # Лучшее решение в текущей популяции
    best_x, best_loss, best_metasurface = min(solutions, key=lambda item: item[1])
    best_params = dict(zip(param_names, best_x))

    print(f"\n{'='*70}")
    print(f"Iteration {iteration+1}/{N_ITER} Summary:")
    print(f"  Best MAE = {best_loss:.6e}")
    print(f"  Best parameters: {best_params}")
    print(f"{'='*70}")

    all_solutions.extend([(x, val) for x, val, _ in solutions])

# ===============================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В HDF5
# ===============================
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
    grp_summary.attrs['test_mode'] = True  # Mark this as a test run

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

        print(f"  Сохранено {i + 1}/{len(all_metasurfaces)} метаповерхностей...")

# Find and report best result
best_metasurface, best_mae = min(all_metasurfaces, key=lambda item: item[1])
best_idx = all_metasurfaces.index((best_metasurface, best_mae))

print(f"\n{'='*70}")
print(f"TEST OPTIMIZATION COMPLETED!")
print(f"{'='*70}")
print(f"Всего кандидатов: {len(all_solutions)}")
print(f"Лучшая метаповерхность: candidate_{best_idx:05d}, MAE = {best_mae:.6e}")
print(f"\nЛучшие параметры:")
for name, value in best_metasurface.geometry.parameters.items():
    print(f"  {name} = {value:.4f}")
print(f"\nВсе результаты сохранены в '{filename_results}'")
print(f"{'='*70}\n")

# ===============================
# ЗАВЕРШЕНИЕ
# ===============================
eng.quit()

# ===============================
# TEST VERIFICATION
# ===============================
print("\n" + "="*70)
print("TEST VERIFICATION - Loading saved results")
print("="*70)

# Test loading functions
from surface_optimization import (
    load_metasurface_from_results,
    get_best_metasurface,
    analyze_optimization_results,
    analyze_metasurface
)

# Analyze the results
analyze_optimization_results(filename_results)

# Load and verify best metasurface
best_ms, best_idx_loaded, best_mae_loaded = get_best_metasurface(filename_results)
print(f"Best metasurface loaded: candidate {best_idx_loaded}, MAE = {best_mae_loaded:.6e}")
analyze_metasurface(best_ms)

# Verify we can load individual candidates
print("Verifying individual candidate loading...")
for i in range(min(3, len(all_metasurfaces))):
    ms = load_metasurface_from_results(filename_results, i)
    print(f"  Candidate {i}: {ms.geometry.summary()}, MAE = {all_metasurfaces[i][1]:.6e}")

print("\n" + "="*70)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*70)
