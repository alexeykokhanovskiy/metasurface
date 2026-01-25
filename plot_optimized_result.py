import h5py
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# НАСТРОЙКИ
# ===============================
H5_FILE = 'optimization_results_4params_PxWyfixed.h5'  # Путь к твоему HDF5-файлу
MATLAB_PATH = 'C:/Users/alexey/YandexDisk/Projects/metasurface_spectral_filters/signal_processing_PCM_metasurfaces/V10_2025/results/new_antennas/'

WL_MIN = 1.7
WL_MAX = 4.0
N_wl_points = 100
wl_model = np.linspace(WL_MIN, WL_MAX, N_wl_points)

# Имена параметров в том же порядке, как в MATLAB-функции r_t_calc
PARAM_NAMES = ['h_cross', 'Px', 'Wy', 'Gst_thickness']

# ===============================
# 1. ЧТЕНИЕ HDF5 И ПОИСК ЛУЧШЕГО РЕШЕНИЯ
# ===============================
print(f"Открываем файл: {H5_FILE}")

with h5py.File(H5_FILE, 'r') as f:
    results_grp = f['results']

    # Проверяем наличие атрибута с именами параметров (если сохранял)
    if 'parameter_names' in results_grp.attrs:
        saved_names = list(results_grp.attrs['parameter_names'])
        if saved_names != PARAM_NAMES:
            print("Предупреждение: порядок параметров в файле отличается от ожидаемого.")
            print(f"В файле: {saved_names}")
            print(f"Ожидается: {PARAM_NAMES}")
            # Можно переупорядочить, если нужно

    total_candidates = results_grp.attrs['total_evaluations']
    print(f"Всего кандидатов в файле: {total_candidates}")

    best_loss = np.inf
    best_params = None
    best_index = -1

    for candidate_name in results_grp.keys():
        if not candidate_name.startswith('candidate_'):
            continue
        candidate = results_grp[candidate_name]
        loss = candidate['loss'][()]
        params = candidate['parameters'][:]  # array shape (4,)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_index = int(candidate_name.split('_')[1])

    if best_params is None:
        raise ValueError("Не найдено ни одного кандидата в HDF5-файле")

print(f"\nЛучший кандидат: candidate_{best_index:05d}")
print(f"Минимальный MAE (loss): {best_loss:.6e}")
print("Оптимальные параметры:")
for name, val in zip(PARAM_NAMES, best_params):
    print(f"  {name} = {val:.6f}")

# ===============================
# 2. ЗАПУСК MATLAB И РАСЧЁТ СПЕКТРА
# ===============================
print("\nЗапуск MATLAB engine...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.pwd(), nargout=0)
eng.addpath(MATLAB_PATH, nargout=0)

# Преобразуем параметры в float для передачи
h_cross_opt = float(best_params[0])
Px_opt = float(best_params[1])
Wy_opt = float(best_params[2])
Gst_thickness_opt = float(best_params[3])

print("Расчёт спектра для оптимальных параметров...")
R_TE_matlab, T_TE_matlab = eng.r_t_calc_multi_param(
    WL_MIN, WL_MAX,
    h_cross_opt, Px_opt, Wy_opt, Gst_thickness_opt,
    nargout=2
)

# Преобразуем в numpy arrays
R_TE_opt = np.array(R_TE_matlab).flatten()
T_TE_opt = np.array(T_TE_matlab).flatten()

print("Расчёт завершён.")

# ===============================
# 3. СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ===============================
import data_processing as dp
data = np.genfromtxt('./LIPPS_sio2_polarized.csv', delimiter=',')
wl, lipss_covered = dp.crop_wl(WL_MIN, WL_MAX, data, 6)

# График
plt.figure(figsize=(8, 5))
plt.plot(wl_model, R_TE_opt, label='R (Reflection)', linewidth=2)
plt.plot(wl_model, T_TE_opt, label='T (Transmission)', linewidth=2, linestyle='--')
plt.plot(wl, lipss_covered/100, label='Experimental', linewidth=1, alpha=0.7)

plt.xlabel('Длина волны, μm')
plt.ylabel('R / T')
plt.title(f'Спектр для оптимальных параметров (MAE = {best_loss:.3e})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(WL_MIN, WL_MAX)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ===============================
# ЗАВЕРШЕНИЕ
# ===============================
eng.quit()
print("\nГотово!")