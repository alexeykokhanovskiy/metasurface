import numpy as np
from functools import lru_cache


MAX_CACHE_SIZE = None
CACHE_PRECISION = 6        # точность округления (знаков после запятой)
MAX_CACHE_SIZE = 10_000    # максимальный размер кэша (можно None для неограниченного)

@lru_cache(maxsize=MAX_CACHE_SIZE)
def _cached_simulation(params_tuple: tuple, eng) -> float:
    """
    Внутренняя функция, которая реально вызывает MATLAB.
    Аргумент params_tuple — кортеж из 4 float (уже округлённых).
    Декоратор @lru_cache автоматически кэширует результаты.
    """
    h_cross, Px, Wy, Gst_thickness = params_tuple
    try:
        signal_model_matlab, _ = eng.r_t_calc_multi_param(
            WL_MIN, WL_MAX,
            float(h_cross), float(Px), float(Wy), float(Gst_thickness),
            nargout=2
        )
        signal_model = np.array(signal_model_matlab).flatten()
        value = calc_mae_model_experimental(wl_exp, wl_model, signal_exp, signal_model)
    except Exception as e:
        print(f"Ошибка MATLAB для параметров {params_tuple}: {e}")
        value = np.inf

    return value


def evaluate_with_cache(x: np.ndarray, eng, cache_precision: int = 6) -> float:
    """
    Основная функция для использования в оптимизации.
    Принимает np.ndarray, преобразует в округлённый кортеж и вызывает кэшированную функцию.
    """
    # Округляем и превращаем в кортеж — теперь он хэшируемый
    key_tuple = tuple(round(float(v), cache_precision) for v in x)
    # Вызываем кэшированную функцию
    return _cached_simulation(key_tuple, eng)
