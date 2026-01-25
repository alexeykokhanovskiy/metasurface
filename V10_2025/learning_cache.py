from functools import lru_cache
import time

@lru_cache(maxsize=128)  # maxsize — максимум элементов в кэше, None — без ограничения
def expensive_function(x):
    print(f"Вычисляю для {x}...")  # Чтобы увидеть, когда функция действительно выполняется
    time.sleep(2)  # Имитация "дорогой" операции
    return x * x

# Первый вызов — вычисление
print(expensive_function(5))  # Вычисляю... → 25 (через 2 секунды)

# Повторный вызов с тем же аргументом — из кэша (мгновенно)
print(expensive_function(5))  # 25 (без задержки и без печати)

# Новый аргумент — снова вычисление
print(expensive_function(10))  # Вычисляю... → 100
print(expensive_function.cache_info())  # hits=1, misses=2, maxsize=128, currsize=2