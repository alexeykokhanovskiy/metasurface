import numpy as np
import time
import os
import matplotlib.pyplot as plt
from cmaes import CMA

import matlab.engine

eng = matlab.engine.start_matlab()


# matlab_directory_class = 'C:/Users/gavriil.romanenko/Desktop/matlab/ott_Optimizer_PY/' # путь, где находится .m файл matlab
# python_directory='C:/Users/gavriil.romanenko/Desktop/matlab/ott_Optimizer_PY/'
# #добавлении директории в матлаб движок
# eng.addpath(eng.genpath(matlab_directory_class)) # связан с OTT_optimiz_polZermike.m
# eng.addpath(eng.genpath(python_directory))

def fintess_function():
    return


# DATA_Kz = fitness_fucntion(Num_n, MACRO_Cnm) # связывается с matlab для нахождения фазовой маски и расчета Fz

if __name__ == "__main__":

    All_num_nm = [1, 3, 6, 10, 15, 21, 28]  # кол-во членов полинома для разных n = (0:6)
    Num_n = 3  # степенной показатель для расчета полиномов Цернике

    Num_Pol_Zern = All_num_nm[Num_n]  # Необходимо понять сколько весов направить в matlab
    min_value = -10  # Minimum value
    max_value = 10  # Maximum value
    MACRO_Cnm = np.zeros((Num_Pol_Zern, 1)) + np.random.randint(min_value, max_value + 1, size=Num_Pol_Zern)
    MACRO_Cnm[-1] += (
                1 - np.sum(MACRO_Cnm))  # ACRO_Cnm - массив рандомно заннах значения весовых коэф полиномов Цернике

    Max_iter = 2  # количество итераций (изменение волнового фронта для нахождения Kz)
    Pop_size = 20  # размер популяции рамках одной итерации N_iter
    Nnm = 10  # количество оптимизируемых весовых коэф полинома Цернике
    start0 = np.zeros(Num_Pol_Zern)
    sigma = 10  # Standard deviation for mutation
    bounds = np.tile([-np.inf, np.inf], (Nnm, 1))

    optimizer = CMA(mean=start0, sigma=sigma, bounds=bounds, seed=0, population_size=Pop_size)

    zernike_weights = [[] for _ in range(
        Max_iter)]  # создание списков для хранения весов при полиномах цернике и жестоксти оптического захвата
    stiffness_list = [[] for _ in range(Max_iter)]
    zernike_weights_best = [[] for _ in range(Max_iter)]
    stiffness_list_best = [[] for _ in range(Max_iter)]

    for generation in range(Max_iter):
        solutions = []  # решения одном поколении (аргумент и значение целевой функции)
        value_best = 100

        for zz in range(optimizer.population_size):
            x = optimizer.ask()
            value = fitness_fucntion(Num_n, x)
            solutions.append((x, value))
            print(f"# {generation} {value} x=", *x)

            # сохранения лучшего значения жесткости
            if zz == 1:
                x_best = x
                value_best = value
            elif value <= value_best:
                x_best = x
                value_best = value
        optimizer.tell(solutions)

        zernike_weights[generation] = x
        stiffness_list[generation] = value
        zernike_weights_best[generation] = x_best
        stiffness_list_best[generation] = value_best

        if optimizer.should_stop():
            print('STOP_OPTIMIZER')
            # popsize multiplied by 2 (or 3) before each restart.
            popsize = optimizer.population_size * 2
            mean = (np.random.rand(Nnm))
            optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
            print(f"Restart CMA-ES with popsize={popsize}")









