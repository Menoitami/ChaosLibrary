#ifndef BIFURCATION_HOST_H
#define BIFURCATION_HOST_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bifurcationCUDA.cuh"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

namespace Bifurcation {
    /**
     * Функция для расчета бифуркационной диаграммы в 1D
     */
    void bifurcation1D(
        const double    tMax,                           // Время моделирования системы
        const int       nPts,                           // Количество разбиений
        const double    h,                              // Шаг интегрирования
        const int       amountOfInitialConditions,      // Количество начальных условий (уравнений в системе)
        const double*   initialConditions,              // Массив с начальными условиями
        const double*   ranges,                         // Диапазон изменения параметров
        const int*      indicesOfMutVars,               // Индекс изменяемого параметра в массиве values
        const int       writableVar,                    // Индекс параметра, по которому будем строить диаграмму
        const double    maxValue,                       // Максимальное значение (по модулю), выше которого система считается "расходящейся"
        const double    transientTime,                  // Время, которое будет промоделировано перед расчетом диаграммы
        const double*   values,                         // Параметры
        const int       amountOfValues,                 // Количество параметров
        const int       preScaller,                     // Множитель, который уменьшает время и объем расчетов
        std::string     OUT_FILE_PATH);                 // Путь к выходному файлу
        
    /**
     * Функция для построения двумерной бифуркационной диаграммы (DBSCAN)
     */
    void bifurcation2D(
        const double tMax,                              // Время моделирования системы
        const int nPts,                                // Разрешение диаграммы
        const double h,                                // Шаг интегрирования
        const int amountOfInitialConditions,          // Количество начальных условий (уравнений в системе)
        const double* initialConditions,               // Массив с начальными условиями
        const double* ranges,                          // Диапазоны изменения параметров
        const int* indicesOfMutVars,                   // Индексы изменяемых параметров
        const int writableVar,                         // Индекс уравнения, по которому будем строить диаграмму
        const double maxValue,                         // Максимальное значение (по модулю), выше которого система считается "расходящейся"
        const double transientTime,                    // Время, которое будет промоделировано перед расчетом диаграммы
        const double* values,                          // Параметры
        const int amountOfValues,                      // Количество параметров
        const int preScaller,                          // Множитель, который уменьшает время и объем расчетов
        const double eps,                              // Эпсилон для алгоритма DBSCAN
        std::string OUT_FILE_PATH);                    // Путь к выходному файлу
}

namespace Bifurcation_constants {
    /**
     * Функция для построения двумерной бифуркационной диаграммы (DBSCAN)
     */
    __host__ void bifurcation2D(
        const double tMax,                              // Время моделирования системы
        const int nPts,                                // Разрешение диаграммы
        const double h,                                // Шаг интегрирования
        const int amountOfInitialConditions,          // Количество начальных условий (уравнений в системе)
        const double* initialConditions,               // Массив с начальными условиями
        const double* ranges,                          // Диапазоны изменения параметров
        const int* indicesOfMutVars,                   // Индексы изменяемых параметров
        const int writableVar,                         // Индекс уравнения, по которому будем строить диаграмму
        const double maxValue,                         // Максимальное значение (по модулю), выше которого система считается "расходящейся"
        const double transientTime,                    // Время, которое будет промоделировано перед расчетом диаграммы
        const double* values,                          // Параметры
        const int amountOfValues,                      // Количество параметров
        const int preScaller,                          // Множитель, который уменьшает время и объем расчетов
        const double eps,                              // Эпсилон для алгоритма DBSCAN
        std::string OUT_FILE_PATH);                    // Путь к выходному файлу
}

#endif // BIFURCATION_HOST_H