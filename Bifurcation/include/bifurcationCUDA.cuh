#ifndef BIFURCATION_CUDA_CUH
#define BIFURCATION_CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

namespace Bifurcation_constants {
    __constant__ double d_tMax;
    __constant__ double d_transTime;
    __constant__ double d_h;
    
    __constant__ int d_size_linspace_A;
    __constant__ int d_nPts;
    
    __constant__ int d_amountOfTransPoints;
    __constant__ int d_amountOfNTPoints;
    __constant__ int d_amountOfAllpoints;
    
    __constant__ int d_paramsSize;
    __constant__ int d_initCondSize;
    
    __constant__ int d_idxParamA;
    __constant__ int d_writableVar;
    __constant__ double d_maxValue;
    __constant__ int d_preScaller;

    __device__ void calculateDiscreteModel(double *x, const double *a, const double h);

    __global__ void calculateBifurcation(
        double* initialConditions,
        double* params,
        const double* paramLinspaceA,
        double* result
    );

    __host__ void bifurcation1D(
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
}

#endif // BIFURCATION_CUDA_CUH
