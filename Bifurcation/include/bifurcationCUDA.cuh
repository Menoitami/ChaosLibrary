#ifndef BIFURCATION_CUDA_CUH
#define BIFURCATION_CUDA_CUH


#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <thread>
#include <nvrtc.h>



#define CHECK_CUDA_ERROR(call)                                              \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " in file " << __FILE__ << " at line " << __LINE__ \
                      << std::endl;                                         \
            exit(err);                                                      \
        }                                                                   \
    }



/**
 * @brief Создает равномерно распределенные точки в заданном интервале.
 * 
 * @param start Начало интервала.
 * @param end Конец интервала.
 * @param num Количество точек (должно быть >= 0).
 * 
 * @return std::vector<double> Вектор с равномерно распределенными точками.
 * 
 * @throws std::invalid_argument Если `num < 0`.
 */
__host__ std::vector<double> linspaceNum(double start, double end, int num);

/**
 * @brief Вычисляет дискретную модель динамической системы.
 * 
 * @param x Массив из X_size элементов, представляющий состояние системы. Массив изменяется на месте.
 * @param a Массив из param_size параметров, определяющих характеристики системы.
 * @param h Шаг интегрирования.
 * 
 * В качестве системы может быть использована любая другая, достаточно поменять реализацию
 */
__device__ __host__ void calculateDiscreteModel(double *x, const double *a, const double h);



/**
 * @brief Выполняет многократные итерации вычисления дискретной модели системы.
 * 
 * @param x Массив из x_size элементов, представляющий текущее состояние системы. Массив изменяется на месте.
 * @param values Массив параметров модели.
 * @param h Шаг интегрирования.
 * @param amountOfIterations Количество итераций основного цикла.
 * @param preScaller Количество итераций внутреннего цикла перед записью данных.
 * @param writableVar Индекс переменной  x , значение которой записывается в массив `data`.
 * @param maxValue Максимальное допустимое значение для переменной  x[writableVar] . 
 * @param data Массив для записи значений переменной  x[writableVar]  на каждой итерации.
 * @param startDataIndex Начальный индекс в массиве `data` для записи значений.
 * @param writeStep Шаг записи в массиве `data` между значениями.
 * 
 * @return `true`, если максимальное значение не превышено; `false` в противном случае.
 */
__device__ __host__ bool loopCalculateDiscreteModel(double *x, const double *values, const double h,
                                                    const int amountOfIterations, const int preScaller,
                                                    int writableVar, const double maxValue, double *data, const int startDataIndex, const int writeStep);




__global__ void bifurcatonKernel1D(const double *X, const double *params, const double *paramLinspaceA,
                                    double* bifurVec){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= d_bifurSize) return;

    double X_locals[32];
    double params_local[32];

    memcpy(X_locals, X, d_XSize * sizeof(double));
    memcpy(params_local, params, d_paramsSize * sizeof(double));

    params_local[d_paramNumber] = paramLinspaceA[idx];

    
    loopCalculateDiscreteModel(X_locals, params_local, d_h,
                               static_cast<int>(d_transTime / d_h),
                               0, 0, 0, nullptr, 0, 0);


}

__device__ void findPeaks(double *X, const double *param, double *bifurCell){

    int iterations = static_cast<int>(d_tMax / d_h);
    double last = X[d_coord];    
    bool lastBigger = false;
    



    }


#endif // BIFURCATION_CUDA_CUH