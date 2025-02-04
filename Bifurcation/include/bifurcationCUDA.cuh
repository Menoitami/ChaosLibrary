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


/**
 * @brief Вычисляет энтропию по заданным данным в гистограмме.
 * 
 * @param bins Массив значений гистограммы.
 * @param binSize Размер массива `bins`.
 * @param sum Сумма всех элементов в массиве `bins`.
 * 
 * @return double Значение энтропии.
 */
__device__ double calculateEntropy(double *bins, int binSize, const int sum);


/**
 * @brief Рассчитывает гистограмму на основе изменения значения координаты системы во времени.
 * 
 * @param X Массив из X_size элементов, представляющий текущее состояние системы.
 * @param param Массив параметров модели.
 * @param sum Переменная для хранения общей суммы всех попаданий в bins. Изменяется на месте.
 * @param bins Массив для хранения значений гистограммы.
 */
__device__ void CalculateHistogram(
    double *X, const double *param,
    int &sum, double *bins);



/**
 * @brief Рассчитывает гистограмму и энтропию для системы в трехмерном параметрическом пространстве на GPU.
 * 
 * @param X Указатель на массив начальных значений координат системы (размер XSize).
 * @param params Указатель на массив параметров системы (размер paramsSize).
 * @param paramLinspaceA Указатель на массив значений первого параметра (размер histEntropySizeRow).
 * @param paramLinspaceB Указатель на массив значений второго параметра (размер histEntropySizeCol).
 * @param histEntropy Указатель на массив для записи нормализованных значений энтропии (размер histEntropySize).
 * @param bins_global Указатель на глобальную память для хранения гистограмм (размер histEntropySize * binSize).
 * 
 * ### Алгоритм:
 * 
 * 1. Вычисление индекса потока  idx и привязка к точке параметрического пространства.
 * 
 * 2. Копирование входных данных (`X` и `params`) в локальную память для ускорения вычислений.
 * 
 * 3. Настройка параметров системы, используя сетки paramLinspaceA и paramLinspaceB.
 * 
 * 4. Стабилизация системы методом `loopCalculateDiscreteModel` на времени d_transTime \f$.
 * 
 * 5. Построение гистограммы методом `CalculateHistogram`.
 * 
 * 6. Вычисление энтропии методом `calculateEntropy` и сохранение результата.
 */
__global__ void calculateHistEntropyCuda3D(const double *X,
                                           const double *params,
                                           const double *paramLinspaceA,
                                           const double *paramLinspaceB,
                                           double *histEntropy,
                                           double *bins_global);


/**
 * @brief Функция для вычисления энтропии по 2 параметрам с использованием CUDA.
 * 
 * Метод выполняет вычисления энтропии гистограммы для данных, представленных в виде двух массивов параметров.
 * Он распределяет данные на несколько блоков и выполняет параллельные вычисления на GPU для оптимизации времени.
 * 
 * @param transTime Время транзиенты.
 * @param tMax Время работы системы.
 * @param h Шаг.
 * @param X координаты системы.
 * @param coord Индекс координаты по которой будет строится энтропия.
 * @param params Вектор параметров, где params[0] - коэффициент симметрии.
 * @param paramNumberA Индекс первого параметра для вычислений в params.
 * @param paramNumberB Индекс второго параметра для вычислений в params.
 * @param startBin Начало диапазона для столбцов гистограмм.
 * @param endBin Конец диапазона для столбцов гистограмм.
 * @param stepBin Шаг столбцов гистограмм.
 * @param linspaceStartA Начало диапазона для массива A.
 * @param linspaceEndA Конец диапазона для массива A.
 * @param linspaceNumA Количество точек в массиве A.
 * @param linspaceStartB Начало диапазона для массива B.
 * @param linspaceEndB Конец диапазона для массива B.
 * @param linspaceNumB Количество точек в массиве B.
 * 
 * @return Вектор двухмерный массив значений.
 */
__host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime, const double tMax, const double h,
    const std::vector<double> &X, const int coord,
    const std::vector<double> &params, const int paramNumberA, const int paramNumberB,
    const double startBin, const double endBin, const double stepBin,
    double linspaceStartA, double linspaceEndA, int linspaceNumA, double linspaceStartB, double linspaceEndB, int linspaceNumB);



/**
 * @brief Функция для вычисления энтропии по 1 параметру с использованием CUDA.
 * 
 * 
 * @param transTime Время транзиенты.
 * @param tMax Время работы системы.
 * @param h Шаг.
 * @param X координаты системы.
 * @param coord Индекс координаты по которой будет строится энтропия.
 * @param params Вектор параметров, где params[0] - коэффициент симметрии.
 * @param paramNumberA Индекс параметра для вычислений в params.
 * @param startBin Начало диапазона для столбцов гистограмм.
 * @param endBin Конец диапазона для столбцов гистограмм.
 * @param stepBin Шаг столбцов гистограмм.
 * @param linspaceStartA Начало диапазона для массива параметра.
 * @param linspaceEndA Конец диапазона для массива параметра.
 * @param linspaceNumA Количество точек в массиве параметра.
 * 
 * @return std::vector<double> Возвращает энтропию по 1 параметру, вычисленную для данных.
 */
__host__ std::vector<double> histEntropyCUDA2D(
    const double transTime, const double tMax, const double h,
    const std::vector<double> &X, const int coord,
    std::vector<double> &params, const int paramNumberA,
    const double startBin, const double endBin, const double stepBin,
    double linspaceStartA, double linspaceEndA, int linspaceNumA);



#endif // BIFURCATION_CUDA_CUH