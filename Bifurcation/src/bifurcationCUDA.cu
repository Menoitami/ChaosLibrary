#include <bifurcationCUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>

namespace Bifurcation_constants {

__device__ void calculateDiscreteModel(double *x, const double *a, const double h) {
    // Здесь должна быть реализация функции расчета модели
    // Конкретный код зависит от модели, которую вы используете
    // Пример для простой системы:
    double x0 = x[0];
    double x1 = x[1];
    
    // Пример простой системы уравнений
    x[0] = x0 + h * (a[0] * x0 - x0 * x1);
    x[1] = x1 + h * (x0 * x1 - a[1] * x1);
}

__global__ void calculateBifurcation(
    double* initialConditions,
    double* params,
    const double* paramLinspaceA,
    double* result
) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_size_linspace_A) return;
    
    // Локальные переменные для хранения состояния
    double X[10]; // Предполагается, что у нас максимум 10 переменных состояния
    double localParams[20]; // Предполагается, что у нас максимум 20 параметров
    
    // Инициализация начальных условий
    for (int i = 0; i < d_initCondSize; i++) {
        X[i] = initialConditions[i];
    }
    
    // Инициализация параметров
    for (int i = 0; i < d_paramsSize; i++) {
        localParams[i] = params[i];
    }
    
    // Установка изменяемого параметра
    localParams[d_idxParamA] = paramLinspaceA[idx];
    
    // Расчет переходного процесса
    for (int i = 0; i < d_amountOfTransPoints; i++) {
        calculateDiscreteModel(X, localParams, d_h);
    }
    
    // Расчет бифуркационной диаграммы
    for (int i = 0; i < d_amountOfNTPoints; i++) {
        calculateDiscreteModel(X, localParams, d_h);
        
        // Записываем результат каждые preScaller шагов
        if (i % d_preScaller == 0) {
            double value = X[d_writableVar];
            
            // Проверка на расхождение системы
            if (abs(value) > d_maxValue) {
                value = 0.0;  // Или другое значение по умолчанию
            }
            
            // Запись результата
            result[idx * (d_amountOfNTPoints / d_preScaller) + (i / d_preScaller)] = value;
        }
    }
}

__host__ double* linspace(double start, double end, int num) {
    // Allocate memory for num doubles
    double* result = new double[num];
    
    // Handle edge cases
    if (num < 0) {
        delete[] result;  // Clean up before throwing
        throw std::invalid_argument("received negative number of points");
    }
    if (num == 0) {
        return result;  // Return empty array
    }
    if (num == 1) {
        result[0] = start;  // Assign single value
        return result;
    }
    
    // Calculate step size
    double step = (end - start) / (num - 1);
    
    // Fill the array
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    
    return result;
}

__host__ void bifurcation1D(
    const double    tMax,
    const int       nPts,
    const double    h,
    const int       amountOfInitialConditions,
    const double*   initialConditions,
    const double*   ranges,
    const int*      indicesOfMutVars,
    const int       writableVar,
    const double    maxValue,
    const double    transientTime,
    const double*   values,
    const int       amountOfValues,
    const int       preScaller,
    std::string     OUT_FILE_PATH
) {
    // Создаем линейное пространство для изменяемого параметра
    double* linspaceA = linspace(ranges[0], ranges[1], nPts);
    
    // Рассчитываем количество точек для переходного процесса и моделирования
    int amountOfTransPoints = static_cast<int>(transientTime / h);
    int amountOfNTPoints = static_cast<int>((tMax - transientTime) / h);
    int amountOfAllPoints = static_cast<int>(tMax / h);
    
    // Получаем информацию о доступной памяти GPU
    size_t freeMemory;
    size_t totalMemory;
    gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
    
    // Копируем константы в память устройства
    gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamA, &indicesOfMutVars[0], sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_writableVar, &writableVar, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_maxValue, &maxValue, sizeof(double)));
    
    gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_A, &nPts, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_nPts, &nPts, sizeof(int)));
    
    gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_transTime, &transientTime, sizeof(double)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    
    gpuErrorCheck(cudaMemcpyToSymbol(d_paramsSize, &amountOfValues, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_initCondSize, &amountOfInitialConditions, sizeof(int)));
    
    gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_preScaller, &preScaller, sizeof(int)));
    
    // Определяем размеры блока и сетки
    int threadsPerBlock = 256;
    int blocksPerGrid = (nPts + threadsPerBlock - 1) / threadsPerBlock;
    
    // Выделяем память для результатов
    int outputSize = nPts * (amountOfNTPoints / preScaller);
    double* d_result;
    gpuErrorCheck(cudaMalloc(&d_result, outputSize * sizeof(double)));
    
    // Выделяем память и копируем данные на устройство
    double* d_initialConditions;
    double* d_params;
    double* d_paramLinspaceA;
    
    gpuErrorCheck(cudaMalloc(&d_initialConditions, amountOfInitialConditions * sizeof(double)));
    gpuErrorCheck(cudaMalloc(&d_params, amountOfValues * sizeof(double)));
    gpuErrorCheck(cudaMalloc(&d_paramLinspaceA, nPts * sizeof(double)));
    
    gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_params, values, amountOfValues * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_paramLinspaceA, linspaceA, nPts * sizeof(double), cudaMemcpyHostToDevice));
    
    // Запускаем ядро для расчета бифуркационной диаграммы
    calculateBifurcation<<<blocksPerGrid, threadsPerBlock>>>(
        d_initialConditions,
        d_params,
        d_paramLinspaceA,
        d_result
    );
    
    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaPeekAtLastError());
    
    // Выделяем память для результатов на хосте
    double* h_result = new double[outputSize];
    gpuErrorCheck(cudaMemcpy(h_result, d_result, outputSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Записываем результаты в файл
    std::ofstream outFileStream(OUT_FILE_PATH);
    if (outFileStream.is_open()) {
        for (int i = 0; i < nPts; ++i) {
            outFileStream << linspaceA[i];
            for (int j = 0; j < amountOfNTPoints / preScaller; ++j) {
                outFileStream << ", " << h_result[i * (amountOfNTPoints / preScaller) + j];
            }
            outFileStream << "\n";
        }
        outFileStream.close();
    } else {
        std::cerr << "Output file open error: " << OUT_FILE_PATH << std::endl;
        exit(1);
    }
    
    // Освобождаем память
    delete[] linspaceA;
    delete[] h_result;
    
    gpuErrorCheck(cudaFree(d_initialConditions));
    gpuErrorCheck(cudaFree(d_params));
    gpuErrorCheck(cudaFree(d_paramLinspaceA));
    gpuErrorCheck(cudaFree(d_result));
}

} // Bifurcation_constants
