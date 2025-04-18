#include <LLECUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>
#include <nvrtc.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "systems.cuh"
namespace LLE_constants{
	// Константы CUDA для GPU
__constant__ double d_tMax;
__constant__ double d_transTime;
__constant__ double d_h;

__constant__ int d_size_linspace_A;
__constant__ int d_size_linspace_B;

__constant__ int d_amountOfTransPoints;
__constant__ int d_amountOfNTPoints;
__constant__ int d_amountOfAllpoints;
__constant__ int d_amountOfCalcBlocks;

__constant__ int d_Nt_steps; 

__constant__ int d_paramsSize;
__constant__ int d_XSize;

__constant__ int d_idxParamA;
__constant__ int d_idxParamB;
__constant__ double d_eps;

__device__ int d_progress; 

__constant__ double d_h1;
__constant__ double d_h2;


__device__ void loopCalculateDiscreteModel(double *X, const double *a,
                                                    const int amountOfIterations)
{


    #pragma unroll
    for (int i = 0; i < amountOfIterations; ++i)
    {
		calculateDiscreteModel(X, a, d_h);
    }

}

__device__ void calculateDiscreteModel(double *X, const double *a, const double h)
{
	CALC_DISCRETE_MODEL(X, a, h);
}

__global__ void calculateTransTime(
    double* X,
    double* params,
    const double* paramLinspaceA,
    const double* paramLinspaceB,
    double* semi_result
) {
    extern __shared__ double sh_mem[];

    const int idx_a = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx_b = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx_a >= d_size_linspace_A || idx_b >= d_size_linspace_B) return;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_size_per_thread = 2 * d_XSize + d_paramsSize; 

    double* my_sh_X = &sh_mem[thread_id * total_size_per_thread];
    double* my_sh_params = &sh_mem[thread_id * total_size_per_thread + d_XSize];
    double* my_sh_perturbated_X = &sh_mem[thread_id * total_size_per_thread + d_XSize + d_paramsSize];

    for (int i = 0; i < d_XSize; ++i) {
        my_sh_X[i] = X[i];
    }
    for (int i = 0; i < d_paramsSize; ++i) {
        my_sh_params[i] = params[i];
    }
    my_sh_params[d_idxParamA] = paramLinspaceA[idx_a];
    my_sh_params[d_idxParamB] = paramLinspaceB[idx_b];

    // Начальное вычисление модели
    loopCalculateDiscreteModel(my_sh_X, my_sh_params, d_amountOfTransPoints);

	size_t seed = idx_a + idx_b * d_size_linspace_A;
    curandState_t state;

	curand_init(seed, 0, 0, &state);

    double norm_factor = 0.0f;
    for (int i = 0; i < d_XSize; ++i) {
        double z = curand_uniform(&state) - 0.5f;
        norm_factor = __fmaf_rn(z, z, norm_factor);
    }
    norm_factor = __fmul_rn(rsqrtf(norm_factor), norm_factor);
    for (int i = 0; i < d_XSize; ++i) {
        double z = __fdiv_rn(curand_uniform(&state) - 0.5f, norm_factor);
        my_sh_perturbated_X[i] = __fmaf_rn(z, (double)d_eps, my_sh_X[i]);
    }

    double* res_sh_X = &semi_result[(idx_a * d_size_linspace_A + idx_b ) * total_size_per_thread];
    for (int i = 0; i < d_XSize; ++i) {
        res_sh_X[i] = my_sh_X[i];                 
        res_sh_X[i + d_XSize + d_paramsSize] = my_sh_perturbated_X[i]; 
    }
    for (int i = 0; i < d_paramsSize; ++i) {
        res_sh_X[i + d_XSize] = my_sh_params[i];    
    }
}

__global__ void calculateSystem(
    double* X,  // Оставлено для совместимости, не используется
    double* params,
    const double* paramLinspaceA,
    const double* paramLinspaceB,
    double* semi_result,
    double** result
) {
    extern __shared__ double sh_mem[];

    const int idx_a = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx_b = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx_a >= d_size_linspace_A || idx_b >= d_size_linspace_B) return;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_size_per_thread = 2 * d_XSize + d_paramsSize;

    double* my_sh_X = &sh_mem[thread_id * total_size_per_thread];
    double* my_sh_params = &sh_mem[thread_id * total_size_per_thread + d_XSize];
    double* my_sh_perturbated_X = &sh_mem[thread_id * total_size_per_thread + d_XSize + d_paramsSize];

    double* res_sh_X = &semi_result[(idx_a * d_size_linspace_A + idx_b ) * total_size_per_thread];
    for (int i = 0; i < d_XSize; ++i) {
        my_sh_X[i] = res_sh_X[i];                    
        my_sh_perturbated_X[i] = res_sh_X[i + d_XSize + d_paramsSize]; 
    }
    for (int i = 0; i < d_paramsSize; ++i) {
        my_sh_params[i] = res_sh_X[i + d_XSize]; 
    }

    double local_result = 0.0;
    const double inv_eps = 1.0 / d_eps;

    for (int i = 0; i <= d_amountOfCalcBlocks; ++i) {
        loopCalculateDiscreteModel(my_sh_X, my_sh_params, d_amountOfNTPoints);
        loopCalculateDiscreteModel(my_sh_perturbated_X, my_sh_params, d_amountOfNTPoints);

        // Расчет расстояния
        double distance = 0.0;
        for (int l = 0; l < d_XSize; ++l) {
            double diff = (my_sh_X[l] - my_sh_perturbated_X[l]);
            distance += diff * diff;
        }
        distance = sqrt(distance)/d_eps;
        local_result += __logf(distance);

        for (int j = 0; j < d_XSize; ++j) {
            my_sh_perturbated_X[j] = my_sh_X[j] - ((my_sh_X[j] - my_sh_perturbated_X[j]) / distance);
        }
    }
	double temp = local_result / d_tMax;
    atomicAdd(&result[idx_a][idx_b], temp);
}

    
 


__host__ double* linspace(double start, double end, int num)
{
    // Allocate memory for num doubles
    double* result = new double[num];

    // Handle edge cases
    if (num < 0)
    {
        delete[] result;  // Clean up before throwing
        throw std::invalid_argument("received negative number of points");
    }
    if (num == 0)
    {
        return result;  // Return empty array
    }
    if (num == 1)
    {
        result[0] = start;  // Assign single value
        return result;
    }

    // Calculate step size
    double step = (end - start) / (num - 1);

    // Fill the array
    for (int i = 0; i < num; ++i)
    {
        result[i] = start + i * step;
    }

    return result;
}

__host__ void LLE2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	std::string OUT_FILE_PATH)
{
	// Базовые параметры для расчетов
	size_t amountOfNT_points = NT / h;
	int amountOfNTPoints = static_cast<int>(NT / h);
	int amountOfTransPoints= static_cast<int>(transientTime / h);
	int amountOfAllPoints= static_cast<int>(tMax / h);
	int amount_of_calc_blocks = static_cast<int>(amountOfAllPoints/amountOfNTPoints) + 1;

	// Создаем линейные пространства из диапазонов
	double* linspaceA = linspace(ranges[0], ranges[1], nPts);
	double* linspaceB = linspace(ranges[2], ranges[3], nPts);

	size_t freeMemory;
	size_t totalMemory;
	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	
	size_t maxPointsPerSegment = (freeMemory / sizeof(double)) / 4; 
	
	size_t segmentSize = std::min(maxPointsPerSegment, static_cast<size_t>(nPts * nPts));
	
	size_t numSegments = (nPts * nPts + segmentSize - 1) / segmentSize;
	
	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);
	
	if (!outFileStream.is_open()) {
		std::cerr << "Output file open error: " << OUT_FILE_PATH << std::endl;
		exit(1);
	}
	outFileStream << ranges[0] << " " << ranges[1] << "\n";
	outFileStream << ranges[2] << " " << ranges[3] << "\n";
	
	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamA, &indicesOfMutVars[0], sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamB, &indicesOfMutVars[1], sizeof(int)));
	
	int size_A = nPts;
	int size_B = nPts;
	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_A, &size_A, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_B, &size_B, sizeof(int)));
	
	gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_transTime, &transientTime, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
	
	int NT_steps = static_cast<int>(NT/h);
	gpuErrorCheck(cudaMemcpyToSymbol(d_Nt_steps, &NT_steps, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_paramsSize, &amountOfValues, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_XSize, &amountOfInitialConditions, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalcBlocks, &amount_of_calc_blocks, sizeof(int)));
	
	double h_h1 = values[0] * h;
	double h_h2 = (1 - values[0]) * h;
	gpuErrorCheck(cudaMemcpyToSymbol(d_h1, &h_h1, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_h2, &h_h2, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_eps, &eps, sizeof(double)));
	
	int stringCounter = 0;
	
	for (size_t segmentIdx = 0; segmentIdx < numSegments; ++segmentIdx) {
		size_t currentSegmentSize = (segmentIdx == numSegments - 1) ? 
			(nPts * nPts) - (segmentSize * segmentIdx) : segmentSize;
		
		size_t segmentSizeA = std::min(static_cast<size_t>(nPts), currentSegmentSize);
		size_t segmentSizeB = std::min(static_cast<size_t>(nPts), (currentSegmentSize + segmentSizeA - 1) / segmentSizeA);
		
		size_t offsetA = (segmentIdx * segmentSize) / nPts;
		size_t offsetB = (segmentIdx * segmentSize) % nPts;
		
		double** d_result;
		double** h_result_temp = new double*[segmentSizeA];
		
		gpuErrorCheck(cudaMalloc(&d_result, segmentSizeA * sizeof(double*)));
		
		for (size_t i = 0; i < segmentSizeA; ++i) {
			gpuErrorCheck(cudaMalloc(&h_result_temp[i], segmentSizeB * sizeof(double)));
			
			double zero = 0.0;
			for (size_t j = 0; j < segmentSizeB; ++j) {
				gpuErrorCheck(cudaMemcpy(h_result_temp[i] + j, &zero, sizeof(double), cudaMemcpyHostToDevice));
			}
		}
		
		gpuErrorCheck(cudaMemcpy(d_result, h_result_temp, segmentSizeA * sizeof(double*), cudaMemcpyHostToDevice));
		
		// Выделяем память для промежуточных результатов
		double* d_semi_result;
		size_t semi_result_size = segmentSizeA * segmentSizeB * (2 * amountOfInitialConditions + amountOfValues) * sizeof(double);
		
		gpuErrorCheck(cudaMalloc(&d_semi_result, semi_result_size));
		
		// Выделяем память для линейных пространств на GPU
		double* d_linspaceA;
		double* d_linspaceB;
		
		gpuErrorCheck(cudaMalloc(&d_linspaceA, segmentSizeA * sizeof(double)));
		gpuErrorCheck(cudaMalloc(&d_linspaceB, segmentSizeB * sizeof(double)));
		
		// Копируем соответствующие части линейных пространств
		gpuErrorCheck(cudaMemcpy(d_linspaceA, linspaceA + offsetA, segmentSizeA * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_linspaceB, linspaceB + offsetB, segmentSizeB * sizeof(double), cudaMemcpyHostToDevice));
		
		// Выделяем память для начальных условий и параметров
		double* d_initialConditions;
		double* d_values;
		
		gpuErrorCheck(cudaMalloc(&d_initialConditions, amountOfInitialConditions * sizeof(double)));
		gpuErrorCheck(cudaMalloc(&d_values, amountOfValues * sizeof(double)));
		
		gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyHostToDevice));
		
		// Определяем размеры блоков и сетки для запуска ядер
		int max_threads_x = 16;
		int max_threads_y = 16;
		
		int gridSizeX = (segmentSizeA + max_threads_x - 1) / max_threads_x;
		int gridSizeY = (segmentSizeB + max_threads_y - 1) / max_threads_y;
		
		dim3 threadsPerBlock(max_threads_x, max_threads_y, 1);
		dim3 blocksPerGrid(gridSizeX, gridSizeY, 1);
		
		// Вычисляем размер разделяемой памяти
		size_t sharedMemSize = (max_threads_x * max_threads_y) * (2 * amountOfInitialConditions + amountOfValues) * sizeof(double);
		
		// Вызываем первое ядро - расчет переходного процесса
		calculateTransTime<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
			d_initialConditions,
			d_values,
			d_linspaceA,
			d_linspaceB,
			d_semi_result
		);
		
		gpuErrorCheck(cudaDeviceSynchronize());
		gpuErrorCheck(cudaPeekAtLastError());
		
		// Вызываем второе ядро - основной расчет
		calculateSystem<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
			d_initialConditions,
			d_values,
			d_linspaceA,
			d_linspaceB,
			d_semi_result,
			d_result
		);
		
		gpuErrorCheck(cudaDeviceSynchronize());
		gpuErrorCheck(cudaPeekAtLastError());
		
		// Копируем результаты на хост
		double** h_result = new double*[segmentSizeA];
		
		for (size_t i = 0; i < segmentSizeA; ++i) {
			h_result[i] = new double[segmentSizeB];
			gpuErrorCheck(cudaMemcpy(h_result[i], h_result_temp[i], segmentSizeB * sizeof(double), cudaMemcpyDeviceToHost));
		}
		
		// Записываем результаты в файл - изменен порядок циклов для отражения по диагонали
		for (size_t j = 0; j < segmentSizeB; ++j) {
			for (size_t i = 0; i < segmentSizeA; ++i) {
				if (stringCounter != 0) {
					outFileStream << ", ";
				}
				
				if (stringCounter == nPts) {
					outFileStream << "\n";
					stringCounter = 0;
				}
				
				outFileStream << (std::isnan(h_result[i][j]) ? 0 : h_result[i][j]);
				++stringCounter;
			}
		}
		
		// Освобождаем память
		for (size_t i = 0; i < segmentSizeA; ++i) {
			delete[] h_result[i];
			gpuErrorCheck(cudaFree(h_result_temp[i]));
		}
		
		delete[] h_result;
		delete[] h_result_temp;
		
		gpuErrorCheck(cudaFree(d_result));
		gpuErrorCheck(cudaFree(d_semi_result));
		gpuErrorCheck(cudaFree(d_linspaceA));
		gpuErrorCheck(cudaFree(d_linspaceB));
		gpuErrorCheck(cudaFree(d_initialConditions));
		gpuErrorCheck(cudaFree(d_values));
		
		// Печатаем прогресс
		printf("Progress: %.2f%%\n", 100.0f * static_cast<double>(segmentIdx + 1) / static_cast<double>(numSegments));
	}
	
	// Закрываем файл
	outFileStream.close();
	
	// Освобождаем память
	delete[] linspaceA;
	delete[] linspaceB;
	
	printf("LLE2D calculation completed\n");
}

} //LLE_constants