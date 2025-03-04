#include <LLECUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>
#include <nvrtc.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
namespace LLE_constants{
	
__device__ void calculateDiscreteModel(double *X, const double *a, const double h)
{
    X[0] += d_h1 * (-a[6] * X[1]);
    X[1] += d_h1 * (a[6] * X[0] + a[1] * X[2]);
    X[2] += d_h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1])); // Убрали cos_term

    // Вычисление общего коэффициента для второй фазы
    double inv_den = 1.0 / (1.0 + a[3] * d_h2);

    // Обновления второй фазы
    X[2] = fma(d_h2, (a[2] + a[4] * cos(a[5] * X[1])), X[2]) * inv_den;
    X[1] += d_h2 * (a[6] * X[0] + a[1] * X[2]);
    X[0] += d_h2 * (-a[6] * X[1]);
}

__device__ bool loopCalculateDiscreteModel(double *x, const double *params,
                                                    const int amountOfIterations)
{
    for (int i = 0; i < amountOfIterations; ++i)
    {

        calculateDiscreteModel(x, params, d_h);
    }
    return true;
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
    const int total_size_per_thread = 2 * d_XSize + d_paramsSize;  // Полное состояние

    // Указатели в общей памяти
    double* my_sh_X = &sh_mem[thread_id * total_size_per_thread];
    double* my_sh_params = &sh_mem[thread_id * total_size_per_thread + d_XSize];
    double* my_sh_perturbated_X = &sh_mem[thread_id * total_size_per_thread + d_XSize + d_paramsSize];

    // Инициализация данных
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

    // Генерация perturbated_X
    curandState_t state;
    curand_init(idx_a + idx_b * d_size_linspace_A, 0, 0, &state);

    double norm_factor = 0.0;
    for (int i = 0; i < d_XSize; ++i) {
        double z = curand_uniform(&state) - 0.5;
        norm_factor += z * z;
    }
    norm_factor = sqrt(norm_factor);

    for (int i = 0; i < d_XSize; ++i) {
        double z = (curand_uniform(&state) - 0.5) / norm_factor;
        my_sh_perturbated_X[i] = my_sh_X[i] + z * d_eps;
    }

    // Сохранение полного состояния в глобальную память
    double* res_sh_X = &semi_result[(idx_a * d_size_linspace_A + idx_b ) * total_size_per_thread];
    for (int i = 0; i < d_XSize; ++i) {
        res_sh_X[i] = my_sh_X[i];                    // X
        res_sh_X[i + d_XSize + d_paramsSize] = my_sh_perturbated_X[i];  // perturbated_X
    }
    for (int i = 0; i < d_paramsSize; ++i) {
        res_sh_X[i + d_XSize] = my_sh_params[i];     // params
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

    // Указатели в общей памяти
    double* my_sh_X = &sh_mem[thread_id * total_size_per_thread];
    double* my_sh_params = &sh_mem[thread_id * total_size_per_thread + d_XSize];
    double* my_sh_perturbated_X = &sh_mem[thread_id * total_size_per_thread + d_XSize + d_paramsSize];

    // Загрузка полного состояния из глобальной памяти
    double* res_sh_X = &semi_result[(idx_a * d_size_linspace_A + idx_b ) * total_size_per_thread];
    for (int i = 0; i < d_XSize; ++i) {
        my_sh_X[i] = res_sh_X[i];                    // X
        my_sh_perturbated_X[i] = res_sh_X[i + d_XSize + d_paramsSize];  // perturbated_X
    }
    for (int i = 0; i < d_paramsSize; ++i) {
        my_sh_params[i] = res_sh_X[i + d_XSize];     // params
    }

    // Основной цикл вычислений
    double local_result = 0.0;
    const double inv_eps = 1.0 / d_eps;

    for (int i = 0; i <= d_amountOfCalcBlocks; ++i) {
        loopCalculateDiscreteModel(my_sh_X, my_sh_params, d_amountOfNTPoints);
        loopCalculateDiscreteModel(my_sh_perturbated_X, my_sh_params, d_amountOfNTPoints);

        // Расчет расстояния
        double distance = 0.0;
        for (int l = 0; l < d_XSize; ++l) {
            double diff = (my_sh_X[l] - my_sh_perturbated_X[l]) * inv_eps;
            distance += diff * diff;
        }
        distance = sqrt(distance);
        local_result += __logf(distance);

        // Обновление perturbated_X
        for (int j = 0; j < d_XSize; ++j) {
            my_sh_perturbated_X[j] = my_sh_X[j] - ((my_sh_X[j] - my_sh_perturbated_X[j]) / distance);
        }
    }

    // Запись результата
    atomicAdd(&result[idx_a][idx_b], local_result);
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
	const double h,
	const double eps,
	const double transientTime,
	const double* initialConditions,
	const int amount_init,
	const double* params,
	const int amount_params,
	const double* linspaceA_params,
	const double* linspaceB_params,
	const int* indicesOfMutVars,
	std::string		OUT_FILE_PATH)
{

	double* linspaceA = linspace(linspaceA_params[0], linspaceA_params[1], linspaceA_params[2]);
	double* linspaceB = linspace(linspaceB_params[0], linspaceB_params[1], linspaceB_params[2]);

	int amountOfNTPoints = static_cast<int>(NT / h);
	int amountOfTransPoints= static_cast<int>(transientTime / h);
	int amountOfAllPoints= static_cast<int>(tMax / h);

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	const int size_A =  static_cast<int>(linspaceA_params[2]);
	const int size_B =  static_cast<int>(linspaceB_params[2]);
	int NT_steps = static_cast<int>(NT/h);

	int amount_of_calc_blocks = static_cast<int>(amountOfAllPoints/amountOfNTPoints) + 1;

	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamA, &indicesOfMutVars[0], sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamB, &indicesOfMutVars[1], sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_A, &size_A, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_B, &size_B, sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_transTime, &transientTime, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_Nt_steps, &NT_steps, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_paramsSize, &amount_params, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_XSize, &amount_init, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalcBlocks, &amount_of_calc_blocks, sizeof(int)));

    double h_h1 = params[0] * h;
    double h_h2 = (1 - params[0]) * h;
    gpuErrorCheck(cudaMemcpyToSymbol(d_h1, &h_h1, sizeof(double)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_h2, &h_h2, sizeof(double)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_eps, &eps, sizeof(double)));


    int max_threads_y =  16;  
    int max_threads_x =  16;  

    int gridSizeY = (size_B - 1) / max_threads_y;
    int gridSizeX =  (size_A - 1) / max_threads_x;


    // Define thread block and grid dimensions
    dim3 threadsPerBlock(max_threads_x, max_threads_y, 1);  // e.g., (16, 8, 1)
    dim3 blocksPerGrid(gridSizeX, gridSizeY, 1);

    size_t sharedMemSizeTrans = (max_threads_x * max_threads_y) * (amount_init + amount_params) * sizeof(double);
    size_t sharedMemSize = (max_threads_x * max_threads_y) * (2 * amount_init + amount_params) * sizeof(double);  // For thread pairs
    printf("Total shared memory: %zu bytes\n", sharedMemSize);


    double** d_result;
    double** h_result_temp = new double*[size_A];
    gpuErrorCheck(cudaMalloc(&d_result, size_A * sizeof(double*)));
    for (int i = 0; i < size_A; ++i) {
        gpuErrorCheck(cudaMalloc(&h_result_temp[i], size_B * sizeof(double)));
        double zero = 0.0;

        for (int j = 0; j < size_B; ++j) {
            gpuErrorCheck(cudaMemcpy(h_result_temp[i] + j, &zero, sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    gpuErrorCheck(cudaMemcpy(d_result, h_result_temp, size_A * sizeof(double*), cudaMemcpyHostToDevice));


    double* d_semi_result;
    gpuErrorCheck(cudaMalloc(&d_semi_result, size_A * size_B * (2 * amount_init + amount_params) * sizeof(double)));

    double* d_paramLinspaceA;
    double* d_paramLinspaceB;
    double* d_X;
    double* d_params;

    gpuErrorCheck(cudaMalloc(&d_X, amount_init * sizeof(double)));
    gpuErrorCheck(cudaMalloc(&d_params, amount_params * sizeof(double)));
    gpuErrorCheck(cudaMalloc(&d_paramLinspaceA, size_A * sizeof(double)));
    gpuErrorCheck(cudaMalloc(&d_paramLinspaceB, size_B * sizeof(double)));

    gpuErrorCheck(cudaMemcpy(d_X, initialConditions, amount_init * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_params, params, amount_params * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_paramLinspaceA, linspaceA, size_A * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_paramLinspaceB, linspaceB, size_B * sizeof(double), cudaMemcpyHostToDevice));

    // Первый вызов: расчет trans_time и perturbated_X
    LLE_constants::calculateTransTime<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_X,
        d_params,
        d_paramLinspaceA,
        d_paramLinspaceB,
        d_semi_result
    );
    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaPeekAtLastError());

    // Второй вызов: расчет системы
    LLE_constants::calculateSystem<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_X,
        d_params,
        d_paramLinspaceA,
        d_paramLinspaceB,
        d_semi_result,
        d_result
    );
    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaPeekAtLastError());

    printf("Calculations ended\n");

    double** h_result = new double*[size_A];
    for (int i = 0; i < size_A; ++i) {
        h_result[i] = new double[size_B];
        gpuErrorCheck(cudaMemcpy(h_result[i], h_result_temp[i], size_B * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::ofstream outFileStream(OUT_FILE_PATH);
    if (outFileStream.is_open()) {
        for (int i = 0; i < size_A; ++i) {
            for (int j = 0; j < size_B; ++j) {
                if (j > 0) outFileStream << ", ";
                outFileStream << (std::isnan(h_result[i][j]) ? 0 : h_result[i][j]);
            }
            outFileStream << "\n";
        }
        outFileStream.close();
    } else {
        std::cerr << "Output file open error: " << OUT_FILE_PATH << std::endl;
        exit(1);
    }

	delete[] linspaceA;
	delete[] linspaceB;
}

} //LLE_constants