#include <LLECUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>
#include <nvrtc.h>
namespace LLE_constants{
	
__device__ void calculateDiscreteModel(double *X, const double *a, const double h)
{
    /*
    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);
    */
   	double h1 = a[0] * h;
	double h2 = (1 - a[0]) * h;
	X[0] = X[0] + h1 * (-a[6] * X[1]);
	X[1] = X[1] + h1 * (a[6] * X[0] + a[1] * X[2]);
	X[2] = X[2] + h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1]));

	X[2] = (X[2] + h2 * (a[2] + a[4] * cos(a[5] * X[1]))) / (1 + a[3] * h2);
	X[1] = X[1] + h2 * (a[6] * X[0] + a[1] * X[2]);
	X[0] = X[0] + h2 * (-a[6] * X[1]);
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


__global__ void calculateSystem(
    double* X,
    double* params,
	const double *paramLinspaceA,
	const double *paramLinspaceB,
	Calc_block ***calculatedBlocks
){

	const int idx_a = threadIdx.x + blockIdx.x * blockDim.x;
	const int idx_b = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx_a >= d_size_linspace_A || idx_b >= d_size_linspace_B) return;
  // printf("%d , %d , %d\n" , params[0],params[1],params[2]);
    Calc_block* local_blocks = calculatedBlocks[idx_a][idx_b];



    double local_X[size_params];
    double local_params[size_params];

    memcpy(local_X, X, d_XSize * sizeof(double));
    memcpy(local_params, params, d_paramsSize * sizeof(double));


	params[d_idxParamA] = paramLinspaceA[idx_a];
    params[d_idxParamB] = paramLinspaceB[idx_b];
    
	loopCalculateDiscreteModel(local_X, local_params,
                               d_amountOfTransPoints);

    int nt = 0;

    for (int i = 0; i < d_amountOfCalcBlocks; ++i) {
        for (int j =0; j < d_amountOfNTPoints; j++){
            calculateDiscreteModel(local_X, local_params, d_h);
        }

            memcpy(local_blocks[i].init, local_X, size_params * sizeof(double));
            memcpy(local_blocks[i].params, local_params, size_params * sizeof(double));
            if (i > 0) {
                memcpy(local_blocks[i - 1].result, local_X, size_params * sizeof(double));
            }
                        
            if (nt > d_amountOfCalcBlocks) break; // Avoid return for now, just break


    }


}

__global__ void calculateBlocks(
	Calc_block ***calculatedBlocks,
	double **result
){
    int total_elements = d_size_linspace_A * d_size_linspace_B * d_amountOfCalcBlocks;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx>=total_elements)return;

    int i = idx / (d_size_linspace_B * d_amountOfCalcBlocks);
    int j = (idx % (d_size_linspace_B * d_amountOfCalcBlocks)) / d_amountOfCalcBlocks;
    int k = idx % d_amountOfCalcBlocks;

    Calc_block* block = &calculatedBlocks[i][j][k];
    
    double* new_init = block->init;

    //printf("%f , %f , %f\n" , new_init[0],new_init[1],new_init[2]);

    new_init[0]+=d_eps;
    loopCalculateDiscreteModel(new_init, block->params,
                               d_amountOfNTPoints);
    
    double distance = 0.0;
    for (int l = 0; l < d_XSize; l++) {  
        double diff = block->result[l] - new_init[l];
        distance += diff * diff;  
    }
    //printf("%f" , distance);
    distance = sqrt(distance);  
    atomicAdd(&result[i][j], log(distance));

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

	int amount_of_calc_blocks = static_cast<int>(amountOfAllPoints/amountOfNTPoints) + 1 ;

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

    gpuErrorCheck(cudaMemcpyToSymbol(d_eps, &eps, sizeof(double)));


	
	int max_threads_in_block = 32;

	int gridSizeX = (size_A + max_threads_in_block - 1) / max_threads_in_block;
	int gridSizeY = (size_B + max_threads_in_block - 1) / max_threads_in_block;

	dim3 threadsPerBlock(max_threads_in_block, max_threads_in_block, 1);
	dim3 blocksPerGrid(gridSizeX, gridSizeY, 1);      


    double** d_result;
    double** h_result_temp = new double*[size_A];  // Хост массив указателей на устройство
    gpuErrorCheck(cudaMalloc(&d_result, size_A * sizeof(double*)));
    for (int i = 0; i < size_A; ++i) {
        gpuErrorCheck(cudaMalloc(&h_result_temp[i], size_B * sizeof(double)));
        // Инициализируем нули (опционально)
        double zero = 0.0;
        for (int j = 0; j < size_B; ++j) {
            gpuErrorCheck(cudaMemcpy(h_result_temp[i] + j, &zero, sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    gpuErrorCheck(cudaMemcpy(d_result, h_result_temp, size_A * sizeof(double*), cudaMemcpyHostToDevice));


    Calc_block*** h_calculatedBlocks = new Calc_block**[size_A];
    Calc_block*** h_temp = new Calc_block**[size_A];
    Calc_block*** d_calculatedBlocks;
    gpuErrorCheck(cudaMalloc(&d_calculatedBlocks, size_A * sizeof(Calc_block**)));

    for (int i = 0; i < size_A; ++i) {
        h_calculatedBlocks[i] = new Calc_block*[size_B];
        gpuErrorCheck(cudaMalloc(&h_temp[i], size_B * sizeof(Calc_block*)));
        for (int j = 0; j < size_B; ++j) {
            gpuErrorCheck(cudaMalloc(&h_calculatedBlocks[i][j], amount_of_calc_blocks * sizeof(Calc_block)));

            Calc_block* h_block_temp = new Calc_block[amount_of_calc_blocks];
            for (int k = 0; k < amount_of_calc_blocks; ++k) {
                // Используем h_result_temp, так как это указатели на устройство
                h_block_temp[k].final_num = h_result_temp[i] + j;  // Указатель на d_result[i][j]
            }
            gpuErrorCheck(cudaMemcpy(h_calculatedBlocks[i][j], h_block_temp, 
                                     amount_of_calc_blocks * sizeof(Calc_block), cudaMemcpyHostToDevice));
            delete[] h_block_temp;
        }
        gpuErrorCheck(cudaMemcpy(h_temp[i], h_calculatedBlocks[i], size_B * sizeof(Calc_block*), cudaMemcpyHostToDevice));
    }
    gpuErrorCheck(cudaMemcpy(d_calculatedBlocks, h_temp, size_A * sizeof(Calc_block**), cudaMemcpyHostToDevice));




	double* d_paramLinspaceA;
	double* d_paramLinspaceB;
	double* d_X;
	double* d_params;

	gpuErrorCheck(cudaMalloc(&d_X, amount_init * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_params, amount_params * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_paramLinspaceA, size_A * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_paramLinspaceB, size_B * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_X, initialConditions, amount_init * sizeof(double),
	 						 cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_params, params, amount_params * sizeof(double),
	 						 cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_paramLinspaceA, linspaceA, size_A * sizeof(double),
	 						 cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_paramLinspaceB, linspaceB, size_B * sizeof(double),
							 cudaMemcpyHostToDevice));



	LLE_constants::calculateSystem<<<blocksPerGrid, threadsPerBlock>>>(
	d_X,
	d_params,
    d_paramLinspaceA,
    d_paramLinspaceB,
    d_calculatedBlocks
	);

    gpuErrorCheck(cudaDeviceSynchronize());

    printf("First calculation ended\n");

    int total_elements = size_A * size_B * amount_of_calc_blocks;

    // Define threads per block (use 256 or 1024, depending on your GPU and preference)
    const int threads_per_block = 1024;

    // Calculate number of blocks, ensuring at least 1 block
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block; // Ceiling division

    dim3 threadsPerBlock2(threads_per_block, 1, 1);
    dim3 blocksPerGrid2(num_blocks, 1, 1);

    LLE_constants::calculateBlocks<<<blocksPerGrid2, threadsPerBlock2>>>(
    d_calculatedBlocks,
    d_result
	);

	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaPeekAtLastError());
    printf("Second calculation ended\n");

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