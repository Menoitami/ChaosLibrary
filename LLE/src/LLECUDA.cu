#include <LLECUDA.cuh>
#include <string>
#include <nvrtc.h>
namespace LLE_constants{
	
__device__ void calculateDiscreteModel(double *x, const double *a, const double h)
{
    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);
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
    printf("%d , %d\n" , idx_a,idx_b);
    Calc_block* local_blocks = calculatedBlocks[idx_a][idx_b];




	params[d_idxParamA] = paramLinspaceA[idx_a];
    params[d_idxParamB] = paramLinspaceB[idx_b];

	loopCalculateDiscreteModel(X, params,
                               d_amountOfTransPoints);

    int nt = 0;

    for (int i = 0; i < d_amountOfAllpoints; ++i) {
        calculateDiscreteModel(X, params, d_h);
        if ((i / d_Nt_steps) % 10 == 0) {
            int index = nt;
            local_blocks[index].init = X;
            local_blocks[index].params = params;
            if (index > 0) {
                local_blocks[index - 1].result = X;
            }

            nt++;
        }
    }


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
	const double transientTime,
	const double* params,
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

	int paramsSize =  static_cast<int>(sizeof(params)/ sizeof(double));
	int XSize =  static_cast<int>(sizeof(initialConditions)/sizeof(double));

	const int size_A =  static_cast<int>(linspaceA_params[2]);
	const int size_B =  static_cast<int>(linspaceB_params[2]);
	int NT_steps = static_cast<int>(NT/h);

	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamA, &indicesOfMutVars[0], sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_idxParamB, &indicesOfMutVars[1], sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_A, &size_A, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_size_linspace_B, &size_B, sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_transTime, &transientTime, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_Nt_steps, &NT_steps, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));



	
	int max_threads_in_block = 32;

	int gridSizeX = (size_A + max_threads_in_block - 1) / max_threads_in_block;
	int gridSizeY = (size_B + max_threads_in_block - 1) / max_threads_in_block;

	dim3 threadsPerBlock(max_threads_in_block, max_threads_in_block, 1);
	dim3 blocksPerGrid(gridSizeX, gridSizeY, 1);      




	int amount_of_calc_blocks = tMax/NT + 1 ;

// 3D array allocation: [size_A][size_B][ammount_of_calc_blocks]
    // Step 1: Host staging arrays
    Calc_block*** h_calculatedBlocks = new Calc_block**[size_A];
    Calc_block** *h_temp = new Calc_block**[size_A];  // Intermediate pointers

    // Step 2: Allocate device memory
    Calc_block*** d_calculatedBlocks;
    gpuErrorCheck(cudaMalloc(&d_calculatedBlocks, size_A * sizeof(Calc_block**)));

    for (int i = 0; i < size_A; ++i) {
        h_calculatedBlocks[i] = new Calc_block*[size_B];
        gpuErrorCheck(cudaMalloc(&h_temp[i], size_B * sizeof(Calc_block*)));
        for (int j = 0; j < size_B; ++j) {
            gpuErrorCheck(cudaMalloc(&h_calculatedBlocks[i][j], amount_of_calc_blocks * sizeof(Calc_block)));
            // Optionally initialize Calc_block members if needed (e.g., init, params)
            // For now, assume kernel fills these
        }
        gpuErrorCheck(cudaMemcpy(h_temp[i], h_calculatedBlocks[i], size_B * sizeof(Calc_block*), cudaMemcpyHostToDevice));
    }
    gpuErrorCheck(cudaMemcpy(d_calculatedBlocks, h_temp, size_A * sizeof(Calc_block**), cudaMemcpyHostToDevice));




	double* d_paramLinspaceA;
	double* d_paramLinspaceB;
	double* d_X;
	double* d_params;

	gpuErrorCheck(cudaMalloc(&d_X, XSize * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_params, paramsSize * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_paramLinspaceA, size_A * sizeof(double)));
	gpuErrorCheck(cudaMalloc(&d_paramLinspaceB, size_B * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_X, initialConditions, XSize * sizeof(double),
	 						 cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_params, params, paramsSize * sizeof(double),
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
	gpuErrorCheck(cudaPeekAtLastError());

	delete[] linspaceA;
	delete[] linspaceB;
}

} //LLE_constants