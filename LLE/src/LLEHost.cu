#include <LLEHost.cuh>
#include <LLECUDA.cuh>
//using namespace old_library;
/*
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
	printf("%d\n", size_B);
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_size_linspace_A, &size_A, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_size_linspace_B, &size_B, sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_transTime, &transientTime, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_Nt_steps, &NT_steps, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_paramsSize, &paramsSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_XSize, &XSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));



	
	int max_threads_in_block = 32;

	int gridSizeX = (size_A + max_threads_in_block - 1) / max_threads_in_block;
	int gridSizeY = (size_B + max_threads_in_block - 1) / max_threads_in_block;

	dim3 threadsPerBlock(max_threads_in_block, max_threads_in_block, 1);
	dim3 blocksPerGrid(gridSizeX, gridSizeY, 1);      




	int ammount_of_calc_blocks = tMax/NT + 1 ;
	size_t required = size_A * sizeof(Calc_block**) +
					size_A * size_B * sizeof(Calc_block*) +
					size_A * size_B * ammount_of_calc_blocks * sizeof(Calc_block);
	printf("Required memory: %zu bytes\n", required);
	if (required > freeMemory) {
		printf("Insufficient GPU memory!\n");
		exit(1);
	}
		// Host-side staging arrays
	Calc_block*** h_calculatedBlocks = new Calc_block**[size_A];
	for (int i = 0; i < size_A; ++i) {
		h_calculatedBlocks[i] = new Calc_block*[size_B];
		for (int j = 0; j < size_B; ++j) {
			gpuErrorCheck(cudaMalloc(&h_calculatedBlocks[i][j], sizeof(Calc_block) * ammount_of_calc_blocks));
		}
	}

	// Device pointer for top-level array
	Calc_block*** d_calculatedBlocks;
	gpuErrorCheck(cudaMalloc(&d_calculatedBlocks, sizeof(Calc_block**) * size_A));

	// Intermediate device pointers
	Calc_block** *d_temp = new Calc_block**[size_A];
	for (int i = 0; i < size_A; ++i) {
		gpuErrorCheck(cudaMalloc(&d_temp[i], sizeof(Calc_block*) * size_B));
		gpuErrorCheck(cudaMemcpy(d_temp[i], h_calculatedBlocks[i], sizeof(Calc_block*) * size_B, cudaMemcpyHostToDevice));
	}
	gpuErrorCheck(cudaMemcpy(d_calculatedBlocks, d_temp, sizeof(Calc_block**) * size_A, cudaMemcpyHostToDevice));

	// Cleanup host memory
	for (int i = 0; i < size_A; ++i) {
		delete[] h_calculatedBlocks[i];
	}
	delete[] h_calculatedBlocks;




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
	// Cleanup (in reverse order)

	


	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &LLE_constants::size, sizeof(int)));
	//freeMemory *= 0.95;
	//size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 10000; // Pizdec kostil' ot Boga

	//nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	//size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LLE2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif
	int stringCounter = 0;
	
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMax;
		int blockSizeMin;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 10000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelCUDA <<< gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >>> (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			amountOfPointsInBlock, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				if (isnan(h_lleResult[i]))
					outFileStream << 0;
				else 
					outFileStream << h_lleResult[i];

				++stringCounter;
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
	*/
//}
