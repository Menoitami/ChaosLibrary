#include <LLEHost.cuh>
#include <LLECUDA.cuh>

using namespace old_library;

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
	const double* params,
	const int amountOfValues,
	std::string		OUT_FILE_PATH)
{

	int amountOfNTPoints = static_cast<int>(NT / h);
	int amountOfTransPoints= static_cast<int>(transientTime / h);
	int amountOfAllPoints= static_cast<int>(tMax / h);

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	int paramsSize = sizeof(params)/ sizeof(double);
	int XSize = sizeof(initialConditions)/sizeof(double);

	double* paramLinspaceA;
	double* paramLinspaceB;



	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_transTime, &transientTime, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_Nt, &NT, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_paramsSize, &paramsSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_XSize, &XSize, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfNTPoints, &amountOfNTPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfTransPoints, &amountOfTransPoints, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_amountOfAllpoints, &amountOfAllPoints, sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_init, &initialConditions, sizeof(XSize)));
	gpuErrorCheck(cudaMemcpyToSymbol(LLE_constants::d_params, &params, sizeof(paramsSize)));
	

/*
	Calc_block*** calculatedBlocks;
	cudaMalloc(&calculatedBlocks, sizeof(Calc_block**) * d_sizeA);
	for (int i = 0; i < d_sizeA; ++i) {
		cudaMalloc(&calculatedBlocks[i], sizeof(Calc_block*) * d_sizeB);
		for (int j = 0; j < d_sizeB; ++j) {
			cudaMalloc(&calculatedBlocks[i][j], sizeof(Calc_block) * (d_tMax / d_h / d_Nt));
		}
	}
*/

/*
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
}
