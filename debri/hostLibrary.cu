// --- ������������ ���� ---
#include "hostLibrary.cuh"

// --- ���� ��� ���������� �������������� ������ ---
//#define OUT_FILE_PATH "C:\\Users\\KiShiVi\\Desktop\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// --- ���������, ���������� ������� ������� � ������� ���������� ��������� ---
#define DEBUG
namespace old_library{
__host__ void FastSynchro(
	const double	tMax,								// ����� ������������� �������
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double	NTime,								// ����� ������� �� �������� ����� ����������� �������������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const double	h,									// ��� ��������������
	const double*	kForward,							// ������ ������������� ������������� ������
	const double*	kBackward,							// ������ ������������� ������������� �����
	const double*	initialConditionsMaster,			// ������ � ���������� ��������� �������
	const double*	initialConditionsSlave,				// ������ � ���������� ��������� ������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������
	const int		iterOfSynchr,						// ����� �������� �������������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ������������ � ���� ������������� ---
	int amountOfNTPoints = NTime / h;

	// --- ����� ���������� ����� � �������� ���������� ---
	int amountOfCTPoints = tMax / h;

	// --- ���������� ����� ����������� �������� ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU
	size_t nPts = (amountOfCTPoints / preScaller);

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
// TODO ������� ������ ��������� ������
	//size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfNTPoints * (amountOfInitialConditions*amountOfInitialConditions*amountOfInitialConditions));
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfNTPoints * amountOfNTPoints*5);
	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter; // ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)
	
	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )


	double* timeDomain = new double[(amountOfCTPoints + amountOfNTPoints)* sizeof(double) * amountOfInitialConditions];
	double* arrayZeros = new double[sizeof(double) * amountOfInitialConditions];
	double* Xm = new double[sizeof(double) * amountOfInitialConditions];
	double* Xs = new double[sizeof(double) * amountOfInitialConditions];

	// --- ������������� ��������� ������� ---
	for (int i = 0; i < amountOfInitialConditions; i++) {
		arrayZeros[i] = 0;
		Xm[i] = initialConditionsMaster[i];
		Xs[i] = initialConditionsSlave[i];
	}
		
	// --- ������ ����������� �������� ---
	for (size_t i = 0; i < amountOfPointsForSkip; i++) {
		calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h);
		calculateDiscreteModelforFastSynchro(Xs, arrayZeros, arrayZeros, values, h);
	}

	//for (int i = 0; i < amountOfInitialConditions; i++) {
	//	Xs[i] = initialConditionsSlave[i];
	//}

	// --- ������ �������� ���������� ---
	for (size_t i = 0; i < amountOfCTPoints + amountOfNTPoints; i++) {

		for (int j = 0; j < amountOfInitialConditions; j++)
			timeDomain[i * amountOfInitialConditions + j] = Xm[j];

		calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h);

	}

	printf(" --- Calculation of trajectory done\n");

	// --- �������� ������ ��� �������� ��������� ���������� 
	double* h_output = new double[nPts * sizeof(double)];

	// --- ��������� �� ������� ������ � GPU ---

	double* d_timeDomain;
	double* d_output;
	double* d_Xs;
	double* d_values;
	double* d_kForward;
	double* d_kBackward;
	// --- �������� ������ � GPU ---

	gpuErrorCheck(cudaMalloc((void**)& d_timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_output, nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_Xs, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kForward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kBackward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	// --- �������� ��������� ������� ��������� � ������ GPU ---

	gpuErrorCheck(cudaMemcpy(d_timeDomain, timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_Xs, Xs, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);
	
	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---

		//blockSize = ceil((1*1024.0f * 4.0f) / (amountOfNTPoints * sizeof(double)));
		//blockSize = ceil((1 * 1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		//blockSize = 10000 / ((5*amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSize = 5*amountOfNTPoints;
		//
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )
		blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

	
			//calculateDiscreteModelforFastSynchroCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			//calculateDiscreteModelforFastSynchroCUDA << <4*1024, 16>> > //, 1*(amountOfInitialConditions + amountOfValues + amountOfNTPoints) * sizeof(double)* 1 >> >
		calculateDiscreteModelforFastSynchroCUDA << < gridSize, blockSize >> >
		(
				nPts,						//const int		nPts,
				nPtsLimiter,				//const int		nPtsLimiter,
				1*amountOfNTPoints,		//const int		sizeOfBlock,
				h, 							//const double	h,
				d_Xs,						//double* initialConditions,
				amountOfInitialConditions,	//const int		amountOfInitialConditions,
				d_values,						//const double* values,
				d_kForward,					//const double* k_forward,
				d_kBackward,					//const double* k_backward,
				iterOfSynchr,							//const int		iterOfSynchr,
				amountOfValues,								//const int		
				amountOfNTPoints,							//const int		amountOfIterations,
				maxValue,									//const double	maxValue,
				d_timeDomain + (i* originalNPtsLimiter)* amountOfInitialConditions * preScaller,								//double* timedomain,
				d_output + (i* originalNPtsLimiter),			//double* output
				preScaller);

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

			
#ifdef DEBUG
		printf(" --- Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}
		// --- ������� ������������ ����������� � gpu 
	gpuErrorCheck(cudaMemcpy(h_output, d_output, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- ������������ ������ � gpu 
	gpuErrorCheck(cudaFree(d_timeDomain));
	gpuErrorCheck(cudaFree(d_output));
	gpuErrorCheck(cudaFree(d_kForward));
	gpuErrorCheck(cudaFree(d_kBackward));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_Xs));

	// --- ������ ���������� � ���� 
	outFileStream << std::setprecision(20);

	for (size_t j = 0; j < nPts; ++j)
		if (outFileStream.is_open())
		{
			for (int k = 0; k < amountOfInitialConditions; k++) 
				outFileStream << timeDomain[amountOfInitialConditions * j * preScaller + k] << ", ";		

			outFileStream << h_output[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}
	outFileStream.close();

	printf(" --- Writing in file done\n");

	delete[] arrayZeros;
	delete[] timeDomain;
	delete[] Xm;
	delete[] Xs;
	delete[] h_output;
}

__host__ void FastSynchro_2(
	const double	NTime,								// ����� ������� �� �������� ����� ����������� �������������
	const int		nPts,							// ���������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const double	h,									// ��� ��������������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const double*	kForward,							// ������ ������������� ������������� ������
	const double*	kBackward,							// ������ ������������� ������������� �����
	const double*	initialConditions,			// ������ � ���������� ��������� �������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������
	const int		iterOfSynchr,						// ����� �������� �������������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	std::string		OUT_FILE_PATH)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = NTime / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = 0;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfInitialConditions * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	double* h_dbscanResult = new double[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	double* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������
	
	double* d_kForward;
	double* d_kBackward;

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfInitialConditions * nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kForward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kBackward, amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCforFastSynchro << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfInitialConditions*amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				iterOfSynchr,
				d_kForward,
				d_kBackward,
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks,
				d_dbscanResult);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������


		//const int		nPts,
		//	const int		nPtsLimiter,
		//	const int		sizeOfBlock,
		//	const int		amountOfCalculatedPoints,
		//	const int		dimension,
		//	double* ranges,
		//	const double	h,
		//	int* indicesOfMutVars,
		//	double* initialConditions,
		//	const int		amountOfInitialConditions,
		//	const double* values,
		//	const int		amountOfValues,
		//	const int		amountOfIterations,
		//	const int		preScaller,
		//	const double	maxValue,
		//	const int		iterOfSynchr,
		//	const double* kForward,
		//	const double* kBackward,
		//	double* data,
		//	int* maxValueCheckerArray,
		//	double* FastSynchroError)

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
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
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
#ifdef DEBUG
				printf("\nOutput file open error\n");
#endif
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	
}

__host__ void distributedSystemSimulation(
	const double	tMax,							// ����� ������������� �������
	const double	h,								// ��� ��������������
	const double	hSpecial,						// ��� �������� ����� ��������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)					// ���������� ����������	
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.8;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)	

	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_data = new double[amountOfPointsInBlock * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Distributed System Simulation\n");
#endif

	int blockSize;			// ���������� ��� �������� ������� �����
	int minGridSize;		// ���������� ��� �������� ������������ ������� �����
	int gridSize;			// ���������� ��� �������� �����

	// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
	// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
	// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
	// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
	blockSize = ceil((1024.0f * 8.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
	if (blockSize < 1)
	{
#ifdef DEBUG
		printf("Error : BlockSize < 1; %d line\n", __LINE__);
		exit(1);
#endif
	}

	blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

	gridSize = (amountOfThreads + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

	distributedCalculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
		(
			amountOfPointsForSkip,
			amountOfThreads,
			h,
			hSpecial,
			d_initialConditions,
			amountOfInitialConditions,
			d_values,
			amountOfValues,
			tMax / hSpecial,
			writableVar,
			d_data
			);

	// --- �������� �� CUDA ������ ---
	gpuGlobalErrorCheck();

	// --- ���� ���� ��� ������ �������� ���� ������ ---
	gpuErrorCheck(cudaDeviceSynchronize());

	// -------------------------------------------------------------------------------------
	// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
	// -------------------------------------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------------------

	// --- �������� ����� � ��������� ������� ---
	outFileStream << std::setprecision(20);

	for (size_t j = 0; j < amountOfPointsInBlock; ++j)
		if (outFileStream.is_open())
		{
			outFileStream << h * j << ", " << h_data[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}


	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	delete[] h_data;
}


// ----------------------------------------------------------------------------
// --- ����������� �������, ��� ������� ���������� �������������� ��������� ---
// ----------------------------------------------------------------------------

__host__ void bifurcation1D(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const double* ranges,							// ��������� ��������� ����������
	const int* indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.9;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	double* h_outIntervals = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// ��� ��� ���� �����
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	
	int* d_sysCheker;			// ��������� �� ������ � GPU �� ��������������� ������
	double* d_avgPeaks;
	double* d_avgIntervals;

	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * sizeof(double)));


	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(
				nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		//calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
		//	(nPts,						// ����� ���������� ��������� - nPts
		//		nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
		//		amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
		//		i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
		//		amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
		//		1,							// ����������� ( ��������� ���������� )
		//		d_ranges,					// ������ � �����������
		//		h,							// ��� ��������������
		//		d_indicesOfMutVars,			// ������� ���������� ����������
		//		d_initialConditions,		// ��������� �������
		//		amountOfInitialConditions,	// ���������� ��������� �������
		//		d_values,					// ���������
		//		amountOfValues,				// ���������� ����������
		//		amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
		//		preScaller,					// ���������, ������� ��������� ����� � ����� ��������
		//		writableVar,				// ������ ���������, �� �������� ����� ������� ���������
		//		maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
		//		d_data,						// ������, ��� ����� �������� ���������� ������
		//		d_sysCheker + (i * originalNPtsLimiter));			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� �������� ����� �� �����
				h*preScaller);							// ��� �������������� �� �����

		////		// --- �������� �� CUDA ������ ---
		////gpuGlobalErrorCheck();

		////// --- ���� ���� ��� ������ �������� ���� ������ ---
		////gpuErrorCheck(cudaDeviceSynchronize());

		////// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		////cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_for2Dbif, 0, nPtsLimiter);
		//////blockSize = blockSize > 512 ? 512 : blockSize;			// �� ��������� ����������� � 512 ������ � �����
		////gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		////avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
		////	(d_data,						// ������ � ������������ ������
		////		amountOfPointsInBlock,		// ���������� ����� � ����� ����������
		////		nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
		////		d_avgPeaks + (i * originalNPtsLimiter),
		////		d_avgIntervals + (i * originalNPtsLimiter),
		////		d_data,						// �������� ������, ���� ����� �������� �������� �����
		////		d_intervals,				// ���������� ��������
		////		d_amountOfPeaks + (i * originalNPtsLimiter),
		////		d_sysCheker + (i * originalNPtsLimiter),
		////		h* preScaller);			// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_outIntervals, d_intervals, nPtsLimiter* amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k) {
			if (h_amountOfPeaks[k] == -1) {
				outFileStream << getValueByIdx(originalNPtsLimiter* i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock] << ", " << 0  << '\n';
			}
			else {
				for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
					if (outFileStream.is_open())
					{
						outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << ", " << h_outIntervals[k * amountOfPointsInBlock + j] << '\n';
					}
					else
					{
#ifdef DEBUG
						printf("\nOutput file open error\n");
#endif
						exit(1);
					}
			}
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}


	//double* h_avgPeaks = new double[nPts];
	//double* h_avgIntervals = new double[nPts];
	//int* h_sysCheker = new int[nPts];
	//int* h_amountOfPeaks = new int[nPts];

	//gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


//	std::ofstream outFileStream;
//	outFileStream.open(OUT_FILE_PATH);
//	outFileStream << std::setprecision(12);
//
//		// --- ���������� ������ � ���� ---
//			for (size_t j = 0; j < nPts; ++j)
//				if (outFileStream.is_open())
//				{
//					outFileStream  << h_avgPeaks[j] << ", " << h_avgIntervals[j] << ", " << h_amountOfPeaks[j] << '\n';
//				}
//				else
//				{
//#ifdef DEBUG
//					printf("\nOutput file open error\n");
//#endif
//					exit(1);
//				}

					   

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));
	gpuErrorCheck(cudaFree(d_sysCheker));



	delete[] h_outPeaks;
	delete[] h_outIntervals;
	delete[] h_amountOfPeaks;
	

	// ---------------------------
}



/**
 * �������, ��� ������� ���������� �������������� ��������� �� ����.
 */
__host__ void bifurcation1DForH(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const double* ranges,							// �������� ��������� ����
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� ����� � ����� ����� ---
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				transientTime,				// ����� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				tMax,						// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				nullptr,					// ���������� �������� ����� �� �����
				0);							// ��� �������������� �� �����

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// -----------------------------------------------------------------------------------------
// --- �������, ��� ������� ���������� �������������� ���������. (�� ��������� ��������) ---
// -----------------------------------------------------------------------------------------

__host__ void bifurcation1DIC(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const double* ranges,							// ��������� ��������� ����������
	const int* indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	// ���������: ����� �� ����� ���� ������, ��� (amountOfPointsInBlock / 2), �.�. ����� ���� �� ����� ����� ���� ���
	double* h_outPeaks = new double[ceil(nPtsLimiter * amountOfPointsInBlock * sizeof(double) / 2.0f)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				nullptr,					// ���������� �������� ����� �� �����
				0);							// ��� �������������� �� �����

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// ------------------------------------------------------------------------
// --- �������, ��� ������� ��������� �������������� ��������� (DBSCAN) ---
// ------------------------------------------------------------------------

__host__ void bifurcation2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.95;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		
		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		blockSize = 20000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

//		if (blockSize < 1)
//		{
//#ifdef DEBUG
//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
//			exit(1);
//#endif
//		}
//
//		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����
//
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >(
				nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// �� ��������� ����������� � 512 ������ � �����
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				h * (double)preScaller);							// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > (	
				d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks,
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
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
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
#ifdef DEBUG
				printf("\nOutput file open error\n");
#endif
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	// ---------------------------
}




__host__ void neuronClasterization2D_2(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.9;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	
	int* d_sysCheker;			// ��������� �� ������ � GPU �� ��������������� ������
	double* d_avgPeaks;
	double* d_avgIntervals;
	double* d_helpfulArray;

	//double* d_valleys;
	//double* d_TimeOfValleys;// ��������� �� ������ � GPU � ����������� ����������� ���������
		//int* d_dbscanResult;
	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));


	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_valleys, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_TimeOfValleys, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

		// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	//for (int i = 1; i < 5; i++) {
	//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
	//	// --- ������� � ����� ������ ����� ����������� �������� ---
	//	if (outFileStream.is_open())
	//	{
	//		outFileStream << ranges[0] << " " << ranges[1] << "\n";
	//		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	//	}
	//	outFileStream.close();
	//}
	

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����



	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---

		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

		//		if (blockSize < 1)
		//		{
		//#ifdef DEBUG
		//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
		//			exit(1);
		//#endif
		//		}
		//
		//		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����
		//
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
				(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_sysCheker + (i* originalNPtsLimiter));			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA_for_NeuronClassification, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// �� ��������� ����������� � 512 ������ � �����
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA_for_NeuronClassification << <gridSize, blockSize >> >
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks + (i* originalNPtsLimiter),			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				nullptr,
				nullptr,
				h * preScaller);							// ��� ��������������

		// -----------------------------------------


		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, NeuronClassificationCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		NeuronClassificationCUDA << <gridSize, blockSize >> >
			(	
				d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks + (i* originalNPtsLimiter),
				d_intervals,
				nullptr,
				nullptr,
				d_helpfulArray,
				eps,
				d_dbscanResult + (i* originalNPtsLimiter)
			);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------



		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		//gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		//gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		//outFileStream << std::setprecision(12);

//		// --- ���������� ������ � ���� ---
//		for (size_t i = 0; i < nPtsLimiter; ++i)
//			if (outFileStream.is_open())
//			{
//				if (stringCounter != 0)
//					outFileStream << ", ";
//				if (stringCounter == nPts)
//				{
//					outFileStream << "\n";
//					stringCounter = 0;
//				}
//				outFileStream << h_dbscanResult[i];
//				++stringCounter;
//			}
//			else
//			{
//#ifdef DEBUG
//				printf("\nOutput file open error\n");
//#endif
//				exit(1);
//			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	//double* h_avgPeaks = new double[nPts * nPts];
	//double* h_avgIntervals = new double[nPts * nPts];
	//int* h_sysCheker = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];
	//int* h_amountOfPeaks = new int[nPts * nPts];

	//gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts* nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts* nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	


	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

		// --- ���������� ��������� ��������� ���������� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_dbscanResult[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_avgPeaks[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_avgIntervals[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_sysCheker[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(4) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_amountOfPeaks[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));

	//gpuErrorCheck(cudaFree(d_valleys));
	//gpuErrorCheck(cudaFree(d_TimeOfValleys));

	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_sysCheker));

	delete[] h_dbscanResult;
	//delete[] h_avgPeaks;
	//delete[] h_avgIntervals;
	//delete[] h_sysCheker; 
	//delete[] h_amountOfPeaks;
	// ---------------------------
}

__host__ void neuronClasterization2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.95;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU

	int* d_sysCheker;			// ��������� �� ������ � GPU �� ��������������� ������
	double* d_avgPeaks;
	double* d_avgIntervals;
	double* d_helpfulArray;
	//int* d_dbscanResult;
// -----------------------------------------

// -----------------------------
// --- �������� ������ � GPU ---
// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));


	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	for (int i = 1; i < 5; i++) {
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
		// --- ������� � ����� ������ ����� ����������� �������� ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}


	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����



	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---

		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

		//		if (blockSize < 1)
		//		{
		//#ifdef DEBUG
		//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
		//			exit(1);
		//#endif
		//		}
		//
		//		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����
		//
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_sysCheker + (i * originalNPtsLimiter));			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_for2Dbif, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// �� ��������� ����������� � 512 ������ � �����
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		//peakFinderCUDA << <gridSize, blockSize >> >
		//	(d_data,						// ������ � ������������ ������
		//		amountOfPointsInBlock,		// ���������� ����� � ����� ����������
		//		nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
		//		d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
		//		d_data,						// �������� ������, ���� ����� �������� �������� �����
		//		d_intervals,				// ���������� ��������
		//		h * preScaller);							// ��� ��������������

		avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
			(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				d_amountOfPeaks + (i * originalNPtsLimiter),
				d_sysCheker + (i * originalNPtsLimiter),
				h * preScaller);			// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> >
			(d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks + (i * originalNPtsLimiter),
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult + (i * originalNPtsLimiter)
				);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------



		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		//outFileStream << std::setprecision(12);

//		// --- ���������� ������ � ���� ---
//		for (size_t i = 0; i < nPtsLimiter; ++i)
//			if (outFileStream.is_open())
//			{
//				if (stringCounter != 0)
//					outFileStream << ", ";
//				if (stringCounter == nPts)
//				{
//					outFileStream << "\n";
//					stringCounter = 0;
//				}
//				outFileStream << h_dbscanResult[i];
//				++stringCounter;
//			}
//			else
//			{
//#ifdef DEBUG
//				printf("\nOutput file open error\n");
//#endif
//				exit(1);
//			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	int* h_sysCheker = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];
	int* h_amountOfPeaks = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));



	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

		// --- ���������� ��������� ��������� ���������� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_dbscanResult[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgIntervals[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_sysCheker[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(4) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_amountOfPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_sysCheker));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_sysCheker;
	delete[] h_amountOfPeaks;
	// ---------------------------
}

// ------------------------------------------------------------------------------
// --- �������, ��� ������� ��������� �������������� ��������� (DBSCAN) �� IC ---
// ------------------------------------------------------------------------------

__host__ void bifurcation2DIC(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				h * preScaller);							// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> >
			(d_data, 					// ������ (����)
				amountOfPointsInBlock, 		// ���������� ����� � ����� �������
				nPtsLimiter,				// ���������� ������ (������) � data
				d_amountOfPeaks, 			// ������, ���������� ���������� ����� ��� ������� ����� � data
				d_intervals, 				// ���������� ���������
				d_helpfulArray, 			// ��������������� ������ 
				eps, 						// �������
				d_dbscanResult);			// �������������� ������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
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
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
#ifdef DEBUG
				printf("\nOutput file open error\n");
#endif
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	// ---------------------------
}



__host__ void LLE1D(
	const double	tMax,								// ����� ������������� �������
	const double	NT,									// ����� ������������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const double	eps,								// ������� ��� LLE
	const double* initialConditions,					// ������ � ���������� ���������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)						// ���������� ����������
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� �� ����� ������������ NT ---
	size_t amountOfNT_points = NT / h;

	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / NT;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;																// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;																// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )

	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_ranges;				   // ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		   // ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	   // ��������� �� ������ � ���������� ���������
	double* d_values;				   // ��������� �� ������ � �����������

	double* d_lleResult;			   // ������ ��� �������� ��������� ����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("LLE 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// ���������� ��� �������� ������� �����
		int minGridSize;	// ���������� ��� �������� ������������ ������� �����
		int gridSize;		// ���������� ��� �������� �����

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double)));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )


		// ------------------------------------
		// --- CUDA ������� ��� ������� LLE ---
		// ------------------------------------

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,								// ����� ����������
				nPtsLimiter, 						// ���������� � ������� �������
				NT, 								// ����� ������������
				tMax, 								// ����� �������������
				amountOfPointsInBlock,				// ���������� �����, ���������� ����� �������� � "data"
				i * originalNPtsLimiter, 			// ���������� ��� ����������� �����
				amountOfPointsForSkip,				// ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
				1, 									// �����������
				d_ranges, 							// ������, ���������� ��������� ������������� ���������
				h, 									// ��� ��������������
				eps, 								// �������
				d_indicesOfMutVars, 				// ������� ���������� ����������
				d_initialConditions,				// ��������� �������
				amountOfInitialConditions, 			// ���������� ��������� �������
				d_values, 							// ���������
				amountOfValues, 					// ���������� ����������
				tMax / NT, 							// ���������� �������� (����������� �� tMax)
				1, 									// ��������� ��� ��������� ��������
				writableVar,						// ������ ���������� � x[] �� �������� ������ ���������
				maxValue, 							// ������������� �������� ���������� ��� �������������
				d_lleResult);						// �������������� ������

		// ------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
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

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



__host__ void LLE1DIC(
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
	std::string		OUT_FILE_PATH)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� �� ����� ������������ NT ---
	size_t amountOfNT_points = NT / h;

	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / NT;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;																// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;																// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )

	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_ranges;				   // ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		   // ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	   // ��������� �� ������ � ���������� ���������
	double* d_values;				   // ��������� �� ������ � �����������

	double* d_lleResult;			   // ������ ��� �������� ��������� ����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("LLE 1DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// ���������� ��� �������� ������� �����
		int minGridSize;	// ���������� ��� �������� ������������ ������� �����
		int gridSize;		// ���������� ��� �������� �����

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double)));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )


		// ------------------------------------
		// --- CUDA ������� ��� ������� LLE ---
		// ------------------------------------

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,								// ����� ����������
				nPtsLimiter, 						// ���������� � ������� �������
				NT, 								// ����� ������������
				tMax, 								// ����� �������������
				amountOfPointsInBlock,				// ���������� �����, ���������� ����� �������� � "data"
				i * originalNPtsLimiter, 			// ���������� ��� ����������� �����
				amountOfPointsForSkip,				// ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
				1, 									// �����������
				d_ranges, 							// ������, ���������� ��������� ������������� ���������
				h, 									// ��� ��������������
				eps, 								// �������
				d_indicesOfMutVars, 				// ������� ���������� ����������
				d_initialConditions,				// ��������� �������
				amountOfInitialConditions, 			// ���������� ��������� �������
				d_values, 							// ���������
				amountOfValues, 					// ���������� ����������
				tMax / NT, 							// ���������� �������� (����������� �� tMax)
				1, 									// ��������� ��� ��������� ��������
				writableVar,						// ������ ���������� � x[] �� �������� ������ ���������
				maxValue, 							// ������������� �������� ���������� ��� �������������
				d_lleResult);						// �������������� ������

		// ------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
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

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	const int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 10000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

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

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > (
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
}


__host__ void LLE2DIC(
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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 22000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

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

		blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
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
}



__host__ void LS1D(
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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LS1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double)) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			1, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0);
				for (int j = 0; j < amountOfInitialConditions; ++j)
					outFileStream << ", " << h_lleResult[k * amountOfInitialConditions + j];
				outFileStream << '\n';
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
}




__host__ void LS2D(
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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts * nPts ? nPts * nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LS2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int* stringCounter = new int[amountOfInitialConditions];

	for (int i = 0; i < amountOfInitialConditions; ++i)
		stringCounter[i] = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		outFileStream.open(OUT_FILE_PATH + std::to_string(i + 1) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double)) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < amountOfInitialConditions; ++k)
		{
			outFileStream.open(OUT_FILE_PATH + std::to_string(k + 1) + ".csv", std::ios::app);
			for (size_t m = 0 + k; m < nPtsLimiter * amountOfInitialConditions; m = m + amountOfInitialConditions)
			{
				if (outFileStream.is_open())
				{
					if (stringCounter[k] != 0)
						outFileStream << ", ";
					if (stringCounter[k] == nPts)
					{
						outFileStream << "\n";
						stringCounter[k] = 0;
					}
					outFileStream << h_lleResult[m];
					stringCounter[k] = stringCounter[k] + 1;
				}
			}
			outFileStream.close();
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

	delete[] stringCounter;
	delete[] h_lleResult;
}



__host__ void basinsOfAttraction(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.8;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )

	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];
	//double* h_helpfulArray = new double[nPts * nPts];			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	//double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	double* d_avgPeaks;
	double* d_avgIntervals;

	int* d_sysCheck;

	//int* h_sysCheck = new int[nPtsLimiter * sizeof(int)];


	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//gpuErrorCheck( cudaMalloc( (void** )&d_helpfulArray,		nPts * nPts * sizeof( double ) ) );

	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_sysCheck, nPts * nPts * sizeof(int)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);


#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	// --- �������� ����� � ��������� ������� ---
	outFileStream << std::setprecision(15);

	// --- ������� � ����� ������ ����� ����������� �������� ---
	outFileStream.open(OUT_FILE_PATH);
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int blockSize;			// ���������� ��� �������� ������� �����
	int minGridSize;		// ���������� ��� �������� ������������ ������� �����
	int gridSize;			// ���������� ��� �������� �����

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		//blockSize = 12000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_sysCheck + (i * originalNPtsLimiter));				// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		avgPeakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				d_sysCheck + (i * originalNPtsLimiter),
				h * preScaller);			// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif

		//gpuErrorCheck(cudaMemcpy(h_sysCheck, d_sysCheck, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


		//outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
		//for (size_t i = 0; i < nPtsLimiter; ++i)
		//	if (outFileStream.is_open())
		//	{
		//		if (stringCounter != 0)
		//			outFileStream << ", ";
		//		if (stringCounter == nPts)
		//		{
		//			outFileStream << "\n";
		//			stringCounter = 0;
		//		}
		//		//outFileStream << h_avgIntervals[i];
		//		if (h_sysCheck[i] != NAN)
		//			outFileStream << h_sysCheck[i];
		//		else
		//			outFileStream << 999;
		//		++stringCounter;
		//	}
		//outFileStream.close();

	}

	// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, 1);
	gridSize = (1 + blockSize - 1) / 1;

	// -----------------------------------------
	// --- CUDA ������� ��� ��������� DBSCAN ---
	// -----------------------------------------

	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	double* h_helpfulArray = new double[2 * nPts * nPts];
	int* h_sysCheck = new int[nPts * nPts];;

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_sysCheck, d_sysCheck, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	//dbscan(h_avgPeaks, h_avgIntervals, h_helpfulArray, 0, nPts * nPts, 2 * nPts * nPts, 0, eps, nullptr);

	//dbscanCUDA << <gridSize, blockSize >> > 
	//	(	d_avgPeaks, 				// ������ (����)
	//		nPts * nPts, 				// ���������� ����� � ����� �������
	//		1,							// ���������� ������ (������) � data
	//		d_plugAmountOfPeaks, 		// ������, ���������� ���������� ����� ��� ������� ����� � data
	//		d_avgIntervals, 			// ���������� ���������
	//		d_helpfulArray, 			// ��������������� ������ 
	//		eps, 						// �������
	//		nullptr);					// �������������� ������

	// -----------------------------------------

	// --- �������� �� CUDA ������ ---
	//gpuGlobalErrorCheck();

	// --- ���� ���� ��� ������ �������� ���� ������ ---
	//gpuErrorCheck(cudaDeviceSynchronize());

	// -------------------------------------------------------------------------------------
	// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
	// -------------------------------------------------------------------------------------

	//gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------------------


			// --- ���������� ������ � ���� ---
	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_helpfulArray[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_sysCheck[i] != NAN)
				outFileStream << h_sysCheck[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();



	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	//gpuErrorCheck(cudaFree(d_plugAmountOfPeaks));
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_sysCheck));
	//gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_helpfulArray;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_sysCheck;

	// ---------------------------
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CUDA_dbscan(double* data, double* intervals, int* labels, int* helpfulArray, const int amountOfData, const double eps)
{
	int resultClusters = 0;
	int amountOfClusters = 0;				// ���������� ���������
	int amountOfNegativeClusters = 0;
	int* amountOfNeighbors = new int[1];			// ��������������� ���������� - ������� ���� ������� ������� � �����
	*amountOfNeighbors = 0;
	int* neighbors = new int[amountOfData];			// ��������������� ���������� - ������� ��������� �������

	int* d_amountOfNeighbors;						// ��������������� ���������� - ������� ���� ������� ������� � �����
	int* d_neighbors;								// ��������������� ���������� - ������� ��������� �������

	cudaMalloc((void**)& d_amountOfNeighbors, sizeof(int));
	cudaMalloc((void**)& d_neighbors, sizeof(int) * amountOfData);

	cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

	int amountOfVisitedPoints = 0;

	int blockSize1;			// ���������� ��� �������� ������� �����
	int minGridSize1;		// ���������� ��� �������� ������������ ������� �����
	int gridSize1;			// ���������� ��� �������� �����


	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, amountOfData);

	blockSize1 = blockSize1 > 512 ? 512 : blockSize1;			// �� ��������� ����������� � 512 ������ � �����
	gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

	int blockSize2;			// ���������� ��� �������� ������� �����
	int minGridSize2;		// ���������� ��� �������� ������������ ������� �����
	int gridSize2;			// ���������� ��� �������� �����

	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, amountOfData);

	blockSize2 = blockSize2 > 512 ? 512 : blockSize2;			// �� ��������� ����������� � 512 ������ � �����
	gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;

	// ���� �� ���� ������ ����
	//while (true)
	for (int i = 0; i < amountOfData; i++)
	{
		int* clearIdx = new int[1];
		*clearIdx = -1;

		int* d_clearIdx;

		cudaMalloc((void**)& d_clearIdx, sizeof(int));

		cudaMemcpy(d_clearIdx, clearIdx, sizeof(int), cudaMemcpyHostToDevice);

		CUDA_dbscan_search_fixed_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
			amountOfData, d_clearIdx);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		//gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

		if (*clearIdx == -1)
		{
			CUDA_dbscan_search_clear_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
				amountOfData, d_clearIdx);

			++amountOfClusters;
			resultClusters = amountOfClusters;
			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

			if (*clearIdx == -1)
				break;
		}
		else
		{
			--amountOfNegativeClusters;
			resultClusters = amountOfNegativeClusters;
		}

		*amountOfNeighbors = 0;
		for (size_t i = 0; i < amountOfData; ++i)
			neighbors[i] = 0;

		cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

		CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
			resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, *clearIdx, helpfulArray);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		//gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);


		for (size_t i = 0; i < *amountOfNeighbors; ++i)
		{
			CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
				resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, neighbors[i], helpfulArray);

			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);

			++amountOfVisitedPoints;
		}

		delete[] clearIdx;
		cudaFree(d_clearIdx);
	}

	delete[] amountOfNeighbors;
	delete[] neighbors;

	cudaFree(d_amountOfNeighbors);
	cudaFree(d_neighbors);

}

__host__ void basinsOfAttraction_2(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.9;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];
	//double* h_helpfulArray = new double[nPts * nPts];			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int* d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	int* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	double* d_avgPeaks;
	double* d_avgIntervals;

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPts * nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- �������� ����� � ��������� ������� ---
	outFileStream << std::setprecision(15);

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	int blockSize;			// ���������� ��� �������� ������� �����
	int minGridSize;		// ���������� ��� �������� ������������ ������� �����
	int gridSize;			// ���������� ��� �������� �����

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_helpfulArray + (i * originalNPtsLimiter));			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);			// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}


	CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps);

	
	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	int* h_helpfulArray = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks,		 nPts * nPts * sizeof(double),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- ���������� ��������� ��������� ���������� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_dbscanResult[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	// --- ���������� ������� �������� ����� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- ���������� ������� �������� �������� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- ���������� ������������� ����� ����� ��������� ������� � ���� ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_helpfulArray[i] != NAN)
				outFileStream << h_helpfulArray[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();


	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_helpfulArray;

	// ---------------------------
}




__host__ void TimeDomainCalculation(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const double* ranges,							// ��������� ��������� ����������
	const int* indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	size_t amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	size_t amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	//double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	//int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int* d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	//double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	//int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPts * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	//gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	//std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	//static curandState *states = NULL;

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
		//calculateDiscreteModelCUDA_rand << <gridSize, blockSize >> >
			(	
				nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				nullptr);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------



#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	double* h_data = new double[amountOfPointsInBlock * nPts];

	gpuErrorCheck(cudaMemcpy(h_data, d_data, nPts* amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- �������� ����� � ��������� ������� ---

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	outFileStream << std::setprecision(16);

	size_t stringCounter = 0;
	//outFileStream.open(OUT_FILE_PATH);
	for (size_t k = 0; k < nPts; ++k) {
		for (size_t i = 0; i < amountOfPointsInBlock; ++i) {
			if (outFileStream.is_open())
			{
				if (h_data[i] != NAN)
					outFileStream << h_data[i + k * amountOfPointsInBlock];
				else
					outFileStream << 999;

				//if (stringCounter != 0)
				//	outFileStream << ", ";
				//if (stringCounter == amountOfPointsInBlock-1)
				//{
				//	outFileStream << "\n";
				//	stringCounter = 0;
				//}
				//else
				outFileStream << ", ";
				//outFileStream << h_avgIntervals[i];

	//			++stringCounter;
			}
		} outFileStream << " \n";
	}
	outFileStream.close();
	printf("End\n");
	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	//gpuErrorCheck(cudaFree(d_outPeaks));
	//gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_data;

	//delete[] h_outPeaks;
	//delete[] h_amountOfPeaks;

	// ---------------------------
}
} // old_library