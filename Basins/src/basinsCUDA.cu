#include "basinsCUDA.cuh"
#include "cudaMacros.cuh"
#include "systems.cuh"
#include <fstream>
#define DEBUG
namespace basinsGPU {
__constant__ double d_tMax;
__constant__ int d_nPts;
__constant__ double d_h;
__constant__ int d_amountOfInitialConditions;

__constant__ int d_writableVar;
__constant__ double d_maxValue;
__constant__ double d_transientTime;

__constant__ int d_amountOfValues;
__constant__ int d_preScaller;
__constant__ double d_eps;


__constant__ int d_sizeOfBlock;
__constant__ int d_dimension;
__constant__ int d_amountOfIterations;


__constant__ int d_nPtsLimiter;

__constant__ int d_amountOfPointsInBlock;
__constant__ int d_amountOfPointsForSkip;
__constant__ int d_originalNPtsLimiter;

__constant__ int d_amountOfCalculatedPoints;


void CUDA_dbscan(double* data, double* intervals, int* labels, int* helpfulArray, const int amountOfData, const double eps)
{
	int resultClusters = 0;
	int amountOfClusters = 0;				// Количество кластеров
	int amountOfNegativeClusters = 0;
	int* amountOfNeighbors = new int[1];			// Вспомогательная переменная - сколько было найдено соседей у точки
	*amountOfNeighbors = 0;
	int* neighbors = new int[amountOfData];			// Вспомогательная переменная - индексы найденных соседей

	int* d_amountOfNeighbors;						// Вспомогательная переменная - сколько было найдено соседей у точки
	int* d_neighbors;								// Вспомогательная переменная - индексы найденных соседей

	cudaMalloc((void**)& d_amountOfNeighbors, sizeof(int));
	cudaMalloc((void**)& d_neighbors, sizeof(int) * amountOfData);

	cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

	int amountOfVisitedPoints = 0;

	int blockSize1;			
	int minGridSize1;		
	int gridSize1;			


	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, amountOfData);

	blockSize1 = blockSize1 > 512 ? 512 : blockSize1;			
	gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

	int blockSize2;			
	int minGridSize2;		
	int gridSize2;			

	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, amountOfData);

	blockSize2 = blockSize2 > 512 ? 512 : blockSize2;			
	gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;


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

			if (*clearIdx == -1) {
				cudaFree(d_clearIdx); // освобождаем память перед выходом
				delete[] clearIdx;
				break;
			}
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

		cudaFree(d_clearIdx); // освобождаем память после использования
		delete[] clearIdx;
	}

	delete[] amountOfNeighbors;
	delete[] neighbors;

	cudaFree(d_amountOfNeighbors);
	cudaFree(d_neighbors);

}
__device__ void calculateDiscreteModel(double* X, const double* a, const double h)
{
	CALC_DISCRETE_MODEL(X,a,h);
}

__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return startRange + ( ( (int)( (int)idx / pow( (double)nPts, (double)valueNumber) ) % nPts )* ( (double)( finishRange - startRange ) / (double)( nPts - 1 ) ) );
}


__global__ void calculateTransTimeCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			semi_result,
	int*			maxValueCheckerArray)
{
	
	extern __shared__ double s[];

	double* localX = s + ( threadIdx.x * d_amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * d_amountOfInitialConditions ) + ( threadIdx.x * d_amountOfValues );

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= d_nPtsLimiter)	return;

	for ( int i = 0; i < d_amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	for (int i = 0; i < d_amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < d_dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( d_amountOfCalculatedPoints + idx, 
			d_nPts, ranges[i * 2], ranges[i * 2 + 1], i );
	
	int flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfPointsForSkip,
		d_amountOfInitialConditions, 1, 0, 0, nullptr, idx * d_sizeOfBlock);

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}


	#pragma unroll
	for (int i = 0; i < d_amountOfInitialConditions; ++i){	
		semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + i] = localX[i];
	}
	#pragma unroll
	for (int i = 0; i < d_amountOfValues; ++i)
		semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + d_amountOfInitialConditions + i] = localValues[i];

}

__global__ void calculateDiscreteModelCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			data, 
	double*			semi_result,
	int*			maxValueCheckerArray)
{
	extern __shared__ double s[];

	double* localX = s + ( threadIdx.x * d_amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * d_amountOfInitialConditions ) + ( threadIdx.x * d_amountOfValues );

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= d_nPtsLimiter)	return;

	for ( int i = 0; i < d_amountOfInitialConditions; ++i )
		localX[i] = semi_result[idx * (SIZE_X + SIZE_A) + i];

	for (int i = 0; i < d_amountOfValues; ++i)
		localValues[i] = semi_result[idx * (SIZE_X + SIZE_A) + SIZE_X + i];

	for (int i = 0; i < d_dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( d_amountOfCalculatedPoints + idx, 
			d_nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	int flag =0;

	if (maxValueCheckerArray[idx] == 1 || maxValueCheckerArray[idx] == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfIterations,
			d_amountOfInitialConditions, d_preScaller, d_writableVar, d_maxValue, data, idx * d_sizeOfBlock);

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}
	

	return;
}


//__device__ __host__ int loopCalculateDiscreteModel_int(
__device__ int loopCalculateDiscreteModel_int(
	double* x, 
	const double* values,
	const double h, 
	const int amountOfIterations, 
	const int amountOfX, 
	const int preScaller,
	int writableVar, 
	const double maxValue, 
	double* data,
	const int startDataIndex, 
	const int writeStep)
{
	double* xPrev = new double[amountOfX];

	for (int i = 0; i < amountOfIterations; ++i)
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}


		if (data != nullptr) 
			data[startDataIndex + i * writeStep] = (x[writableVar]);


		for (int j = 0; j < preScaller - 1; ++j)
			calculateDiscreteModel(x, values, h);

		calculateDiscreteModel(x, values, h);

		if (isnan(x[writableVar]) || isinf(x[writableVar]))
		{
			delete[] xPrev;
			return 0;
		}

		if (maxValue != 0)
			if (fabsf(x[writableVar]) > maxValue)
			{
				delete[] xPrev;
				return 0;
			}
	}

	double tempResult = 0;

	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	}


	if (sqrt(abs(tempResult)) < 1e-9)
	{
		delete[] xPrev;
		return -1;
	}

	delete[] xPrev;
	return 1;
}

__device__ __host__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- ���������� ��� �������� ��������� ����� ---
	int amountOfPeaks = 0;

	// --- �������� ������������� �������� �������� �� ������� ����� ---
	for ( int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i )
	{
		// --- ���� ������� ����� ������ ���������� � ������ ��� ����� ���������, ��... ( �� ����, ��� ��� ��� ( ��������: 2 3 3 4 ) ) ---
		if ( data[i] - data[i - 1] > 1e-13 && data[i] >= data[i + 1] ) //&&data[j] > 0.2
		{
			// --- �� ��������� ����� �������� ���� ������, ���� �� ��������� �� ����� ������ ������ ��� ������ ---
			for ( int j = i; j < startDataIndex + amountOfPoints - 1; ++j )
			{
				// --- ���� ���������� �� ����� ������ ������, ������ ��� ��� �� ��� ---
				if ( data[j] < data[j + 1] )
				{
					i = j + 1;	// --- ��������� ������� �������, ����� ������ �� ��������� ���� � ��� �� ��������
					break;		// --- ������������ � �������� �����
				}
				// --- ���� � ����, �� ����� ����� ������, ��� �������, ������ �� ����� ��� ---
				if ( data[j] - data[j + 1] > 1e-13  ) //&&data[j] > 0.2
				{
					// --- ���� ������ outPeaks �� ����, �� ������ ������ ---
					if ( outPeaks != nullptr )
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					// --- ���� ������ timeOfPeaks �� ����, �� ������ ������ ---
					if ( timeOfPeaks != nullptr )
						timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (double)j + (double)i ) / (double)2 );	// �������� ������ ���������� ����� j � i
					++amountOfPeaks;
					i = j + 1; // ������ ��� ��������� ����� ����� �� ����� ���� ����� ( ��� ���� �� ����� ���� ������ )
					break;
				}
			}
		}
	}
	// --- ��������� ���������� ��������� ---
	if ( amountOfPeaks > 1 ) {
		// --- ����������� �� ���� ��������� ����� � �� �������� ---
		for ( size_t i = 0; i < amountOfPeaks - 1; i++ )
		{
			// --- ������� ��� ���� �� ���� ������ �����, � ������ ��� ������� ---
			if ( outPeaks != nullptr )
				outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
			// --- ��������� ���������� ��������. ��� ������� ������� ���������� ����� � �����������, ���������� �� ��� ---
			if ( timeOfPeaks != nullptr )
				timeOfPeaks[startDataIndex + i] = (double)( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h;
		}
		// --- ��� ��� ���� ��� ������� - �������� ������� �� ���������� ---
		amountOfPeaks = amountOfPeaks - 1;
	}
	else {
		amountOfPeaks = 0;
	}


	return amountOfPeaks;
}

__host__ void basinsOfAttraction_2(
	const double	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,					// Массив с начальными условиями
	const double* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const double	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	int* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	double* d_avgPeaks;
	double* d_avgIntervals;


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

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_nPts, &nPts, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfIterations, &amountOfIteration, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfInitialConditions, &amountOfInitialConditions, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfValues, &amountOfValues, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_nPtsLimiter, &nPtsLimiter, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsInBlock, &amountOfPointsInBlock, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsForSkip, &amountOfPointsForSkip, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_writableVar, &writableVar, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_maxValue, &maxValue, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_preScaller, &preScaller, sizeof(int)));

	int dimension = 2;
	gpuErrorCheck(cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(int)));	
	
	


	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(15);

	// --- Выводим в самое начало файла исследуемые диапазон ---
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



	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------
		int calculatedPoints = i * originalNPtsLimiter;
		gpuErrorCheck(cudaMemcpyToSymbol(d_nPtsLimiter, &nPtsLimiter, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalculatedPoints, &calculatedPoints, sizeof(int)));

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_helpfulArray + (i * originalNPtsLimiter));			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());
		
		// int calculatedPoints = i * originalNPtsLimiter;
		// gpuErrorCheck(cudaMemcpyToSymbol(d_nPtsLimiter, &nPtsLimiter, sizeof(int)));
		// gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalculatedPoints, &calculatedPoints, sizeof(int)));

		// double* d_semi_result;
		// gpuErrorCheck(cudaMalloc((void**)& d_semi_result, nPtsLimiter * (amountOfInitialConditions + amountOfValues) * sizeof(double)));

		// calculateTransTimeCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >(
		// 				d_ranges,
		// 				d_indicesOfMutVars,
		// 				d_initialConditions,
		// 				d_values,
		// 				d_semi_result,
		// 				d_amountOfPeaks);
		// cudaDeviceSynchronize();

		// calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >(
		// 				d_ranges,
		// 				d_indicesOfMutVars,
		// 				d_initialConditions,
		// 				d_values,
		// 				d_data,
		// 				d_semi_result,
		// 				d_amountOfPeaks);

		// gpuGlobalErrorCheck();

		// gpuErrorCheck(cudaDeviceSynchronize());
		// cudaFree(d_semi_result);

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


		avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						
				amountOfPointsInBlock,		
				nPtsLimiter,				
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						
				d_intervals,				
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);			
		gpuGlobalErrorCheck();

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

			if (h_helpfulArray[i] != NAN)
				outFileStream << h_helpfulArray[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();



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



__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const int		amountOfPointsForSkip,
	const int		dimension, 
	double*			ranges, 
	const double	h,
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const int		amountOfInitialConditions, 
	const double*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- ����� ������ � ������ ������ ����� ---
	// --- �������� ������: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ��������� �����...} ---
	extern __shared__ double s[];

	// --- � ������ ������ ������� ��������� �� ��������� � ����������, ����� �������� � ���� ��� � ��������� ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���������� localX[] ���������� ��������� ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- ���������� localValues[] ���������� ����������� ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- ������ �������� ���������� ���������� �� ��������� ������� getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	//__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	//	const double startRange, const double finishRange, const int valueNumber)
	//{
	//	return startRange + (((int)((int)idx / powf((double)nPts, (double)valueNumber)) % nPts)
	//		* ((double)(finishRange - startRange) / (double)(nPts - 1)));
	//}

	//// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	//bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
	//	amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	//// --- ������ ��� ��-��������� ���������� ������� --- 
	//if (flag)
	//	flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
	//	amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	//// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	//if (!flag && maxValueCheckerArray != nullptr)
	//	maxValueCheckerArray[idx] = -1;
	//else
	//	maxValueCheckerArray[idx] = 1;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	
	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- ������ ��� ��-��������� ���������� ������� --- 
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}
__global__ void avgPeakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* systemCheker, double h)
{
	// ---   ,     ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		//     ,   -   
		return;

	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock-1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}


	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;


	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0) 
	{
		outAvgPeaks[idx] = 1000;
		AvgTimeOfPeaks[idx] = 1000;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}

	outAvgPeaks[idx] /= amountOfPeaks;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	return;
}

__global__ void CUDA_dbscan_search_clear_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
	if (idx >= amountOfData)								//    -   
		return;

	if (labels[idx] == 0 && helpfulArray[idx] == 1)
	{
		*res = idx;
		return;
	}
}

__global__ void CUDA_dbscan_search_fixed_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
	if (idx >= amountOfData)								//    -   
		return;

	if (helpfulArray[idx] == -1 && labels[idx] == 0)
	{
		*res = idx;
		return;
	}
}

__global__ void CUDA_dbscan_kernel(double* data, double* intervals, int* labels,
	const int amountOfData, const double eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
	if (idx >= amountOfData)								//    -   
		return;



	labels[idxCurPoint] = amountOfClusters;

	if (labels[idx] != 0)									  
		return;

	if (idx == idxCurPoint)									   
		return;
 

	if (helpfulArray[idxCurPoint] == 0) {
		labels[idxCurPoint] = 0;
		return;
	}

	if (sqrt((data[idxCurPoint] - data[idx]) * (data[idxCurPoint] - data[idx]) + (intervals[idxCurPoint] - intervals[idx]) * (intervals[idxCurPoint] - intervals[idx])) <= eps)
	{
		labels[idx] = labels[idxCurPoint];						     
		neighbors[atomicAdd(amountOfNeighbors, 1)] = idx;		 
	}
}

} 
