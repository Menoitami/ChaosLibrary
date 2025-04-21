#include <bifurcationCUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <systems.cuh>

namespace Bifurcation_constants {
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

__constant__ double d_init_conditions[SIZE_X];
__constant__ double d_values[SIZE_A];


__host__ void bifurcation2D(
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
	int amountOfPointsInBlock = tMax / h / preScaller;

	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											
	size_t totalMemory;											
	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	
	freeMemory *= 0.95;				
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);
	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;
	size_t originalNPtsLimiter = nPtsLimiter;				

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(int)];



	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив


	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);


	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);
	outFileStream << ranges[0] << " " << ranges[1] << "\n";
	outFileStream << ranges[2] << " " << ranges[3] << "\n";

	int stringCounter = 0; 


	gpuErrorCheck(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_nPts, &nPts, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfInitialConditions, &amountOfInitialConditions, sizeof(int)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_writableVar, &writableVar, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_maxValue, &maxValue, sizeof(double)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_transientTime, &transientTime, sizeof(double)));

	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfValues, &amountOfValues, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_preScaller, &preScaller, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_eps, &eps, sizeof(double)));
	
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsInBlock, &amountOfPointsInBlock, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_originalNPtsLimiter, &originalNPtsLimiter, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsForSkip, &amountOfPointsForSkip, sizeof(int)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_init_conditions, &initialConditions, SIZE_X * sizeof(double)));
	//gpuErrorCheck(cudaMemcpyToSymbol(d_values, &values, amountOfValues * sizeof(double)));

	int dimension = 2;
	gpuErrorCheck(cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(int)));


	for (int i = 0; i < amountOfIteration; ++i)
	{

		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			
		int minGridSize;		
		int gridSize;			
		
		blockSize = 160;

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		int calculatedPoints = i * originalNPtsLimiter;
		gpuErrorCheck(cudaMemcpyToSymbol(d_nPtsLimiter, &nPtsLimiter, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalculatedPoints, &calculatedPoints, sizeof(int)));

		double* d_semi_result;
		gpuErrorCheck(cudaMalloc((void**)& d_semi_result, nPtsLimiter * (amountOfInitialConditions + amountOfValues) * sizeof(double)));

		calculateTransTimeCUDA << <gridSize, blockSize >> >(
						d_ranges,
						d_indicesOfMutVars,
						d_initialConditions,
						d_values,
						d_semi_result,
						d_amountOfPeaks);
		cudaDeviceSynchronize();

		// Затем строим модель только для точек с флагом 1
		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >(
						d_ranges,
						d_indicesOfMutVars,
						d_initialConditions,
						d_values,
						d_data,
						d_semi_result,
						d_amountOfPeaks);


		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());
		cudaFree(d_semi_result);

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,					
				d_amountOfPeaks,			
				d_data,						
				d_intervals,				
				h * (double)preScaller);	

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		dbscanCUDA << <gridSize, blockSize >> > (	
				d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks,
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult);

		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		outFileStream << std::setprecision(12);

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
				exit(1);
			}
	}


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

__global__ void calculateTransTimeCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			semi_result,
	int*			maxValueCheckerArray)
{

	double localX[SIZE_X];
	double localValues[SIZE_A];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= d_nPtsLimiter)		
			return;

	#pragma unroll
	for ( int i = 0; i < d_amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	#pragma unroll
	for (int i = 0; i < d_amountOfValues; ++i)
		localValues[i] = values[i];

	#pragma unroll
	for (int i = 0; i < d_dimension; ++i) {
		double divisor;
		switch(i) {
			case 0: divisor = 1.0; break;
			case 1: divisor = d_nPts; break;
			case 2: divisor = d_nPts * d_nPts; break;
			default: divisor = __powf(d_nPts, i);
		}
		
		int normalizedIdx = ((d_amountOfCalculatedPoints + idx) / (int)divisor) % d_nPts;
		double scale = (ranges[i * 2 + 1] - ranges[i * 2]) / (d_nPts - 1);
		localValues[indicesOfMutVars[i]] = ranges[i * 2] + normalizedIdx * scale;
	}


	maxValueCheckerArray[idx] = 1;
	#pragma unroll  4
	for (int i = 0; i < d_amountOfPointsForSkip; ++i) {
		CALC_DISCRETE_MODEL(localX, localValues, d_h);
	}
	
	if (d_maxValue != 0 && maxValueCheckerArray[idx]) {
			maxValueCheckerArray[idx] = (localX[d_writableVar] <= d_maxValue && localX[d_writableVar] >= -d_maxValue);
	}
	#pragma unroll
	for (int i = 0; i < SIZE_X; ++i)	
		semi_result[idx * (SIZE_X + SIZE_A) + i] = localX[i];
	#pragma unroll
	for (int i = 0; i < SIZE_A; ++i)
		semi_result[idx * (SIZE_X + SIZE_A) + SIZE_X + i] = localValues[i];

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
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= d_nPtsLimiter)		
			return;

	// double localX[SIZE_X];
	// double localValues[SIZE_A];
	extern __shared__ double s[];

	double* localX = s + ( threadIdx.x * d_amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * d_amountOfInitialConditions ) + ( threadIdx.x * d_amountOfValues );
	
	for ( int i = 0; i < d_amountOfInitialConditions; ++i )
		localX[i] = semi_result[idx * (SIZE_X + SIZE_A) + i];

	for (int i = 0; i < d_amountOfValues; ++i)
		localValues[i] = semi_result[idx * (SIZE_X + SIZE_A) + SIZE_X + i];

	for (int i = 0; i < d_dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(d_amountOfCalculatedPoints + idx, 
			d_nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	int flag =0;

	if (maxValueCheckerArray[idx] == 1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfPointsInBlock,
			d_amountOfInitialConditions, d_writableVar, d_maxValue, data, idx * d_amountOfPointsInBlock);


	return;
}

__global__ void peakFinderCUDA(
    double* data, 
    int* amountOfPeaks, 
    double* outPeaks, 
    double* timeOfPeaks, 
    double h)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= d_nPtsLimiter)
		return;

	if (amountOfPeaks[idx] == -1) {
		amountOfPeaks[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0) {
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * d_amountOfPointsInBlock, d_amountOfPointsInBlock, outPeaks, timeOfPeaks, h);
	return;
}

__global__ void dbscanCUDA(
    double* data, 
    const int sizeOfBlock, 
    const int amountOfBlocks,
    const int* amountOfPeaks, 
    double* intervals, 
    double* helpfulArray,
    const double eps, 
    int* outData)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1) {
		outData[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0) {
		outData[idx] = 0;
		return;
	}

	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
}

__device__ double getValueByIdx(
    const int idx, 
    const int nPts,
    const double startRange, 
    const double finishRange, 
    const int valueNumber)
{
    // Предварительно вычисляем степень
    double divisor;
    switch(valueNumber) {
        case 0: divisor = 1.0; break;
        case 1: divisor = nPts; break;
        case 2: divisor = nPts * nPts; break;
        default: divisor = __powf(nPts, valueNumber);
    }
    
    int normalizedIdx = (idx / (int)divisor) % nPts;
    double scale = (finishRange - startRange) / (nPts - 1);
    return startRange + normalizedIdx * scale;
}

__device__ int loopCalculateDiscreteModel_int(
    double* x, 
    const double* values,
    const double h, 
    const int amountOfIterations, 
    const int amountOfX, 
    int writableVar, 
    const double maxValue, 
    double* data,
    const int startDataIndex)
{
	const int MAX_REG_SIZE = 12;
	double xPrev[MAX_REG_SIZE];

	for (int i = 0; i < amountOfIterations; ++i) {
		for (int j = 0; j < amountOfX; ++j) {
			xPrev[j] = x[j];
		}

		if (data != nullptr) 
			data[startDataIndex + i] = (x[writableVar]);

		CALC_DISCRETE_MODEL(x, values, h);

		double val = x[writableVar];
		if (val != val || val == val + 1.0) {
			return 0;
		}

		if (maxValue != 0)
			if (fabsf(x[writableVar]) > maxValue) {
				return 0;
			}
	}

	double tempResult = 0;

	for (int j = 0; j < amountOfX; ++j) {
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	}

	if (sqrtf(fabsf(tempResult)) < 1e-9) {
		return -1;
	}

	return 1;
}

__device__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)
{
	int amountOfPeaks = 0;
	
	for ( int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i )
	{
		if ( data[i] - data[i - 1] > 1e-13 && data[i] >= data[i + 1] )
		{
			for ( int j = i; j < startDataIndex + amountOfPoints - 1; ++j )
			{
				if ( data[j] < data[j + 1] )
				{
					i = j + 1;
					break;
				}
				if ( data[j] - data[j + 1] > 1e-13  )
				{
					if ( outPeaks != nullptr )
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					if ( timeOfPeaks != nullptr )
						timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (double)j + (double)i ) / (double)2 );
					++amountOfPeaks;
					i = j + 1;
					break;
				}
			}
		}
	}
	if ( amountOfPeaks > 1 ) {
		for ( size_t i = 0; i < amountOfPeaks - 1; i++ )
		{
			if ( outPeaks != nullptr )
				outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
			if ( timeOfPeaks != nullptr )
				timeOfPeaks[startDataIndex + i] = (double)( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h;
		}
		amountOfPeaks = amountOfPeaks - 1;
	}
	else {
		amountOfPeaks = 0;
	}


	return amountOfPeaks;
}

__device__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	int cluster = 0;
	int NumNeibor = 0;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	for (int i = 0; i < amountOfPeaks; i++) {
		data[startDataIndex + i] = 0; 
	}

	for (int i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1];
			helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}
		else if (helpfulArray[startDataIndex + i] == 0) {
			NumNeibor = 0;
			++cluster;
			helpfulArray[startDataIndex + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}

	return cluster - 1;
}


__device__ double distance(double x1, double y1, double x2, double y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	double dx = x2 - x1;
	double dy = y2 - y1;

	return hypotf(dx, dy);
}

} // Bifurcation_constants
