#include <bifurcationCUDA.cuh>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace Bifurcation_constants {

__global__ void calculateDiscreteModelCUDA(
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
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 


	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, idx * sizeOfBlock, 1);

	// --- ������ ��� ��-��������� ���������� ������� --- 
	if (flag == 1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock, 1);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}

__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if ( amountOfPeaks[idx] == -1 )
	{
		amountOfPeaks[idx] = -1;
		return;
	}

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == 0)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

		amountOfPeaks[idx] = peakFinder( data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h );
	return;
}

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	//outData[idx] = idx;
	//return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == -1)
	{
		outData[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		outData[idx] = 0;
		return;
	}

	// --- ��������� �������� dbscan � ������ �������
	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, 
		idx, eps, outData);
}

__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return startRange + ( ( (int)( (int)idx / pow( (double)nPts, (double)valueNumber) ) % nPts )* ( (double)( finishRange - startRange ) / (double)( nPts - 1 ) ) );
}

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

	// --- ���������� ����, ������� ���������� ���������� �������� amountOfIterations ��� ---
	for (int i = 0; i < amountOfIterations; ++i)
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// --- ���� ���-���� �������� ������ ��� ������ - ���������� �������� ���������� ---


		if (data != nullptr) 
			data[startDataIndex + i * writeStep] = (x[writableVar]);
		
		// --- ���������� ������� preScaller ��� ( �� ���� ���� preScaller > 1, �� �� ��������� ( preScaller - 1 ) � ��������������� ���������� ) ---
		//for (int j = 0; j < preScaller; ++j) {
		//	//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
		//	calculateDiscreteModel(x, values, h);
		//}

		for (int j = 0; j < preScaller - 1; ++j)
			calculateDiscreteModel(x, values, h);

		calculateDiscreteModel(x, values, h);

		// 1 - stability, -1 - fixed point, 0 - unbound solution

		// --- ���� isnan ��� isinf - ���������� false, ��� ��� ������������� ��������� ������� ---
		if (isnan(x[writableVar]) || isinf(x[writableVar]))
		{
			delete[] xPrev;
			return 0;
		}

		// --- ���� maxValue == 0, ��� ������ ������������ �� �������� �����������, ����� ��������� ��� ��������� ---
		if (maxValue != 0)
			if (fabsf(x[writableVar]) > maxValue)
			{
				delete[] xPrev;
				return 0;
			}
	}

	// --- �������� �� ���������� � ����� ---
	double tempResult = 0;

	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
		//tempResult += abs(x[j] - xPrev[j]);
	}


	//if (abs(tempResult) < 1e-8)
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

__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{
	// ------------------------------------------------------------
	// --- ���� ����� 0 ��� 1 - ���� �� ������������ ��� ������ ---
	// ------------------------------------------------------------

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	//if (amountOfPeaks >= 3600)
	//	return 0;


	// ------------------------------------------------------------


	int cluster = 0;
	int NumNeibor = 0;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	// ------------------------------------------------------------
	//for (int i = 0; i < amountOfPeaks; i++) {
	//	helpfulArray[startDataIndex + i] = (int)(100*sqrt(data[startDataIndex + i] * data[startDataIndex + i] + intervals[startDataIndex + i] * intervals[startDataIndex + i]));
	//}

	for (int i = 0; i < amountOfPeaks; i++) {
		data[startDataIndex + i] = 0; 
		//intervals[startDataIndex + i] = intervals[startDataIndex + i];
	}
	// ------------------------------------------------------------

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


__device__ __host__ void calculateDiscreteModel(double* X, const double* a, const double h)
{
	/**
	 * here we abstract from the concept of parameter names. 
	 * ALL parameters are numbered with indices. 
	 * In the current example, the parameters go like this:
	 * 
	 * values[0] - sym
	 * values[1] - A
	 * values[2] - B
	 * values[3] - C
	 */

	// --- Chameleon 02 --- 
	double h1 = a[0] * h;
	double h2 = (1 - a[0]) * h;
	X[0] = X[0] + h1 * (-a[6] * X[1]);
	X[1] = X[1] + h1 * (a[6] * X[0] + a[1] * X[2]);
	X[2] = X[2] + h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1]));

	X[2] = (X[2] + h2 * (a[2] + a[4] * cos(a[5] * X[1]))) / (1 + a[3] * h2);
	X[1] = X[1] + h2 * (a[6] * X[0] + a[1] * X[2]);
	X[0] = X[0] + h2 * (-a[6] * X[1]);

	// --- RLCs-JJ ---
	//double h1 = h * a[0];
	//double h2 = h * (1 - a[0]);
	//double X1;
	//X[0] = X[0] + h1 * (X[1]);
	//X[1] = X[1] + h1 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
	//X[2] = X[2] + h1 * ((1 / a[1]) * (X[1] - X[2]));
	//
	//X1 = X[1];
	//	
	//X[2] = (X[2] + h2 * (1 / a[1]) * X[1]) / (1 + h2 * (1 / a[1]));
	//X[1] = X1 + h2 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
	//X[1] = X1 + h2 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
	//X[0] = X[0] + h2 * (X[1]);

	// --- Lorenz
	//double h1 = a[0] * h;
	//double h2 = (1 - a[0]) * h;
	//X[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]));
	//X[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1]);
	//X[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2]);
	//X[2] = (X[2] + h2 * (X[0] * X[1])) / (1 + h2 * a[3] * a[4]);
	//X[1] = (X[1] + h2 * (X[0] * (a[2] - X[2]))) / (1 + h2 * a[4]);
	//X[0] = (X[0] + h2 * (a[1] * (X[1]))) / (1 + a[1] * a[4] * h2);

}

__device__ __host__ double distance(double x1, double y1, double x2, double y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	double dx = x2 - x1;
	double dy = y2 - y1;

	return hypotf(dx, dy);
}

} // Bifurcation_constants
