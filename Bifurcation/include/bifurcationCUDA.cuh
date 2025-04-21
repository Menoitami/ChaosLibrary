#ifndef BIFURCATION_CUDA_CUH
#define BIFURCATION_CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

namespace Bifurcation_constants {

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
	std::string		OUT_FILE_PATH);					// Эпсилон для алгоритма DBSCAN 


__global__ void calculateTransTimeCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			semi_result,
	int*			maxValueCheckerArray);

__global__ void calculateDiscreteModelCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			data, 
	double*			semi_result,
	int*			maxValueCheckerArray);

__global__ void peakFinderCUDA(double* data, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h);

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData);
__device__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber);

    
__device__  int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h);

__device__  int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData);

__device__ int loopCalculateDiscreteModel_int(
	double* x, 
	const double* values,
	const double h, 
	const int amountOfIterations, 
	const int amountOfX, 
	int writableVar, 
	const double maxValue, 
	double* data,
	const int startDataIndex);

    __device__  double distance(double x1, double y1, double x2, double y2);
}
#endif // BIFURCATION_CUDA_CUH
