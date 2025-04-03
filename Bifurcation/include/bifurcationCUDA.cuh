#ifndef BIFURCATION_CUDA_CUH
#define BIFURCATION_CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

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
	int*			maxValueCheckerArray);

__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h);

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData);
__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber);

    
__device__ __host__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h);

__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData);

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
	const int writeStep);

    __device__ __host__ void calculateDiscreteModel(double* X, const double* a, const double h);

    __device__ __host__ double distance(double x1, double y1, double x2, double y2);
}
#endif // BIFURCATION_CUDA_CUH
