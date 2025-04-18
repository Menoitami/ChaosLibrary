#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaMacros.cuh"

#include <fstream>
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

namespace old_library{
/**  
 * ��������� ��������� �������� ���������� ������
 * � ���������� ��������� � x
 * 
 * \param x			- ��������� ������� ��� �������� ���������� ���������� �����
 * \param values	- ���������
 * \param h			- ��� ��������������
 */
__device__ __host__ void calculateDiscreteModel(double* x, const double* values, const double h);

__device__ void calculateDiscreteModel_rand(size_t seed, double* X, const double* a, const double h);
/**
 * ��������� ���������� ��� ����� ������� � ���������� ��������� � "data" (���� data != nullptr)
 * 
 * \param x						- ��������� ������� ��� �������������
 * \param values				- ��������� �������
 * \param h						- ��� ��������������
 * \param amountOfIterations	- ���������� ��������
 * \param preScaller			- ��������� ���������. ������ 'preScaller' ����� ����� �������� � ���������
 * \param writableVar			- ����� �� ���������� � x[] ����� �������� � ���������
 * \param maxValue				- ������������ ��������. ���� ����� x[writableVar] > maxValue, ����� ������� ������ false
 * \param data					- ������ ��� ������ ������
 * \param startDataIndex		- ������, � �������� ������� �������� ������ � data
 * \param writeStep				- ��� ������ � ������� � ������� (��������, ���� ��� = 2, �� ������ ����� � �������: 0, 2, 4, ...)
 * \return						- ��������� true, ���� ������ �� ���������
 */
__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, 
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller=0,
	const int writableVar = 0, const double maxValue = 0,
	double* data = nullptr, const int startDataIndex = 0, 
	const int writeStep = 1);

//__device__ __host__ int loopCalculateDiscreteModel_int(
__device__ int loopCalculateDiscreteModel_int(
	double* x, const double* values,
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller = 0,
	const int writableVar = 0, const double maxValue = 0,
	double* data = nullptr, const int startDataIndex = 0,
	const int writeStep = 1);

__device__ double loopCalculateDiscreteModelForFastSynchro_int(
	double* x, const double* values,
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller = 0,
	const double maxValue = 0, const int	iterOfSynchr = 1, 
	const double* kForward = nullptr,	const double* kBackward = nullptr,
	double* data = nullptr, const int startDataIndex = 0,
	const int writeStep = 1);


/**
 * ���������� �������, ������� ������������� ��������� ���������� ����� ������
 *
 * \param amountOfThreads			- 
 * \param h							- ��� ��������������
 * \param hSpecial					- 
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \return -
 */

__global__ void distributedCalculateDiscreteModelCUDA(
	const int		amountOfPointsForSkip,
	const int		amountOfThreads,
	const double	h,
	const double	hSpecial,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar = 0,
	double*			data = nullptr);



/**
 * ���������� �������, ������� ��������� ���������� ���������� ������
 * 
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param amountOfPointsForSkip		- ���������� ����� ��� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
 * \return -
 */
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
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			data = nullptr,
	int*			maxValueCheckerArray = nullptr);


// --------------------------------------------------------------------------

__global__ void calculateDiscreteModelforFastSynchroCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const double	h,
	double* initialConditions,
	const int		amountOfInitialConditions,
	const double* values,
	const double* k_forward,
	const double* k_backward,
	const int		iterOfSynchr,
	const int		amountOfValues,
	const int		amountOfIterations,
	const double	maxValue,
	double* timedomain,
	double* output,
	const int		preScaller);

__global__ void calculateDiscreteModelforFastSynchroCUDA_2(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const double	h,
	double* initialConditions,
	const int		amountOfInitialConditions,
	const double* values,
	const double* k_forward,
	const double* k_backward,
	const int		iterOfSynchr,
	const int		amountOfValues,
	const int		amountOfIterations,
	const double	maxValue,
	double* timedomain,
	double* output,
	const int		preScaller);

__device__ double loopCalculateDiscreteModelForFastSynchro(
	double* Xs,
	const double* values,
	const double h,
	const double* K_Forward,
	const double* K_Backward,
	const double iterOfSynchr,
	const int amountOfIterations,
	const int amountOfX,
	const double maxValue,
	double* timedomain,
	const int startDataIndex);

__device__ double loopCalculateDiscreteModelForFastSynchro_2(
	double* Xs,
	const double* values,
	const double h,
	const double* K_Forward,
	const double* K_Backward,
	const double iterOfSynchr,
	const int amountOfIterations,
	const int amountOfX,
	const double maxValue,
	double* timedomain,
	const int startDataIndex);

__global__ void calculateDiscreteModelICCforFastSynchro(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		dimension,
	double* ranges,
	const double	h,
	int* indicesOfMutVars,
	double* initialConditions,
	const int		amountOfInitialConditions,
	const double* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const double	maxValue,
	const int		iterOfSynchr,
	const double*   kForward,
	const double*   kBackward,
	double* data,
	int* maxValueCheckerArray,
	double* FastSynchroError);

/**
 * ���������� �������, ������� ��������� ���������� ���������� ������ �� ����
 *
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param transientTime				- ����� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
 * \return -
 */
__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const double	transientTime,
	const int		dimension,
	double*			ranges,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const double	amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double* data = nullptr,
	int* maxValueCheckerArray = nullptr);

// --------------------------------------------------------------------------



/**
 * ���������� �������, ������� ��������� ���������� ���������� ������ (�� ��������� ��������)
 *
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param amountOfPointsForSkip		- ���������� ����� ��� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
 * \return -
 */
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
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			data = nullptr,
	int*			maxValueCheckerArray = nullptr);



/**
 * �������, ������� ������� ������ � ������������������ ��������
 * ������:
 * ������������������:
 * 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
 * 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 * 
 * getValueByIdx(7, 5, 1, 5, 0) = 3
 * getValueByIdx(7, 5, 1, 5, 1) = 2
 * getValueByIdx(7, 5, 1, 5, 2) = 1
 * 
 * \param idx			- ������� idx � ������
 * \param nPts			- ���������� ����� ��� ��������� ��������� (����������)
 * \param startRange	- ����� ������� ���������
 * \param finishRange	- ������ ������� ���������
 * \param valueNumber	- ����� ��������� ����������
 * \return Value		- ���������
 */
__device__ __host__ double getValueByIdx(const int idx, const int nPts, 
	const double startRange, const double finishRange, const int valueNumber);



/**
 * �������, ������� ������� ������ � ������������������ �������� �� ��������������� �����
 *
 * \param idx			- ������� idx � ������
 * \param nPts			- ���������� ����� ��� ��������� ��������� (����������)
 * \param startRange	- ����� ������� ���������
 * \param finishRange	- ������ ������� ���������
 * \param valueNumber	- ����� ��������� ����������
 * \return Value		- ���������
 */
__device__ __host__ double getValueByIdxLog(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber);



/**
 * ������� ���� � ��������� [startDataIndex; startDataIndex + amountOfPoints] � "data" �������
 * ��������� ������������ � outPeaks � timeOfPeaks ( ���� outPeaks != nullptr � timeOfPeaks != nullptr )
 * 
 * \param data				- ������ � �������
 * \param startDataIndex	- ������, ������ �������� ������ ����
 * \param amountOfPoints	- ���������� ����� ��� ������ �����
 * \param outPeaks			- �������� ������ ��� ������ � ���� ��������� �����
 * \param timeOfPeaks		- �������� ������ ��� ������ � ���� �������� ��������� �����
 * \param h					- ��� �������������� (��� ���������� ����������� ���������)
 * \return - Amount of found peaks
 */
__device__ __host__ int peakFinder(double* data, const int startDataIndex, const int amountOfPoints, 
	double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h = 0);

__device__ __host__ int peakFinder_for_neuronClasses(double* data, const int startDataIndex, const int amountOfPoints,
	double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double* valleys = nullptr, double* timeOfValleys = nullptr, double h = 0);

/**
 * ���������� ����� � "data" ������� � ������������� ������ 
 * ��������� ������������ � "outPeaks", "timeOfPeaks" � "amountOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr � amountOfPeaks != nullptr )
 * 
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param amountOfPeaks		- �������� ������, ���������� ���������� ����� ��� ������� ����� ( ������� )
 * \param outPeaks			- �������� ������, ���������� f��������� ��������� �����
 * \param timeOfPeaks		- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
 */
__global__ void peakFinderCUDA( double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h = 0 );

__global__ void peakFinderCUDA_for_NeuronClassification(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double* valleys = nullptr, double* timeOfValleys = nullptr, double h = 0);

__device__ __host__ void calculateDiscreteModelforFastSynchro(double* X, double* S1, double* K, const double* a, const double h);

__device__ __host__ void calculateDiscreteModelforFastSynchroBackward(double* X, double* S1, double* K, const double* a, const double h);

/**
 * ���������� ����� � "data" ������� � ������������� ������
 * ��������� ������������ � "outPeaks", "timeOfPeaks" � "amountOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr � amountOfPeaks != nullptr )
 *
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param amountOfPeaks		- �������� ������, ���������� ���������� ����� ��� ������� ����� ( ������� )
 * \param outPeaks			- �������� ������, ���������� f��������� ��������� �����
 * \param timeOfPeaks		- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
 */
__global__ void peakFinderCUDA_H(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h = 0);



/**
 * ��������� ���������� ����� ����� �������
 * 
 * \param x1 - x ������ �����
 * \param y1 - y ������ �����
 * \param x2 - x ������ �����
 * \param y2 - y ������ �����
 * \return - ���������
 */
__device__ __host__ double distance(double x1, double y1, double x2, double y2);



/**
 * ������� DBSCAN
 * 
 * \param data					- ������ (����)
 * \param intervals				- ���������� ���������
 * \param helpfulArray			- ��������������� ������
 * \param startDataIndex		- ������, � �������� ����� ������� ������ � outData
 * \param amountOfPeaks			- ������ � ����������� ����� � ������ �������
 * \param sizeOfHelpfulArray	- ������ ���������������� �������
 * \param idx					- ������� idx � ������
 * \param eps					- �������
 * \param outData				- �������������� ������
 */
__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData);

__device__ __host__ int NeuronClassification(double* data, double* intervals, double* valleys, double* TimeOfValleys, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData);


/**
 * ���������� ������� DBSCAN
 * 
 * \param data				- ������ (����)
 * \param sizeOfBlock		- ���������� ����� � ����� �������
 * \param amountOfBlocks	- ���������� ������ (������) � data
 * \param amountOfPeaks		- ������, ���������� ���������� ����� ��� ������� ����� � data
 * \param intervals			- ���������� ���������
 * \param helpfulArray		- ��������������� ������
 * \param eps				- �������
 * \param outData			- �������������� ������
 */
__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray, const double eps, int* outData);

__global__ void NeuronClassificationCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* valleys, double* TimeOfValleys, double* helpfulArray, const double eps, int* outData);



/**
 * ���� ��� LLE
 * 
 * \param nPts						- ����� ����������
 * \param nPtsLimiter				- ���������� � ������� �������
 * \param NT						- ����� ������������
 * \param tMax						- ����� �������������
 * \param sizeOfBlock				- ���������� �����, ���������� ����� �������� � "data"
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� �����
 * \param amountOfPointsForSkip		- ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
 * \param dimension					- �����������
 * \param ranges					- ������, ���������� ��������� ������������� ���������
 * \param h							- ��� ��������������
 * \param eps						- �������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� (����������� �� tMax)
 * \param preScaller				- ��������� ��� ��������� ��������
 * \param writableVar				- ������ ���������� � x[] �� �������� ������ ���������
 * \param maxValue					- ������������� �������� ���������� ��� �������������
 * \param resultArray				- �������������� ������
 * \return -
 */
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			resultArray = nullptr);



/**
 * ���� ��� LLE (IC)
 *
 * \param nPts						- ����� ����������
 * \param nPtsLimiter				- ���������� � ������� �������
 * \param NT						- ����� ������������
 * \param tMax						- ����� �������������
 * \param sizeOfBlock				- ���������� �����, ���������� ����� �������� � "data"
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� �����
 * \param amountOfPointsForSkip		- ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
 * \param dimension					- �����������
 * \param ranges					- ������, ���������� ��������� ������������� ���������
 * \param h							- ��� ��������������
 * \param eps						- �������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� (����������� �� tMax)
 * \param preScaller				- ��������� ��� ��������� ��������
 * \param writableVar				- ������ ���������� � x[] �� �������� ������ ���������
 * \param maxValue					- ������������� �������� ���������� ��� �������������
 * \param resultArray				- �������������� ������
 * \return -
 */
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			resultArray = nullptr);



/**
 * Kernel for metric LS
 *
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* resultArray = nullptr);

/**
 * ���������� �������� �������� ����� � ���������� ���������� � "data" ������� � ������������� ������
 * ��������� ������������ � "outAvgPeaks", "AvgTimeOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr )
 *
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param outAvgPeaks		- �������� ������, ���������� f��������� ��������� �����
 * \param AvgTimeOfPeaks	- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
 */
__global__ void avgPeakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* systemCheker, double h = 0);

__global__ void avgPeakFinderCUDA_for2Dbif(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* PeaksAmount, int* systemCheker, double h = 0);

__global__ void CUDA_dbscan_kernel(double* data, double* intervals, int* labels,
	const int amountOfData, const double eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray);



__global__ void CUDA_dbscan_search_clear_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);



__global__ void CUDA_dbscan_search_fixed_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);

__global__ void CUDA_dbscan_search_unbound_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);


} //old_library