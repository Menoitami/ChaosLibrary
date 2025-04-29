#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

namespace basinsGPU {
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
        std::string		OUT_FILE_PATH);	                  // Путь к выходному файлу

    __global__ void calculateDiscreteModelICCUDA(
	double*			ranges, 
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const double*	values, 
	double*			data, 
	int*			maxValueCheckerArray);

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

    
    __global__ void avgPeakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* systemCheker, double h = 0);

    __device__ __host__ double getValueByIdx(
        const int idx,
        const int nPts,
        const double startRange,
        const double finishRange,
        const int valueNumber);


    __device__ int loopCalculateDiscreteModel_int(
	double* x, const double* values,
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller = 0,
	const int writableVar = 0, const double maxValue = 0,
	double* data = nullptr, const int startDataIndex = 0,
	const int writeStep = 1);
        
    // Function for finding peaks in time series data
    __device__ int peakFinder(
        double* data, 
        const int startDataIndex, 
        const int amountOfPoints, 
        double* outPeaks, 
        double* timeOfPeaks, 
        double h = 0);


    __global__ void CUDA_dbscan_kernel(double* data, double* intervals, int* labels,
        const int amountOfData, const double eps, int amountOfClusters,
        int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray);



    __global__ void CUDA_dbscan_search_clear_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
        const int amountOfData, int* res);



    __global__ void CUDA_dbscan_search_fixed_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
        const int amountOfData, int* res);


} 