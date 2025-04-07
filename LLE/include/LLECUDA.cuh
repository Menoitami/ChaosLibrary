#ifndef LLECUDA_CUH
#define LLECUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

const int size_params = 8;

namespace LLE_constants {

// Константы CUDA для GPU
__constant__ double d_tMax;
__constant__ double d_transTime;
__constant__ double d_h;

__constant__ int d_size_linspace_A;
__constant__ int d_size_linspace_B;

__constant__ int d_amountOfTransPoints;
__constant__ int d_amountOfNTPoints;
__constant__ int d_amountOfAllpoints;
__constant__ int d_amountOfCalcBlocks;

__constant__ int d_Nt_steps; 

__constant__ int d_paramsSize;
__constant__ int d_XSize;

__constant__ int d_idxParamA;
__constant__ int d_idxParamB;
__constant__ double d_eps;

__device__ int d_progress; 

__constant__ double d_h1;
__constant__ double d_h2;

// Функции CUDA-ядер
__global__ void calculateTransTime(
    double* X,
    double* params,
    const double* paramLinspaceA,
    const double* paramLinspaceB,
    double* semi_result
);

__global__ void calculateSystem(
    double* X,
    double* params,
    const double* paramLinspaceA,
    const double* paramLinspaceB,
    double* semi_result,
    double** result
);

// Device функции
__device__ void loopCalculateDiscreteModel(
    double *X,
    const double *a,
    const int amountOfIterations
);

__device__ void calculateDiscreteModel(
    double *x,
    const double *a,
    const double h
);

// Host функции
__host__ double* linspace(double start, double end, int num);

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
	std::string		OUT_FILE_PATH
);

} // namespace LLE_constants

#endif // LLECUDA_CUH