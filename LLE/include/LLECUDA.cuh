#ifndef LLECUDA_CUH
#define LLECUDA_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"
#include <iomanip>
#include <string>
const int size_params = 8;

struct Calc_block{
double init[size_params];
double params[size_params];
double result[size_params];
};
namespace LLE_constants{
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

__device__ void calculateDiscreteModel(double *x, const double *a, const double h);

__device__ void loopCalculateDiscreteModel(double *X, const double *a,
                                                    const int amountOfIterations);

__global__ void calculateSystem(
	double* X,
    double* params,
	const double *paramLinspaceA,
	const double *paramLinspaceB,
    double **result
);

__global__ void calculateBlocks(
	Calc_block ***calculatedBlocks,
	double **result
);


__host__ void LLE2D(
	const double tMax,
	const double NT,
	const double h,
	const double eps,
	const double transientTime,
	const double* initialConditions,
	const int amount_init,
	const double* params,
	const int amount_params,
	const double* linspaceA_params,
	const double* linspaceB_params,
	const int* indicesOfMutVars,
	std::string		OUT_FILE_PATH);

}


#endif