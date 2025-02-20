


struct Calc_block{
double* init;
double* params;
double* result;

double* final_num;
};
namespace LLE_constants{
__constant__ double d_init;
__constant__ double d_params; 
__constant__ double d_tMax;
__constant__ double d_transTime;
__constant__ double d_h;


__constant__ int d_amountOfTransPoints;
__constant__ int d_amountOfNTPoints;
__constant__ int d_amountOfAllpoints;

__constant__ int d_Nt; 

__constant__ int d_paramsSize;
__constant__ int d_XSize;

__constant__ int d_idxParamA;
__constant__ int d_idxParamB;

__device__ int d_progress; 

}

__device__ __host__ void calculateDiscreteModel(double *x, const double *a, const double h);

__device__ __host__ bool loopCalculateDiscreteModel(double *x, const double *params,
                                                    const int amountOfIterations);

__global__ void calculateSystem(
	const double *  ,
	const double *paramLinspaceB,
	Calc_block *calculatedBlocks
);
