#include <LLECUDA.cuh>

namespace LLE_constants{
	
__device__ __host__ void calculateDiscreteModel(double *x, const double *a, const double h)
{
    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);
}

__device__ __host__ bool loopCalculateDiscreteModel(double *x, const double *params,
                                                    const int amountOfIterations)
{
    for (int i = 0; i < amountOfIterations; ++i)
    {
        calculateDiscreteModel(x, params, d_h);
    }
    return true;
}


__global__ void calculateSystem(
	const double *X,
	const double *params,
	const double *paramLinspaceA,
	const double *paramLinspaceB,
	Calc_block ***calculatedBlocks
){
    
	const int idx_a = threadIdx.x + blockIdx.x * blockDim.x;
	const int idx_b = threadIdx.y + blockIdx.y * blockDim.y;
    Calc_block* local_blocks = calculatedBlocks[idx_a][idx_b];

	double local_params[32];
	double local_X[32];
	memcpy(local_X, X,d_XSize * sizeof(double));
	memcpy(local_params, params, d_paramsSize * sizeof(double));

	local_params[d_idxParamA] = paramLinspaceA[idx_a];
    local_params[d_idxParamB] = paramLinspaceB[idx_b];
  
	loopCalculateDiscreteModel(local_X, local_params,
                               d_amountOfTransPoints);

    int nt = 0;
    
    for (int i = 0; i < d_amountOfAllpoints; ++i) {
        calculateDiscreteModel(local_X, local_params, d_h);
        if ((i / d_Nt) % 10 == 0) {
            int index = nt;
            local_blocks[index].init = local_X;
            local_blocks[index].params = local_params;
            if (index > 0) {
                local_blocks[index - 1].result = local_X;
            }

            nt++;
        }
    }

}

} //LLE_constants