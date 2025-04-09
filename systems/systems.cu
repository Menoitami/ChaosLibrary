#include "systems.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvrtc.h>
#include <cooperative_groups.h>

namespace chameleon_1_cd {
    __device__ void calculateDiscreteModel(double* X, const double* a, const double h)
{
	double h1 = a[0] * h;
	double h2 = (1 - a[0]) * h;
	double cos_term = cosf(a[5] * X[1]);
	X[0] = __fma_rn(h1, (-a[6] * X[1]), X[0]); 
	X[1] = __fma_rn(h1, (a[6] * X[0] + a[1] * X[2]), X[1]); 
	X[2] = __fma_rn(h1, (a[2] - a[3] * X[2] + a[4] * cos_term), X[2]); 

	float inv_den = __frcp_rn(__fmaf_rn(a[3], h2, 1.0f));   

	// Вторая фаза
	X[2] = __fma_rn(h2, (a[2] + a[4] * cos_term), X[2]) * inv_den; 
	X[1] = __fma_rn(h2, (a[6] * X[0] + a[1] * X[2]), X[1]);
	X[0] = __fma_rn(h2, (-a[6] * X[1]), X[0]);        
}

}

namespace rossler_cd {
    __device__ void calculateDiscreteModel(double* X, const double* a, const double h) {
		double h1 = 0.5 * h + a[0];
		double h2 = 0.5 * h - a[0];

		
		X[0] += h1 * (-X[1] - X[2]);
		X[1] += h1 * (X[0] + a[1] * X[1]);
		X[2] += h1 * (a[2] + X[2] * (X[0] - a[3]));

		X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));
		X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);
		X[0] += h2 * (-X[1] - X[2]);
    }
}
