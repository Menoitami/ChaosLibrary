#ifndef SYSTEMS_CUH
#define SYSTEMS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace chameleon_1_cd {
    __device__ void calculateDiscreteModel(double* X, const double* a, const double h);
}

namespace rossler_cd {
    __device__ void calculateDiscreteModel(double* X, const double* a, const double h);
}

#ifndef SELECTED_SYSTEM_NAMESPACE
#define SELECTED_SYSTEM_NAMESPACE chameleon_1_cd
#endif

#define SYSTEM_CALCULATE_MODEL(x, a, h) SELECTED_SYSTEM_NAMESPACE::calculateDiscreteModel(x, a, h)

#endif // SYSTEMS_CUH 