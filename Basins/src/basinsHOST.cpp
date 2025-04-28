#include "basinsHOST.h"
#include "basinsCUDA.cuh"

namespace Basins {
    void basinsOfAttraction_2(
        const double tMax,
        const int nPts,
        const double h,
        const int amountOfInitialConditions,
        const double* initialConditions,
        const double* ranges,
        const int* indicesOfMutVars,
        const int writableVar,
        const double maxValue,
        const double transientTime,
        const double* values,
        const int amountOfValues,
        const int preScaller,
        const double eps,
        std::string OUT_FILE_PATH)
    {
        // Delegate to the CUDA implementation
        basinsGPU::basinsOfAttraction_2(
            tMax,
            nPts,
            h,
            amountOfInitialConditions,
            initialConditions,
            ranges,
            indicesOfMutVars,
            writableVar,
            maxValue,
            transientTime,
            values,
            amountOfValues,
            preScaller,
            eps,
            OUT_FILE_PATH
        );
    }
} 