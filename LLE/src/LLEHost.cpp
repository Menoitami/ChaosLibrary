#include "LLEHost.h"
#include "LLECUDA.cuh"
#include <iostream>
#include <stdexcept>

namespace LLE {

void LLE2D(
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
	std::string		OUT_FILE_PATH)
{
    LLE_constants::LLE2D(
        tMax,
        NT,
        nPts,
        h,
        eps,
        initialConditions,
        amountOfInitialConditions,
        ranges,
        indicesOfMutVars,
        writableVar,
        maxValue,
        transientTime,
        values,
        amountOfValues,
        OUT_FILE_PATH
        );

}

} // namespace LLE_constants 