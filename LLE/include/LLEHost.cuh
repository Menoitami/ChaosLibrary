#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"


#include <iomanip>
#include <string>



__host__ void LLE2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	std::string		OUT_FILE_PATH);
