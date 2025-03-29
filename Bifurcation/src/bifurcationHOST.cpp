#include <bifurcationHOST.h>
#include <bifurcationCUDA.cuh>

namespace Bifurcation {
	void bifurcation1D(
		const double	tMax,							
		const int		nPts,							
		const double	h,								
		const int		amountOfInitialConditions,		 
		const double* initialConditions,				
		const double* ranges,							
		const int* indicesOfMutVars,				
		const int		writableVar,					
		const double	maxValue,						
		const double	transientTime,					
		const double* values,							
		const int		amountOfValues,					
		const int		preScaller,
		std::string		OUT_FILE_PATH)						
	{
		// Перенаправляем вызов на CUDA реализацию
		Bifurcation_constants::bifurcation1D(
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
			OUT_FILE_PATH
		);
	}
}