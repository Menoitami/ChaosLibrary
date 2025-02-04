#include "bifurcationHOST.h"
#include <iostream>

#define gpuErrorCheck(call)                                                 \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " in file " << __FILE__ << " at line " << __LINE__ \
                      << std::endl;                                         \
            exit(err);                                                      \
        }                                                                   \
    }



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
	// ---  ,          ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// ---  ,       ---
	// --- (amountOfPointsForSkip        ) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											//        GPU
	size_t totalMemory;											//        GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	//       GPU

	freeMemory *= 0.9;											//   (     GPU )		

	// ---   ,          ---
	// TODO    
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	//      ,  ,      (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				//    nPts    ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// ---       (      ) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	double* h_outIntervals = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// ---      GPU ---
	// -----------------------------------------

	double* d_data;					//      GPU    
	double* d_ranges;				//       
	int* d_indicesOfMutVars;		//          values
	double* d_initialConditions;	//      
	double* d_values;				//     

	double* d_outPeaks;				//     GPU    . 
	int* d_amountOfPeaks;		//     GPU  -    .

	//    
	double* d_intervals;			//     GPU    
	
	int* d_sysCheker;			//     GPU   
	double* d_avgPeaks;
	double* d_avgIntervals;

	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * sizeof(double)));


	// -----------------------------------------

	// -----------------------------
	// ---    GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// ---       GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ---        ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// ---       ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// ---  ,   amountOfIteration     nPtsLimiter  ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// ---     ,   nPtsLimiter     ---
		// ---    ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			//     
		int minGridSize;		//      
		int gridSize;			//    

		// --- ,         48  ---
		// ---      (amountOfInitialConditions + amountOfValues) * sizeof(double)  ---
		// ---  ,          ---
		// --- ,        1024  ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		//     1024   

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	//    (    ceil() )

		// --------------------------------------------------
		// --- CUDA      ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(
				nPts,						//    - nPts
				nPtsLimiter,				//  ,      - nPtsLimiter
				amountOfPointsInBlock,		//      ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	//     
				amountOfPointsForSkip,		//     ( transientTime )
				1,							//  (   )
				d_ranges,					//   
				h,							//  
				d_indicesOfMutVars,			//   
				d_initialConditions,		//  
				amountOfInitialConditions,	//   
				d_values,					// 
				amountOfValues,				//  
				amountOfPointsInBlock,		//   (       )
				preScaller,					// ,      
				writableVar,				//  ,     
				maxValue,					//   ( ),     ""
				d_data,						// ,     
				d_amountOfPeaks);			//  ,       '-1'   

		//calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
		//	(nPts,						//    - nPts
		//		nPtsLimiter,				//  ,      - nPtsLimiter
		//		amountOfPointsInBlock,		//      ( tMax / h / preScaller ) 
		//		i * originalNPtsLimiter,	//     
		//		amountOfPointsForSkip,		//     ( transientTime )
		//		1,							//  (   )
		//		d_ranges,					//   
		//		h,							//  
		//		d_indicesOfMutVars,			//   
		//		d_initialConditions,		//  
		//		amountOfInitialConditions,	//   
		//		d_values,					// 
		//		amountOfValues,				//  
		//		amountOfPointsInBlock,		//   (       )
		//		preScaller,					// ,      
		//		writableVar,				//  ,     
		//		maxValue,					//   ( ),     ""
		//		d_data,						// ,     
		//		d_sysCheker + (i * originalNPtsLimiter));			//  ,       '-1'   

		// --------------------------------------------------

		// ---   CUDA  ---
		gpuGlobalErrorCheck();

		// ---        ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// ---    CUDA,        ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA     ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						//    
				amountOfPointsInBlock,		//     
				nPtsLimiter,				//  ,    
				d_amountOfPeaks,			//  ,        
				d_outPeaks,					//  ,     
				d_intervals,				//     
				h*preScaller);							//    

		////		// ---   CUDA  ---
		////gpuGlobalErrorCheck();

		////// ---        ---
		////gpuErrorCheck(cudaDeviceSynchronize());

		////// ---    CUDA,        ---
		////cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_for2Dbif, 0, nPtsLimiter);
		//////blockSize = blockSize > 512 ? 512 : blockSize;			//     512   
		////gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		////avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
		////	(d_data,						//    
		////		amountOfPointsInBlock,		//     
		////		nPtsLimiter,				//  ,    
		////		d_avgPeaks + (i * originalNPtsLimiter),
		////		d_avgIntervals + (i * originalNPtsLimiter),
		////		d_data,						//  ,     
		////		d_intervals,				//  
		////		d_amountOfPeaks + (i * originalNPtsLimiter),
		////		d_sysCheker + (i * originalNPtsLimiter),
		////		h* preScaller);			//  

		// -----------------------------------------

		// ---   CUDA  ---
		gpuGlobalErrorCheck();

		// ---        ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// ---         GPU    ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_outIntervals, d_intervals, nPtsLimiter* amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// ---      ---
		outFileStream << std::setprecision(12);

		// ---     ---
		for (size_t k = 0; k < nPtsLimiter; ++k) {
			if (h_amountOfPeaks[k] == -1) {
				outFileStream << getValueByIdx(originalNPtsLimiter* i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock] << ", " << 0  << '\n';
			}
			else {
				for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
					if (outFileStream.is_open())
					{
						outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << ", " << h_outIntervals[k * amountOfPointsInBlock + j] << '\n';
					}
					else
					{
#ifdef DEBUG
						printf("\nOutput file open error\n");
#endif
						exit(1);
					}
			}
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}


	//double* h_avgPeaks = new double[nPts];
	//double* h_avgIntervals = new double[nPts];
	//int* h_sysCheker = new int[nPts];
	//int* h_amountOfPeaks = new int[nPts];

	//gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


//	std::ofstream outFileStream;
//	outFileStream.open(OUT_FILE_PATH);
//	outFileStream << std::setprecision(12);
//
//		// ---     ---
//			for (size_t j = 0; j < nPts; ++j)
//				if (outFileStream.is_open())
//				{
//					outFileStream  << h_avgPeaks[j] << ", " << h_avgIntervals[j] << ", " << h_amountOfPeaks[j] << '\n';
//				}
//				else
//				{
//#ifdef DEBUG
//					printf("\nOutput file open error\n");
//#endif
//					exit(1);
//				}

					   

	// ---------------------------
	// ---   ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));
	gpuErrorCheck(cudaFree(d_sysCheker));



	delete[] h_outPeaks;
	delete[] h_outIntervals;
	delete[] h_amountOfPeaks;
	

	// ---------------------------
}
