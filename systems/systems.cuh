#ifndef SYSTEMS_CUH
#define SYSTEMS_CUH


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#define USE_ROSSLER_MODEL  

#ifdef USE_CHAMELEON_MODEL
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
    	double h1 = a[0] * h;
	double h2 = (1 - a[0]) * h;
	X[0] = X[0] + h1 * (-a[6] * X[1]);
	X[1] = X[1] + h1 * (a[6] * X[0] + a[1] * X[2]);
	X[2] = X[2] + h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1]));

	X[2] = (X[2] + h2 * (a[2] + a[4] * cos(a[5] * X[1]))) / (1 + a[3] * h2);
	X[1] = X[1] + h2 * (a[6] * X[0] + a[1] * X[2]);
	X[0] = X[0] + h2 * (-a[6] * X[1]);
        // double h1 = a[0] * h;                                             
        // double h2 = (1 - a[0]) * h;                                       
                                                                          
        // /* Первый этап расчета */                                         
        // X[0] = __fma_rn(h1, -a[6] * X[1], X[0]);                          
        // X[1] = __fma_rn(h1, a[6] * X[0] + a[1] * X[2], X[1]);             
        // double cos_term = cos(a[5] * X[1]);                               
        // X[2] = __fma_rn(h1, a[2] - a[3] * X[2] + a[4] * cos_term, X[2]);  
                                                                          
        // /* Второй этап расчета */                                         
        // X[2] = __fma_rn(h2, (a[2] + a[4] * cos_term), X[2])               
        //        / (1 + a[3] * h2);                                         
        // X[1] = __fma_rn(h2, (a[6] * X[0] + a[1] * X[2]), X[1]);           
        // X[0] = __fma_rn(h2, -a[6] * X[1], X[0]);                          
    }
    #define SIZE_X 3
    #define SIZE_A 7
    #define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_ROSSLER_MODEL
__device__ inline void calcDiscreteModel(double* x, const double* a, double h) {
        double h1 = 0.5 * h + a[0];
        double h2 = 0.5 * h - a[0];

        // First stage calculations using __fma_rn
        x[0] = __fma_rn(h1, -x[1] - x[2], x[0]);
        x[1] = __fma_rn(h1, x[0] + a[1] * x[1], x[1]);
        x[2] = __fma_rn(h1, a[2] + x[2] * (x[0] - a[3]), x[2]);

        // Second stage calculations using __fma_rn where possible
        double temp = __fma_rn(-h2, x[0] - a[3], 1.0); // Calculate denominator 1 - h2 * (x[0] - a[3])
        x[2] = __fma_rn(h2, a[2], x[2]) / temp;
        
        temp = __fma_rn(-h2, a[1], 1.0); // Calculate denominator 1 - h2 * a[1]
        x[1] = __fma_rn(h2, x[0], x[1]) / temp;
        
        x[0] = __fma_rn(h2, -x[1] - x[2], x[0]);                        
    }
    #define SIZE_X 3
    #define SIZE_A 4
    #define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif


#endif // SYSTEMS_CUH