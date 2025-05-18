#ifndef SYSTEMS_CUH
#define SYSTEMS_CUH


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#define USE_DAMIR_SYSTEM

#ifdef USE_CHAMELEON_MODEL
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
        // double h1 = a[0] * h;
        // double h2 = (1 - a[0]) * h;
        // X[0] = X[0] + h1 * (-a[6] * X[1]);
        // X[1] = X[1] + h1 * (a[6] * X[0] + a[1] * X[2]);
        // X[2] = X[2] + h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1]));

        // X[2] = (X[2] + h2 * (a[2] + a[4] * cos(a[5] * X[1]))) / (1 + a[3] * h2);
        // X[1] = X[1] + h2 * (a[6] * X[0] + a[1] * X[2]);
        // X[0] = X[0] + h2 * (-a[6] * X[1]);

        double h1 = a[0] * h;                                             
        double h2 = (1 - a[0]) * h;                                                                                      
        /* Первый этап расчета */                                         
        X[0] = __fma_rn(h1, -a[6] * X[1], X[0]);                          
        X[1] = __fma_rn(h1, a[6] * X[0] + a[1] * X[2], X[1]);             
        double cos_term = cos(a[5] * X[1]);                               
        X[2] = __fma_rn(h1, a[2] - a[3] * X[2] + a[4] * cos_term, X[2]);  
                                                     
        /* Второй этап расчета */                                         
        X[2] = __fma_rn(h2, (a[2] + a[4] * cos_term), X[2]) / (1 + a[3] * h2);                                         
        X[1] = __fma_rn(h2, (a[6] * X[0] + a[1] * X[2]), X[1]);           
        X[0] = __fma_rn(h2, -a[6] * X[1], X[0]);                          
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



#ifdef USE_SYSTEM_FOR_BASINS
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
        float h1 = h * a[0];
        float h2 = h * (1 - a[0]);

        X[0] = X[0] + h * (sin(X[1]) - a[1] * X[0]);
        X[1] = X[1] + h * (sin(X[2]) - a[1] * X[1]);
        X[2] = X[2] + h * (sin(X[0]) - a[1] * X[2]);

        X[2] = (X[2] + h2 * sin(X[0])) / (1 + h2 * a[1]);
        X[1] = (X[1] + h2 * sin(X[2])) / (1 + h2 * a[1]);
        X[0] = (X[0] + h2 * sin(X[1])) / (1 + h2 * a[1]);        
    }
    #define SIZE_X 3
    #define SIZE_A 5
    #define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_SYSTEM_FOR_BASINS_2
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
    float h1 = h * a[0];
    float h2 = h * (1 - a[0]);
    
    X[0] = X[0] + h1 * (-X[1]);
    X[1] = X[1] + h1 * (a[1] * X[0] + sin(X[1]));
    
    float z = X[1];
    
    X[1] = z + h2 * (a[1] * X[0] + sin(X[1]));
    X[0] = X[0] + h2 * (-X[1]);       
    }
    #define SIZE_X 2
    #define SIZE_A 2
    #define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_DAMIR_SYSTEM
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
    // X[0] = Vc, X[1] = XSV, X[2] = sin(omega t), X[3] = cos(omega t)
    double Vc = X[0];
    double XSV = X[1];
    double S = X[2];
    double C = X[3];

    double U1 = 0.0;
    double U2 = -0.04;
    double C1 = 22e-9;

    // --- gi403 ---
    double Is = 1.15E-7;
    double Vt = 1.0/15.0;
    double Vp = 0.09;
    double Ip = 2.1e-5;
    double Iv = -3e-6;
    double D = 26.0;
    double E = 0.14;

    double Vd = Vc + U2;
    double Vm = Vc + U1;

    double idiode = Is * (exp(Vd/Vt) - exp(-Vd/Vt));
    double itunnel = Ip/Vp * Vd * exp(-(Vd - Vp)/Vp);
    double iex = Iv * (atan(D*(Vd - E)) + atan(D*(Vd + E)));
    double Id = idiode + itunnel + iex;

    // --- and_ts ---
    double Ron = 1434.0;
    double Roff = 1e6;
    double Von1 = 0.28;
    double Voff1 = 0.14;
    double Von2 = -0.12;
    double Voff2 = -0.006;
    double TAU = 1e-7;
    double T = 2.0;
    double boltz = 1.380649e-23;
    double echarge = 1.602176634e-19;

    double g = XSV/Ron + (1.0 - XSV)/Roff;

    double arg1 = -1.0/(T*boltz/echarge)*(Vm-Von1)*(Vm-Von2);
    double arg2 = -1.0/(T*boltz/echarge)*(Vm-Voff2)*(Vm-Voff1);

    double exp1 = exp(arg1);
    double exp2 = exp(arg2);

    double f1 = 1.0/(1.0 + exp1);
    double f2 = 1.0/(1.0 + exp2);

    double Ix = (1.0/TAU) * (f1 * (1.0 - XSV) - (1.0 - f2) * XSV);
    double Imem = Vm * g;

    // --- Iin: синусоида через прошлое состояние ---
    // a[0] = амплитуда, a[1] = частота (omega)
    double Iin = a[0] * S;

    // --- Производные ---
    double dVc = (Iin - Imem - Id) / C1;
    double dXSV = Ix;

    // --- Обновление синуса и косинуса ---
    double omega = a[1];
    double sh = sin(omega * h);
    double ch = cos(omega * h);
    double S_new = S * ch + C * sh;
    double C_new = C * ch - S * sh;

    // --- Эйлеровский шаг ---
    X[0] = Vc + h * dVc;
    X[1] = XSV + h * dXSV;
    X[2] = S_new;
    X[3] = C_new;
}
#define SIZE_X 4
#define SIZE_A 2
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif
#endif // SYSTEMS_CUH