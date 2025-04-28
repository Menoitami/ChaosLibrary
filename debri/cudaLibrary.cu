#include "cudaLibrary.cuh"
#include <curand_kernel.h>
// ---------------------------------------------------------------------------------
// --- ��������� ��������� �������� ���������� ������ � ���������� ��������� � x ---
// ---------------------------------------------------------------------------------
namespace old_library{
__device__ __host__ void calculateDiscreteModelforFastSynchro(double* X, double* S1, double* K, const double* a, const double h)
{

	double N[3], h1, h2;
	h1 = h * a[0];
	h2 = h * (1 - a[0]);

	N[0] = K[0] * (S1[0] - X[0]);
	N[1] = K[1] * (S1[1] - X[1]);
	N[2] = K[2] * (S1[2] - X[2]);


	//// --- DissipCons CD ---
	//X[0] = X[0] + h1 * (N[0] + (X[2] * X[2] - a[3]) * X[0] - a[5] * X[1]);
	//X[1] = X[1] + h1 * (N[1] + a[5] * X[0] + (X[2] * X[2] - a[3]) * X[1]);
	//X[2] = X[2] + h1 * (N[2] + (a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]) * X[2] + a[4] * X[0]);

	//X[2] = (X[2] + h2 * (N[2] +a[4] * X[0])) / (1 - h2 * (a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]));
	//X[1] = (X[1] + h2 * (N[1] +a[5] * X[0])) / (1 - h2 * (X[2] * X[2] - a[3]));
	//X[0] = (X[0] + h2 * (N[0] -a[5] * X[1])) / (1 - h2 * (X[2] * X[2] - a[3]));

	//// --- Lorenz CD ---
	// X[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]) + N[0]);
	// X[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1] + N[1]);
	// X[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2] + N[2]);
	// X[2] = (X[2] + h2 * (X[0] * X[1] + N[2])) / (1 + h2 * a[3] * a[4]);
	// X[1] = (X[1] + h2 * (X[0] * (a[2] - X[2]) + N[1])) / (1 + h2 * a[4]);
	// X[0] = (X[0] + h2 * (a[1] * (X[1]) + N[0])) / (1 + a[1] * h2 * a[4]);

	// X[3] = -a[2] * X[0] * (a[1] * (X[1] - a[4] * X[0])) + X[1] * (a[2] * X[0] - a[4] * X[1] - X[0] * X[2]) + X[2] * (X[0] * X[1] - a[3] * a[4] * X[2]); //dH/dt
	// X[3] = 0.5 * (-a[2] * X[0] * X[0] / a[1] + X[1] * X[1] + X[2] * X[2]); //Pseudo-Hamiltonian 
	// X[3] = a[1] * ( - a[2]*a[3] + a[3]*X[2] + a[3] - X[0]*X[0] - X[0]*X[1]); //Jacobian = -a b c + a c z + a c - a x ^ 2 - a x y
	// X[3] = (-a[2] * a[3] + a[3] * X[2] + a[3] - X[0] * X[0] - X[0] * X[1]);
	// X[3] = (-a[2] * a[3] + a[3] * X[2] + a[3] - X[0] * X[0] - X[0] * X[1]) * 0.5 * (-a[2] * X[0] * X[0] / a[1] + X[1] * X[1] + X[2] * X[2]);
	// X[3] = -0.5 * (a[1] * X[0] * X[0] + X[1]*X[1] + a[3]*X[2]*X[2] ); //dissipativity 
	// X[3] = 0.5 * (X[1] * X[1] + (X[2] - a[2]) * (X[2] - a[2]));
	// X[3] = a[1]*X[2] - X[0]*X[0]/2;
	// X[3] = 0.5 * (a[2] * X[0] * X[0] + a[1] * X[1] * X[1] + a[1] * (X[2] - 2 * a[2]) * (X[2] - 2 * a[2]));

	// --- Lorenz RK2
	// double X1[3];
	// X1[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]) + N[0]);
	// X1[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1] + N[1]);
	// X1[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2] + N[2]);
	// h2 = 1 * h;
	// X[0] = X[0] + h2 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]);
	// X[1] = X[1] + h2 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]);
	// X[2] = X[2] + h2 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]);
	
	// --- Lorenz MEMP
	//double X1[3], X2[3];
	//h2 = 1 * h;
	//X1[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]) + N[0]);
	//X1[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1] + N[1]);
	//X1[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2] + N[2]);

	//X2[0] = X[0] + h2 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]);
	//X2[1] = X[1] + h2 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]);
	//X2[2] = X[2] + h2 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]);

	//X[0] = X1[0];	X[1] = X1[1];	X[2] = X1[2];
	//X1[0] = X2[0]; X1[1] = X2[1]; X1[2] = X2[2];

	//X2[0] = 0.5 * (X[0] + X1[0] + h1 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]));
	//X2[1] = 0.5 * (X[1] + X1[1] + h1 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]));
	//X2[2] = 0.5 * (X[2] + X1[2] + h1 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]));
	//X[0] = X2[0];	X[1] = X2[1];	X[2] = X2[2];

	//// --- Nose-Hoover CD ---
	//X[0] = X[0] + h1 * (a[1] * X[1] + N[0]);
	//X[1] = (X[1] - h1 * (X[0] - N[1])) / (1 - h1 * a[2] * X[2]);
	//X[2] = X[2] + h1 * (1 - a[3] * X[1] * X[1] + N[2]);
	//X[2] = X[2] + h2 * (1 - a[3] * X[1] * X[1] + N[2]);
	//X[1] = X[1] + h2 * (a[2] * X[1] * X[2] - X[0] + N[1]);
	//X[0] = X[0] + h2 * (a[1] * X[1] + N[0]);

	// --- Rossler CD ---
	//X[0] = X[0] + h1 * (-X[1] - X[2] + N[0]);
	//X[1] = (X[1] + h1 * (X[0] + N[1])) / (1 - a[1] * h1);
	//X[2] = (X[2] + h1 * (a[2] + N[2])) / (1 - h1 * (X[0] - a[3]));
	//X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]) + N[2]);
	//X[1] = X[1] + h2 * (X[0] + a[1] * X[1] + N[1]);
	//X[0] = X[0] + h2 * (-X[1] - X[2] + N[0]);

	// --- Case B CD ---
	//X[0] = X[0] + h1 * (X[1] * X[2] + N[0]);
	//X[1] = (X[1] + h1 * (X[0] + N[1])) / (1 + h2);

	////X[2] = X[2] + h1 * (1 - X[0] * X[1] + N[2]);
	////X[2] = X[2] + h2 * (1 - X[0] * X[1] + N[2]);
	//X[2] = X[2] + h * (1 - X[0] * X[1] + N[2]);

	//X[1] = X[1] + h2 * (X[0] - X[1] + N[1]);
	//X[0] = X[0] + h2 * (X[1] * X[2] + N[0]);

	// --- Dadras Mimeni CD ---
	//X[0] = (X[0] + h1 * (N[0] + X[1] + a[2] * X[1] * X[2])) / (1 + a[1] * h1);
	//X[2] = (X[2] + h1 * (N[2] + a[4] * X[0] * X[1])) / (1 + h1 * a[5]);
	//X[1] = (X[1] + h1 * (N[1] - X[0] * X[2] + X[2])) / (1 - h1 * a[3]);
	//X[1] = X[1] + h2 * (N[1] + a[3] * X[1] - X[0] * X[2] + X[2]);
	//X[2] = X[2] + h2 * (N[2] + a[4] * X[0] * X[1] - a[5] * X[2]);
	//X[0] = X[0] + h2 * (N[0] + X[1] - a[1] * X[0] + a[2] * X[1] * X[2]);

	// --- analog Chua model CD ---

	//double ax_1[3], ax_2[2], ax_3[3], ay[3], az[3];
	//double Ep = 1.0;
	//double En = -0.9;

	//ax_1[0] = -2.953439169151177e+03;
	//ax_1[1] = -1.135841174789108e+03;
	//ax_1[2] = 4.893809454164432e+03;
	//ax_2[0] = 2.069181630271321e+03;
	//ax_2[1] = 4.925167383017512e+03;
	//ax_3[0] = 3.137530613834112e+03;
	//ax_3[1] = -1.152680011500723e+03;
	//ax_3[2] = 4.896700576792318e+03;
	//ay[0] = 4.530191157690266e+02;
	//ay[1] = -8.916884945316002e+03;
	//ay[2] = 8.481167795704148e+03;
	//az[0] = 4.521739769426720e+02;
	//az[1] = -9.447899219498395e+03;
	//az[2] = 8.464586388517310e+03;
	//   
	//if (X[0] < En)
	//	X[0] = X[0] + h1 * (N[0] + ax_1[0] + ax_1[1] * X[0] + ax_1[2] * X[1]);
	//else if (X[0] >= En && X[0] <= Ep)
	//	X[0] = X[0] + h1 * (N[0] + ax_2[0] * X[0] + ax_2[1] * X[1]);
	//else
	//	X[0] = X[0] + h1 * (N[0] + ax_3[0] + ax_3[1] * X[0] + ax_3[2] * X[1]);


	//X[1] = X[1] + h1 * (N[1] + ay[0] * X[0] + ay[1] * X[1] + ay[2] * X[2]);
	//X[2] = X[2] + h1 * (N[2] + az[0] * X[0] + az[1] * X[1] + az[2] * X[2]);

	//X[2] = (X[2] + h2 * (N[2] + az[0] * X[0] + az[1] * X[1])) / (1 - h2 * az[2]);
	//X[1] = (X[1] + h2 * (N[1] + ay[0] * X[0] + ay[2] * X[2])) / (1 - h2 * ay[1]);

	//if (X[0] < En)
	//	X[0] = (X[0] + h2 * (N[0] + ax_1[0] + ax_1[2] * X[1])) / (1 - h2 * ax_1[1]);
	//else if (X[0] >= En && X[0] <= Ep)
	//	X[0] = (X[0] + h2 * (N[0] + ax_2[1] * X[1])) / (1 - h2 * ax_2[0]);
	//else
	//	X[0] = (X[0] + h2 * (N[0] + ax_3[0] + ax_3[2] * X[1])) / (1 - h2 * ax_3[1]);

}

__device__ __host__ void calculateDiscreteModelforFastSynchroBackward(double* X, double* S1, double* K, const double* a, const double h)
{

	double N[3], h1, h2;
	h1 = -1 * h * a[0];
	h2 = -1 * h * (1 - a[0]);

	N[0] = -K[0] * (S1[0] - X[0]);
	N[1] = -K[1] * (S1[1] - X[1]);
	N[2] = -K[2] * (S1[2] - X[2]);

	//// --- DissipCons CD ---
	//X[0] = X[0] + h1 * (N[0] + (X[2] * X[2] - a[3]) * X[0] - a[5] * X[1]);
	//X[1] = X[1] + h1 * (N[1] + a[5] * X[0] + (X[2] * X[2] - a[3]) * X[1]);
	//X[2] = X[2] + h1 * (N[2] + (a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]) * X[2] + a[4] * X[0]);

	//X[2] = (X[2] + h2 * (N[2] + a[4] * X[0])) / (1 - h2 * (a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]));
	//X[1] = (X[1] + h2 * (N[1] + a[5] * X[0])) / (1 - h2 * (X[2] * X[2] - a[3]));
	//X[0] = (X[0] + h2 * (N[0] - a[5] * X[1])) / (1 - h2 * (X[2] * X[2] - a[3]));

	//// --- Lorenz CD ---
	// X[0] = X[0] + h1 * (a[1] * (X[1] - a[4]*X[0]) + N[0]);
	// X[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4]*X[1] + N[1]);
	// X[2] = X[2] + h1 * (X[0] * X[1] - a[4]* a[3] * X[2] + N[2]);
	// X[2] = (X[2] + h2 * (X[0] * X[1] + N[2])) / (1 + h2 * a[3]*a[4]);
	// X[1] = (X[1] + h2 * (X[0] * (a[2] - X[2]) + N[1])) / (1 + h2*a[4]);
	// X[0] = (X[0] + h2 * (a[1] * (X[1]) + N[0])) / (1 + a[1] * h2 * a[4]);

	// X[3] = -a[2] * X[0] * (a[1] * (X[1] - a[4] * X[0])) + X[1] * (a[2] * X[0] - a[4] * X[1] - X[0] * X[2]) + X[2] * (X[0] * X[1] - a[3] * a[4] * X[2]); //dH/dt
	// X[3] = 0.5 * (-a[2] * X[0] * X[0] / a[1] + X[1] * X[1] + X[2] * X[2]); //Pseudo-Hamiltonian 
	// X[3] = a[1] * (-a[2] * a[3] + a[3] * X[2] + a[3] - X[0] * X[0] - X[0] * X[1]); //Jacobian = -a b c + a c z + a c - a x ^ 2 - a x y
	// X[3] = (-a[2] * a[3] + a[3] * X[2] + a[3] - X[0] * X[0] - X[0] * X[1]);
	// X[3] = (-a[2] * a[3] + a[3] * X[2] + a[3] - X[0] * X[0] - X[0] * X[1]) * 0.5 * (-a[2] * X[0] * X[0] / a[1] + X[1] * X[1] + X[2] * X[2]);
	// X[3] = -0.5 * (a[1] * X[0] * X[0] + X[1] * X[1] + a[3] * X[2] * X[2]);
	// X[3] = 0.5 * (X[1] * X[1] + (X[2] - a[2]) * (X[2] - a[2]));
	// X[3] = a[1] * X[2] - X[0] * X[0] / 2;
	// X[3] = 0.5 * (a[2] * X[0] * X[0] + a[1] * X[1] * X[1] + a[1] * (X[2] - 2 * a[2]) * (X[2] - 2 * a[2]));

	// --- Lorenz RK2
	//double X1[3];
	//X1[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]) + N[0]);
	//X1[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1] + N[1]);
	//X1[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2] + N[2]);
	//h2 = -1*h;
	//X[0] = X[0] + h2 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]);
	//X[1] = X[1] + h2 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]);
	//X[2] = X[2] + h2 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]);

	// --- Lorenz MEMP
	//double X1[3], X2[3];
	//h2 = -1 * h;
	//X1[0] = X[0] + h1 * (a[1] * (X[1] - a[4] * X[0]) + N[0]);
	//X1[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - a[4] * X[1] + N[1]);
	//X1[2] = X[2] + h1 * (X[0] * X[1] - a[4] * a[3] * X[2] + N[2]);

	//X2[0] = X[0] + h2 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]);
	//X2[1] = X[1] + h2 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]);
	//X2[2] = X[2] + h2 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]);

	//X[0] = X1[0];	X[1] = X1[1];	X[2] = X1[2];
	//X1[0] = X2[0]; X1[1] = X2[1]; X1[2] = X2[2];

	//X2[0] = 0.5 * (X[0] + X1[0] + h1 * (a[1] * (X1[1] - a[4] * X1[0]) + N[0]));
	//X2[1] = 0.5 * (X[1] + X1[1] + h1 * (X1[0] * (a[2] - X1[2]) - a[4] * X1[1] + N[1]));
	//X2[2] = 0.5 * (X[2] + X1[2] + h1 * (X1[0] * X1[1] - a[4] * a[3] * X1[2] + N[2]));
	//X[0] = X2[0];	X[1] = X2[1];	X[2] = X2[2];

	//// --- Nose-Hoover CD ---
	//X[0] = X[0] + h1 * (a[1] * X[1] + N[0]);
	//X[1] = (X[1] - h1 * (X[0] - N[1])) / (1 - h1 * a[2] * X[2]);
	//X[2] = X[2] + h1 * (1 - a[3] * X[1] * X[1] + N[2]);
	//X[2] = X[2] + h2 * (1 - a[3] * X[1] * X[1] + N[2]);
	//X[1] = X[1] + h2 * (a[2] * X[1] * X[2] - X[0] + N[1]);
	//X[0] = X[0] + h2 * (a[1] * X[1] + N[0]);

	// --- Rossler CD ---
	//X[0] = X[0] + h1 * (-X[1] - X[2] + N[0]);
	//X[1] = (X[1] + h1 * (X[0] + N[1])) / (1 - a[1] * h1);
	//X[2] = (X[2] + h1 * (a[2] + N[2])) / (1 - h1 * (X[0] - a[3]));
	//X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]) + N[2]);
	//X[1] = X[1] + h2 * (X[0] + a[1] * X[1] + N[1]);
	//X[0] = X[0] + h2 * (-X[1] - X[2] + N[0]);

	// --- Case B CD ---
	//X[0] = X[0] + h1 * (X[1] * X[2] + N[0]);
	//X[1] = (X[1] + h1 * (X[0] + N[1])) / (1 + h2);

	////X[2] = X[2] + h1 * (1 - X[0] * X[1] + N[2]);
	////X[2] = X[2] + h2 * (1 - X[0] * X[1] + N[2]);
	//X[2] = X[2] - h * (1 - X[0] * X[1] + N[2]);

	//X[1] = X[1] + h2 * (X[0] - X[1] + N[1]);
	//X[0] = X[0] + h2 * (X[1] * X[2] + N[0]);

	// --- Dadras Mimeni CD ---
	//X[0] = (X[0] + h1 * (N[0] + X[1] + a[2] * X[1] * X[2])) / (1 + a[1] * h1);
	//X[2] = (X[2] + h1 * (N[2] + a[4] * X[0] * X[1])) / (1 + h1 * a[5]);
	//X[1] = (X[1] + h1 * (N[1] - X[0] * X[2] + X[2])) / (1 - h1 * a[3]);
	//X[1] = X[1] + h2 * (N[1] + a[3] * X[1] - X[0] * X[2] + X[2]);
	//X[2] = X[2] + h2 * (N[2] + a[4] * X[0] * X[1] - a[5] * X[2]);
	//X[0] = X[0] + h2 * (N[0] + X[1] - a[1] * X[0] + a[2] * X[1] * X[2]);

	// --- analog Chua model CD ---

	//double ax_1[3], ax_2[2], ax_3[3], ay[3], az[3];
	//double Ep = 1.0;
	//double En = -0.9;

	//ax_1[0] = -2.953439169151177e+03;
	//ax_1[1] = -1.135841174789108e+03;
	//ax_1[2] = 4.893809454164432e+03;
	//ax_2[0] = 2.069181630271321e+03;
	//ax_2[1] = 4.925167383017512e+03;
	//ax_3[0] = 3.137530613834112e+03;
	//ax_3[1] = -1.152680011500723e+03;
	//ax_3[2] = 4.896700576792318e+03;
	//ay[0] = 4.530191157690266e+02;
	//ay[1] = -8.916884945316002e+03;
	//ay[2] = 8.481167795704148e+03;
	//az[0] = 4.521739769426720e+02;
	//az[1] = -9.447899219498395e+03;
	//az[2] = 8.464586388517310e+03;

	//if (X[0] < En)
	//	X[0] = X[0] + h1 * (N[0] + ax_1[0] + ax_1[1] * X[0] + ax_1[2] * X[1]);
	//else if (X[0] >= En && X[0] <= Ep)
	//	X[0] = X[0] + h1 * (N[0] + ax_2[0] * X[0] + ax_2[1] * X[1]);
	//else
	//	X[0] = X[0] + h1 * (N[0] + ax_3[0] + ax_3[1] * X[0] + ax_3[2] * X[1]);


	//X[1] = X[1] + h1 * (N[1] + ay[0] * X[0] + ay[1] * X[1] + ay[2] * X[2]);
	//X[2] = X[2] + h1 * (N[2] + az[0] * X[0] + az[1] * X[1] + az[2] * X[2]);

	//X[2] = (X[2] + h2 * (N[2] + az[0] * X[0] + az[1] * X[1])) / (1 - h2 * az[2]);
	//X[1] = (X[1] + h2 * (N[1] + ay[0] * X[0] + ay[2] * X[2])) / (1 - h2 * ay[1]);

	//if (X[0] < En)
	//	X[0] = (X[0] + h2 * (N[0] + ax_1[0] + ax_1[2] * X[1])) / (1 - h2 * ax_1[1]);
	//else if (X[0] >= En && X[0] <= Ep)
	//	X[0] = (X[0] + h2 * (N[0] + ax_2[1] * X[1])) / (1 - h2 * ax_2[0]);
	//else
	//	X[0] = (X[0] + h2 * (N[0] + ax_3[0] + ax_3[2] * X[1])) / (1 - h2 * ax_3[1]);
}

__device__ void calculateDiscreteModel_rand(size_t seed, double* X, const double* a, const double h)
{
	curandState_t state;


	//curand_init(seed+1, 0, 0, &state);
	//X[0] = curand_uniform(&state);

	//double h1 = h / 2;

	//X[0] = X[0] - h1 * (X[1] + X[2]);
	//X[1] = X[1] + h1 * (X[0] + X[1] * a[0]);
	//X[2] = X[2] + h1 * (a[1] + X[2] * (X[0] - a[2]));

	//X[2] = (X[2] + h1 * a[1]) / (1 - h1 * (X[0] - a[2]));
	//X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[0]);
	//X[0] = X[0] - h1 * (X[1] + X[2]);

	//X[0] = X[0] + 0.5*(curand_uniform(&state) - 0.5);

	//double A[2], B[2], buff, X_1, Y_1;
	//A[0] = 1;
	//A[1] = -0.996863331833438;
	//B[0] = 0.001568334083281;
	//B[1] = 0.001568334083281;

	//X_1 = X[0];
	//Y_1 = X[1];

	//curand_init(seed+1, 0, 0, &state);
	//X[0] = curand_uniform(&state);
	//curand_init(seed * seed, 0, 0, &state);

	//X[0] = sinf(2 * 3.14159265359 * X[0]) * sqrtf(-2 * log(curand_uniform(&state)));
	//X[1] =   X[0] * B[0] +  X_1 * B[1] - Y_1 * A[1];


	
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//double X1[3], k[4][4], Im, Id, Iin;
	//double pi = 3.14159265359;
	//double u1, u2;
	//int N = 3;
	//int i, j;


	//for (i = 0; i < N; i++) {
	//	X1[i] = X[i];
	//}

	//for (j = 0; j < 4; j++) {


	//
	//	////////////////////////////////////////////////////////////////////////////////////
	//	
	//	//Id = a[4] * (exp( (X[0] + a[6]) / a[9]) - exp( -(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * exp( -(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atan( a[17] * (X[0] + a[6] - a[18])) + atan( a[17] * (X[0] + a[6] + a[18])));

	//	Id = a[4] * (expf((X[0] + a[6]) / a[9]) - expf(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * expf(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atanf(a[17] * (X[0] + a[6] - a[18])) + atanf(a[17] * (X[0] + a[6] + a[18])));

	//	if ((-X[0] + a[7]) > 0)
	//		Im = (-X[0] + a[7]) * X[1] / a[19] + a[5];
	//	else
	//		Im = (X[0] + a[7]) * X[1] / a[20] - a[5];

	//	Iin   = a[25];
	//	//Iin = a[25] * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
	//	//Iin = a[25] *  (fmod(X[2] + a[28], a[26])  )/a[26];		

	//	k[0][j] = (Iin - Im - Id) / a[8];
	//	k[1][j] = (1 / a[21]) * (1 / (1 + expf(-1 / (a[15] * a[15]) * ((-X[0] + a[7]) - X[3]) * ((-X[0] + a[7]) - a[13])))) * ((1 - 1 / (expf((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + expf(-1 / (a[16] * a[16]) * ((-X[0] + a[7]) - a[14]) * ((-X[0] + a[7]) - X[4])))) * ((1 - 1 / (expf((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]) + a[24]))));
	//	//k[1][j] = (1 / a[21]) * (1 / (1 + exp( -1 / (a[15] * a[15]) * ((-X[0] + a[7]) - z[0]) * ((-X[0] + a[7]) - a[13])))) * ((1 - 1 / (exp( (a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp( a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + exp( -1 / (a[16] * a[16]) * ((-X[0] + a[7]) - a[14]) * ((-X[0] + a[7]) - z[1])))) * ((1 - 1 / (exp( (a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp( a[1] * (1 - X[1]) + a[24]))));


	//	if ((k[1][j] < 0) && (X[5] == 0)) {
	//		//curand_init(seed, 0, 0, &state);
	//		//u1 = curand_uniform(&state);
	//		//curand_init(seed*seed, 0, 0, &state);
	//		//u2 = curand_uniform(&state);
	//		//X[3] = sqrt(-2 * log(u1)) * cos(2 * pi * u2) * 0.02236 + a[11];
	//		X[3] = a[11];
	//		X[5] = 1;
	//		X[6] = 0;
	//	}
	//	else if ((k[1][j] > 0) && (X[6] == 0)) {
	//		//curand_init(seed+50, 0, 0, &state);
	//		//u1 = curand_uniform(&state);
	//		//curand_init(seed*seed+250, 0, 0, &state);
	//		//u2 = curand_uniform(&state);
	//		//X[4] = sqrt(-2 * log(u1)) * sin(2 * pi * u2) * 0.02 + a[12];
	//		X[4] = a[12];
	//		X[5] = 0;
	//		X[6] = 1;
	//	}
	//	k[2][j] = 1;


	//	if (j == 3) {
	//		for (i = 0; i < N; i++) {
	//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
	//		}
	//	}
	//	else if (j == 2) {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + h * k[i][j];
	//		}
	//	}
	//	else {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + 0.5 * h * k[i][j];
	//		}
	//	}
	//}
	//X[7] = Iin;
}

__device__ __host__ void calculateDiscreteModel(double* X, const double* a, const double h)
{

	// float h1 = h * a[0];
	// float h2 = h * (1 - a[0]);

	// X[0] = X[0] + h * (sin(X[1]) - a[1] * X[0]);
	// X[1] = X[1] + h * (sin(X[2]) - a[1] * X[1]);
	// X[2] = X[2] + h * (sin(X[0]) - a[1] * X[2]);

	// X[2] = (X[2] + h2 * sin(X[0])) / (1 + h2 * a[1]);
	// X[1] = (X[1] + h2 * sin(X[2])) / (1 + h2 * a[1]);
	// X[0] = (X[0] + h2 * sin(X[1])) / (1 + h2 * a[1]);


	    float h1 = h * a[0];
        float h2 = h * (1 - a[0]);
        
        X[0] = X[0] + h1 * (-X[1]);
        X[1] = X[1] + h1 * (a[1] * X[0] + sin(X[1]));
        
        float z = X[1];
        
        X[1] = z + h2 * (a[1] * X[0] + sin(X[1]));
        //X[1] = z + h2 * (a[1] * X[0] + sin(X[1]));
        X[0] = X[0] + h2 * (-X[1]);  


	// double h1 = a[0] * h;
	// double h2 = (1 - a[0]) * h;
	// X[0] = X[0] + h1 * (-a[6] * X[1]);
	// X[1] = X[1] + h1 * (a[6] * X[0] + a[1] * X[2]);
	// X[2] = X[2] + h1 * (a[2] - a[3] * X[2] + a[4] * cos(a[5] * X[1]));

	// X[2] = (X[2] + h2 * (a[2] + a[4] * cos(a[5] * X[1]))) / (1 + a[3] * h2);
	// X[1] = X[1] + h2 * (a[6] * X[0] + a[1] * X[2]);
	// X[0] = X[0] + h2 * (-a[6] * X[1]);
	    // double h1 = 0.5 * h + a[0];
        // double h2 = 0.5 * h - a[0];

        
        // X[0] += h1 * (-X[1] - X[2]);
        // X[1] += h1 * (X[0] + a[1] * X[1]);
        // X[2] += h1 * (a[2] + X[2] * (X[0] - a[3]));

        // X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));
        // X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);
        // X[0] += h2 * (-X[1] - X[2]); 

}



// -----------------------------------------------------------------------------------------------------
// --- ��������� ���������� ��� ����� ������� � ���������� ��������� � "data" (���� data != nullptr) ---
// -----------------------------------------------------------------------------------------------------

__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, 
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const double maxValue, double* data, 
	const int startDataIndex, const int writeStep)
{
	double* xPrev = new double[amountOfX];
	for ( int i = 0; i < amountOfIterations; ++i )
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		if ( data != nullptr )
			data[startDataIndex + i * writeStep] = x[writableVar];

		for ( int j = 0; j < preScaller; ++j )
			calculateDiscreteModel(x, values, h);

		if ( isnan( x[writableVar] ) || isinf( x[writableVar] ) )
		{
			delete[] xPrev;
			return false;
		}

		if ( maxValue != 0 )
			if ( fabsf( x[writableVar] ) > maxValue )
			{
				delete[] xPrev;
				return false;
			}
	}

	double tempResult = 0;
	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	}

	if (tempResult == 0)
	{
		delete[] xPrev;
		return false;
	}

	if (sqrt(tempResult) < 1e-12)
	{
		delete[] xPrev;
		return false;
	}

	delete[] xPrev;
	return true;
}

__device__ double loopCalculateDiscreteModelForFastSynchro(
	double* Xs, 
	const double* values,
	const double h,
	const double* K_Forward,
	const double* K_Backward,
	const double iterOfSynchr, 
	const int amountOfIterations, 
	const int amountOfX,
	const double maxValue, 
	double* timedomain, 
	const int startDataIndex)
{
	//double* norm_error = new double[amountOfIterations - 1];
	double* Xm = new double[amountOfX];
	double* K_local = new double[amountOfX];
	double rms_error = 0;
	//double err0 = 0;
	//double err1 = 0;

	//for (int j = 0; j < amountOfIterations - 1; ++j)
	//	norm_error[j] = 0;

	for (int j = 0; j < amountOfX; ++j) {
		Xs[j] = timedomain[startDataIndex + j] + 0.05;
	}


	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = K_Forward[j];

		// --- ���������� ����, ������� ���������� ���������� �������� amountOfIterations ��� ---
		for (int i = 0; i < amountOfIterations - 1; ++i)
		{
			for (int j = 0; j < amountOfX; ++j) {
				Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
			}	

			if (m == iterOfSynchr - 1) {
				for (int j = 0; j < amountOfX; ++j)
					rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);		
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h);
		}



		for (int j = 0; j < amountOfX; ++j)
			K_local[j] = K_Backward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			for (int j = 0; j < amountOfX; ++j)
				Xm[j] = timedomain[startDataIndex + i * amountOfX + j];

			calculateDiscreteModelforFastSynchroBackward(Xs, Xm, K_local, values, h);

		}
	}
	
	rms_error = log10(sqrt(rms_error / (double)(amountOfIterations - 1)));



	delete[] Xm;
	delete[] K_local;

	if (isinf(rms_error) || isnan(rms_error))
		rms_error = 15;

	return rms_error;
}


//__device__ __host__ int loopCalculateDiscreteModel_int(
__device__ int loopCalculateDiscreteModel_int(
	double* x, 
	const double* values,
	const double h, 
	const int amountOfIterations, 
	const int amountOfX, 
	const int preScaller,
	int writableVar, 
	const double maxValue, 
	double* data,
	const int startDataIndex, 
	const int writeStep)
{
	double* xPrev = new double[amountOfX];

	// --- ���������� ����, ������� ���������� ���������� �������� amountOfIterations ��� ---
	for (int i = 0; i < amountOfIterations; ++i)
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// --- ���� ���-���� �������� ������ ��� ������ - ���������� �������� ���������� ---


		if (data != nullptr) 
			data[startDataIndex + i * writeStep] = (x[writableVar]);
		
		// --- ���������� ������� preScaller ��� ( �� ���� ���� preScaller > 1, �� �� ��������� ( preScaller - 1 ) � ��������������� ���������� ) ---
		//for (int j = 0; j < preScaller; ++j) {
		//	//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
		//	calculateDiscreteModel(x, values, h);
		//}

		for (int j = 0; j < preScaller - 1; ++j)
			calculateDiscreteModel(x, values, h);

		calculateDiscreteModel(x, values, h);

		// 1 - stability, -1 - fixed point, 0 - unbound solution

		// --- ���� isnan ��� isinf - ���������� false, ��� ��� ������������� ��������� ������� ---
		if (isnan(x[writableVar]) || isinf(x[writableVar]))
		{
			delete[] xPrev;
			return 0;
		}

		// --- ���� maxValue == 0, ��� ������ ������������ �� �������� �����������, ����� ��������� ��� ��������� ---
		if (maxValue != 0)
			if (fabsf(x[writableVar]) > maxValue)
			{
				delete[] xPrev;
				return 0;
			}
	}

	// --- �������� �� ���������� � ����� ---
	double tempResult = 0;

	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
		//tempResult += abs(x[j] - xPrev[j]);
	}


	//if (abs(tempResult) < 1e-8)
	if (sqrt(abs(tempResult)) < 1e-9)
	{
		delete[] xPrev;
		return -1;
	}

	delete[] xPrev;
	return 1;
}

__device__ double loopCalculateDiscreteModelForFastSynchro_int(
	double* x,
	const double* values,
	const double h,
	const int amountOfIterations,
	const int amountOfX,
	const int preScaller,
	const double maxValue,
	const int	iterOfSynchr,
	const double* kForward,
	const double* kBackward,
	double* data,
	const int startDataIndex,
	const int writeStep)
{
	double* Xm = new double[amountOfX];
	double* Xs = new double[amountOfX];
	double* arrayZeros = new double[amountOfX];
	double* K_local = new double[amountOfX];
	double rms_error = 0;

	if (data != nullptr) {
		for (int w = 0; w < amountOfX; w++) {
			data[startDataIndex + w] = x[w];
			//Xs[w] = x[w] - 0.005;
			Xs[w] = x[w];
			Xm[w] = x[w];
			arrayZeros[w] = 0;
		}
	}

	for (int i = 1; i < amountOfIterations; i++) {
		calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h);

		for (int w = 0; w < amountOfX; w++)
			data[startDataIndex + w + i*amountOfX] = Xm[w];
	}

	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = kForward[j];

		// --- ���������� ����, ������� ���������� ���������� �������� amountOfIterations ��� ---
		for (int i = 0; i < amountOfIterations - 1; ++i) {

			for (int j = 0; j < amountOfX; ++j) {
				Xm[j] = data[startDataIndex + i * amountOfX + j];
				//Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
			}

			if (m == iterOfSynchr - 1) {
				for (int j = 0; j < amountOfX; ++j)
					rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				//norm_error[i] = norm_error[i] + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h);
		}



		for (int j = 0; j < amountOfX; ++j)
			K_local[j] = kBackward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			for (int j = 0; j < amountOfX; ++j)
				Xm[j] = data[startDataIndex + i * amountOfX + j];

			calculateDiscreteModelforFastSynchroBackward(Xs, Xm, K_local, values, h);

		}
	}

	//for (int j = 0; j < amountOfIterations - 1; ++j) {
	//	rms_error = rms_error + norm_error[j];
	//}

	rms_error = log10(sqrt(rms_error / (double)(amountOfIterations - 1)));

	//log(norm(err0 - err_fin)) / (n_iter * t_sync)
	//rms_error = (log10(err1) - log10(err0)) / (double)(amountOfIterations) / h / iterOfSynchr;

	//rms_error = log10(err1) - log10(err0);

	delete[] Xm;
	delete[] Xs;
	delete[] K_local;
	delete[] arrayZeros;
	//delete[] norm_error;

	if (isinf(rms_error) || isnan(rms_error))
		rms_error = 15;

	return rms_error;
}


__global__ void distributedCalculateDiscreteModelCUDA(
	const int		amountOfPointsForSkip,
	const int		amountOfThreads,
	const double	h,
	const double	hSpecial,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar,
	double*			data)
{
	extern __shared__ double s[];
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfThreads)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0);

	loopCalculateDiscreteModel(localX, localValues, h, idx,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0, 0);

	loopCalculateDiscreteModel(localX, localValues, hSpecial, amountOfIterations,
		amountOfInitialConditions, 1, writableVar, 0, data, idx, amountOfThreads);

	return;
}



// --------------------------------------------------------------------------
// --- ���������� �������, ������� ��������� ���������� ���������� ������ ---
// --------------------------------------------------------------------------

__global__ void calculateDiscreteModelforFastSynchroCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const double	h,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const double*	k_forward,
	const double*	k_backward,
	const int		iterOfSynchr,
	const int		amountOfValues,
	const int		amountOfNTPoints,
	const double	maxValue,
	double*			timedomain,
	double*			output,
	const int		preScaller)
{

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x +blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	//double Xs[3] = { initialConditions[0] , initialConditions[1], initialConditions[2] };
	double* Xs = new double[sizeof(double) * amountOfInitialConditions];

	for (int i = 0; i < amountOfInitialConditions; i++)
		Xs[i] = initialConditions[i];

	output[idx] = loopCalculateDiscreteModelForFastSynchro(
		Xs,//double* Xs,
		values,//const double* values,
		h, //const double h,
		k_forward,//const double* K_Forward,
		k_backward,//const double* K_Backward,
		iterOfSynchr,//const double iterOfSynchr,
		amountOfNTPoints,//const int amountOfIterations,
		amountOfInitialConditions,//const int amountOfX,
		maxValue,//const double maxValue,
		timedomain,//double* timedomain,
		amountOfInitialConditions*idx* preScaller//const int startDataIndex
	);

	delete[] Xs;

	return;
}



__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const int		amountOfPointsForSkip,
	const int		dimension, 
	double*			ranges, 
	const double	h,
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const int		amountOfInitialConditions, 
	const double*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- ����� ������ � ������ ������ ����� ---
	// --- �������� ������: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ��������� �����...} ---
	extern __shared__ double s[];

	// --- � ������ ������ ������� ��������� �� ��������� � ����������, ����� �������� � ���� ��� � ��������� ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���������� localX[] ���������� ��������� ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- ���������� localValues[] ���������� ����������� ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- ������ �������� ���������� ���������� �� ��������� ������� getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 


	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, idx * sizeOfBlock, 1);

	// --- ������ ��� ��-��������� ���������� ������� --- 
	if (flag == 1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock, 1);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}



__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const double	transientTime,
	const int		dimension,
	double*			ranges,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const double	tMax,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			data,
	int*			maxValueCheckerArray)
{
	// --- ����� ������ � ������ ������ ����� ---
	// --- �������� ������: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ��������� �����...} ---
	extern __shared__ double s[];

	// --- � ������ ������ ������� ��������� �� ��������� � ����������, ����� �������� � ���� ��� � ��������� ---
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���������� localX[] ���������� ��������� ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- ���������� localValues[] ���������� ����������� ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//// --- ������ �������� ���������� ���������� �� ��������� ������� getValueByIdx ---
	//for (int i = 0; i < dimension; ++i)
	//	localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
	//		nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	double h = (double)powf(10, getValueByIdxLog(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0));

	// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, transientTime / h,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- ������ ��� ��-��������� ���������� ������� --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, tMax / h / preScaller,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;
	else
		maxValueCheckerArray[idx] = tMax / h / preScaller;

	return;
}



__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const int		amountOfPointsForSkip,
	const int		dimension, 
	double*			ranges, 
	const double	h,
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const int		amountOfInitialConditions, 
	const double*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- ����� ������ � ������ ������ ����� ---
	// --- �������� ������: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ��������� �����...} ---
	extern __shared__ double s[];

	// --- � ������ ������ ������� ��������� �� ��������� � ����������, ����� �������� � ���� ��� � ��������� ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���������� localX[] ���������� ��������� ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- ���������� localValues[] ���������� ����������� ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- ������ �������� ���������� ���������� �� ��������� ������� getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	//__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	//	const double startRange, const double finishRange, const int valueNumber)
	//{
	//	return startRange + (((int)((int)idx / powf((double)nPts, (double)valueNumber)) % nPts)
	//		* ((double)(finishRange - startRange) / (double)(nPts - 1)));
	//}

	//// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	//bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
	//	amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	//// --- ������ ��� ��-��������� ���������� ������� --- 
	//if (flag)
	//	flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
	//	amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	//// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	//if (!flag && maxValueCheckerArray != nullptr)
	//	maxValueCheckerArray[idx] = -1;
	//else
	//	maxValueCheckerArray[idx] = 1;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// --- ��������� ������� amountOfPointsForSkip ��� ( ��� ��������� transientTime ) --- 
	
	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- ������ ��� ��-��������� ���������� ������� --- 
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}



__global__ void calculateDiscreteModelICCforFastSynchro(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		dimension,
	double* ranges,
	const double	h,
	int* indicesOfMutVars,
	double* initialConditions,
	const int		amountOfInitialConditions,
	const double* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const double	maxValue,
	const int		iterOfSynchr,
	const double*	kForward,
	const double*	kBackward,
	double* data,
	int* maxValueCheckerArray,
	double* FastSynchroError)
{
	// --- ����� ������ � ������ ������ ����� ---
	// --- �������� ������: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ��������� �����...} ---
	extern __shared__ double s[];
	
	// --- � ������ ������ ������� ��������� �� ��������� � ����������, ����� �������� � ���� ��� � ��������� ---
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���������� localX[] ���������� ��������� ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- ���������� localValues[] ���������� ����������� ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- ������ �������� ���������� ���������� �� ��������� ������� getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	// 1 - stability, 0 - fixed point, -1 - unbound solution
	FastSynchroError[idx] = loopCalculateDiscreteModelForFastSynchro_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, maxValue, iterOfSynchr, kForward, kBackward, data, idx * sizeOfBlock);

	// --- ���� ������� ������������� ������ false - ������ �� ���� �� ����� �������� �� ��� ������� � ���������� ������� ---

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}


// --- �������, ������� ������� ������ � ������������������ �������� ---
__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return startRange + ( ( (int)( (int)idx / pow( (double)nPts, (double)valueNumber) ) % nPts )* ( (double)( finishRange - startRange ) / (double)( nPts - 1 ) ) );
}



// --- �������, ������� ������� ������ � ������������������ �������� ---
__device__ __host__ double getValueByIdxLog(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return log10(startRange) + (((int)((int)idx / pow((double)nPts, (double)valueNumber)) % nPts)
		* ((double)(log10(finishRange) - log10(startRange)) / (double)(nPts - 1)));
}



// ---------------------------------------------------------------------------------------------------
// --- ������� ���� � ��������� [startDataIndex; startDataIndex + amountOfPoints] � "data" ������� ---
// ---------------------------------------------------------------------------------------------------
//peakFinder(data, idx* sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

__device__ __host__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- ���������� ��� �������� ��������� ����� ---
	int amountOfPeaks = 0;

	// --- �������� ������������� �������� �������� �� ������� ����� ---
	for ( int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i )
	{
		// --- ���� ������� ����� ������ ���������� � ������ ��� ����� ���������, ��... ( �� ����, ��� ��� ��� ( ��������: 2 3 3 4 ) ) ---
		if ( data[i] - data[i - 1] > 1e-13 && data[i] >= data[i + 1] ) //&&data[j] > 0.2
		{
			// --- �� ��������� ����� �������� ���� ������, ���� �� ��������� �� ����� ������ ������ ��� ������ ---
			for ( int j = i; j < startDataIndex + amountOfPoints - 1; ++j )
			{
				// --- ���� ���������� �� ����� ������ ������, ������ ��� ��� �� ��� ---
				if ( data[j] < data[j + 1] )
				{
					i = j + 1;	// --- ��������� ������� �������, ����� ������ �� ��������� ���� � ��� �� ��������
					break;		// --- ������������ � �������� �����
				}
				// --- ���� � ����, �� ����� ����� ������, ��� �������, ������ �� ����� ��� ---
				if ( data[j] - data[j + 1] > 1e-13  ) //&&data[j] > 0.2
				{
					// --- ���� ������ outPeaks �� ����, �� ������ ������ ---
					if ( outPeaks != nullptr )
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					// --- ���� ������ timeOfPeaks �� ����, �� ������ ������ ---
					if ( timeOfPeaks != nullptr )
						timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (double)j + (double)i ) / (double)2 );	// �������� ������ ���������� ����� j � i
					++amountOfPeaks;
					i = j + 1; // ������ ��� ��������� ����� ����� �� ����� ���� ����� ( ��� ���� �� ����� ���� ������ )
					break;
				}
			}
		}
	}
	// --- ��������� ���������� ��������� ---
	if ( amountOfPeaks > 1 ) {
		// --- ����������� �� ���� ��������� ����� � �� �������� ---
		for ( size_t i = 0; i < amountOfPeaks - 1; i++ )
		{
			// --- ������� ��� ���� �� ���� ������ �����, � ������ ��� ������� ---
			if ( outPeaks != nullptr )
				outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
			// --- ��������� ���������� ��������. ��� ������� ������� ���������� ����� � �����������, ���������� �� ��� ---
			if ( timeOfPeaks != nullptr )
				timeOfPeaks[startDataIndex + i] = (double)( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h;
		}
		// --- ��� ��� ���� ��� ������� - �������� ������� �� ���������� ---
		amountOfPeaks = amountOfPeaks - 1;
	}
	else {
		amountOfPeaks = 0;
	}


	return amountOfPeaks;
}



__device__ __host__ int peakFinder_for_neuronClasses(double* data, const int startDataIndex,
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double* valleys, double* TimeOfValleys, double h)
{
	// --- ���������� ��� �������� ��������� ����� ---
	int amountOfPeaks = 0;
	int amountOfValleys = 0;
	// --- �������� ������������� �������� �������� �� ������� ����� ---
	for (int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i)
	{
		// --- ���� ������� ����� ������ ���������� � ������ ��� ����� ���������, ��... ( �� ����, ��� ��� ��� ( ��������: 2 3 3 4 ) ) ---
		if (data[i] - data[i - 1] > 1e-9 && data[i] >= data[i + 1] && data[i] > 0.11)
		{
			// --- �� ��������� ����� �������� ���� ������, ���� �� ��������� �� ����� ������ ������ ��� ������ ---
			for (int j = i; j < startDataIndex + amountOfPoints - 1; ++j)
			{
				// --- ���� ���������� �� ����� ������ ������, ������ ��� ��� �� ��� ---
				if (data[j] < data[j + 1])
				{
					i = j + 1;	// --- ��������� ������� �������, ����� ������ �� ��������� ���� � ��� �� ��������
					break;		// --- ������������ � �������� �����
				}
				// --- ���� � ����, �� ����� ����� ������, ��� �������, ������ �� ����� ��� ---
				if (data[j] - data[j + 1] > 1e-9)
				{
					// --- ���� ������ outPeaks �� ����, �� ������ ������ ---
					if (outPeaks != nullptr)
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					// --- ���� ������ timeOfPeaks �� ����, �� ������ ������ ---
					if (timeOfPeaks != nullptr)
						timeOfPeaks[startDataIndex + amountOfPeaks] = h * (double)(i - startDataIndex);	// �������� ������ ���������� ����� j � i
					++amountOfPeaks;
					i = j + 1; // ������ ��� ��������� ����� ����� �� ����� ���� ����� ( ��� ���� �� ����� ���� ������ )
					break;
				}
			}
		}



		// --- ���� ������� ����� ������ ���������� � ������ ��� ����� ���������, ��... ( �� ����, ��� ��� ��� ( ��������: 2 3 3 4 ) ) ---
		//////if (data[i - 1] - data[i] > 1e-9 && data[i] <= data[i + 1] && data[i] > 0.0) //&& data[i] > 0.10)
		//////{
		//////	// --- �� ��������� ����� �������� ���� ������, ���� �� ��������� �� ����� ������ ������ ��� ������ ---
		//////	for (int j = i; j < startDataIndex + amountOfPoints - 1; ++j)
		//////	{
		//////		// --- ���� ���������� �� ����� ������ ������, ������ ��� ��� �� ������� ---
		//////		if (data[j] > data[j + 1])
		//////		{
		//////			i = j + 1;	// --- ��������� ������� �������, ����� ������ �� ��������� ���� � ��� �� ��������
		//////			break;		// --- ������������ � �������� �����
		//////		}
		//////		// --- ���� � ����, �� ����� ����� ������, ��� �������, ������ �� ����� ��� ---
		//////		if (data[j + 1] - data[j] > 1e-9)
		//////		{
		//////			// --- ���� ������ outPeaks �� ����, �� ������ ������ ---
		//////			if (valleys != nullptr)
		//////				valleys[startDataIndex + amountOfValleys] = data[j];
		//////			// --- ���� ������ timeOfPeaks �� ����, �� ������ ������ ---
		//////			if (TimeOfValleys != nullptr)
		//////				TimeOfValleys[startDataIndex + amountOfValleys] = h * (double)(i - startDataIndex);	// �������� ������ ���������� ����� j � i
		//////			++amountOfValleys;
		//////			i = j + 1; // ������ ��� ��������� ����� ����� �� ����� ���� ����� ( ��� ���� �� ����� ���� ������ )
		//////			break;
		//////		}
		//////	}
		//////}
	}

	//if (amountOfPeaks >= amountOfValleys)
	//	return amountOfValleys;
	//else 
		return amountOfPeaks;
}


// ----------------------------------------------------------------
// --- ���������� ����� � "data" ������� � ������������� ������ ---
// ----------------------------------------------------------------

__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if ( amountOfPeaks[idx] == -1 )
	{
		amountOfPeaks[idx] = -1;
		return;
	}

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == 0)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

		amountOfPeaks[idx] = peakFinder( data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h );
	return;
}




__global__ void peakFinderCUDA_for_NeuronClassification(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double* valleys, double* TimeOfValleys,  double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}
	
	amountOfPeaks[idx] = peakFinder_for_neuronClasses(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, valleys, TimeOfValleys, h);
	return;
}



__global__ void peakFinderCUDA_H(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, amountOfPeaks[idx], outPeaks, timeOfPeaks, h);
	return;
}



__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, bool* flags, double ostrovskyThreshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		flags[idx * 5 + 3] = true;
		return;
	}

	double lastPoint = data[idx * sizeOfBlock + sizeOfBlock - 1];

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks);

	//FIRST CONDITION
	flags[idx * 5 + 0] = true;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] - outPeaks[i - 1] > 0)
		{
			flags[idx * 5 + 0] = false;
			break;
		}
	}

	//SECOND & THIRD CONDITION
	bool flagOne = false;
	bool flagZero = false;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] > ostrovskyThreshold)
			flagOne = true;
		else
			flagZero = true;
		if (flagOne && flagZero)
			break;
	}

	if (flagOne && flagZero)
		flags[idx * 5 + 1] = true;
	else
		flags[idx * 5 + 1] = false;

	if (flagOne && !flagZero)
		flags[idx * 5 + 2] = false;
	else
		flags[idx * 5 + 2] = true;

	//FOUR CONDITION
	if (amountOfPeaks[idx] == 0 || amountOfPeaks[idx] == 1)
		flags[idx * 5 + 3] = true;
	else
		flags[idx * 5 + 3] = false;

	//FIVE CONDITION
	if (lastPoint > ostrovskyThreshold)
		flags[idx * 5 + 4] = true;
	else
		flags[idx * 5 + 4] = false;
	return;
}



__device__ __host__ int kde(double* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	if (amountOfPoints == 0)
		return 0;
	if (amountOfPoints == 1 || amountOfPoints == 2)
		return 1;
	if (amountOfPoints > maxAmountOfPeaks)
		return maxAmountOfPeaks;

	double k1 = kdeSampling * amountOfPoints;
	double k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	double delt = 0;
	double prevPrevData2 = 0;
	double prevData2 = 0;
	double data2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < amountOfPoints; ++m)
		{
			double tempData = (data[startDataIndex + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}

		if (w < 2)
			continue;
		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	return resultKde;
}



__global__ void kdeCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		kdeResult[idx] = 0;
		return;
	}
	kdeResult[idx] = kde(data, idx * sizeOfBlock, amountOfPeaks[idx], maxAmountOfPeaks,
		kdeSampling, kdeSamplesInterval1, kdeSamplesInterval2, kdeSmoothH);
}


// ------------------------------------------------
// --- ��������� ���������� ����� ����� ������� ---
// ------------------------------------------------

__device__ __host__ double distance(double x1, double y1, double x2, double y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	double dx = x2 - x1;
	double dy = y2 - y1;

	return hypotf(dx, dy);
}



// ----------------------
// --- ������� DBSCAN ---
// ----------------------

__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{
	// ------------------------------------------------------------
	// --- ���� ����� 0 ��� 1 - ���� �� ������������ ��� ������ ---
	// ------------------------------------------------------------

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	//if (amountOfPeaks >= 3600)
	//	return 0;


	// ------------------------------------------------------------


	int cluster = 0;
	int NumNeibor = 0;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	// ------------------------------------------------------------
	//for (int i = 0; i < amountOfPeaks; i++) {
	//	helpfulArray[startDataIndex + i] = (int)(100*sqrt(data[startDataIndex + i] * data[startDataIndex + i] + intervals[startDataIndex + i] * intervals[startDataIndex + i]));
	//}

	for (int i = 0; i < amountOfPeaks; i++) {
		data[startDataIndex + i] = 0; 
		//intervals[startDataIndex + i] = intervals[startDataIndex + i];
	}
	// ------------------------------------------------------------

	for (int i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1];
			helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}
		else if (helpfulArray[startDataIndex + i] == 0) {
			NumNeibor = 0;
			++cluster;
			helpfulArray[startDataIndex + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}

	return cluster - 1;
}

__device__ __host__ int NeuronClassification(double* data, double* intervals, double* valleys, double* TimeOfValleys, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{
	// ------------------------------------------------------------
	// --- ���� ����� 0 ��� 1 - ���� �� ������������ ��� ������ ---
	// ------------------------------------------------------------

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 0;

	// ------------------------------------------------------------


	int classOfNeuron = 0;

	int counter = 0;
	int index_new = 0;
	int index_write = 0;
	int index = 99999999;



	double a28 = -0.002;
	double a26 = 0.05;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	// ------------------------------------------------------------
	// ���������� � ������ ���� ��� ����� ���� �������� ��� ������� 0 �������
	for (int i = 0; i < amountOfPeaks; i++) {

		//index_new = (int)floor((intervals[startDataIndex + i] + a28) / a26);
		index_new = (int)floor((intervals[startDataIndex + i]) / a26);
		if (index_new > index) {

			helpfulArray[startDataIndex + index] = counter;
			//index_write++;
			counter = 0;
		}
		counter++;
		index = index_new;
		
	}
	index_write = index;
	// ------------------------------------------------------------
	
	//return index_write;



	int index_max = 0;
	//int index_min = 0;
	double maxFreq = -1;
	double minFreq = 999999999;

	for (int i = 0; i < index_write; i++) {
		if (helpfulArray[startDataIndex + i] > maxFreq) {
			maxFreq = helpfulArray[startDataIndex + i];
			index_max = i;
		}
		if (helpfulArray[startDataIndex + i] < minFreq && helpfulArray[startDataIndex + i] >= 1) {
			minFreq = helpfulArray[startDataIndex + i];
			//index_max = i;
		}
	}

	double time_last_peak_0 = 0;
	double time_last_peak_1 = 0;
	double time_last_peak_2 = 0;

	//bool flag_0 = 1;
	//int ind_0 = 0;

	if (maxFreq <= 1)
		classOfNeuron = 3;

	for (int i = 0; i < index_write; i++) {
		if (helpfulArray[startDataIndex + i] >= 1 && helpfulArray[startDataIndex + i + 1] >= 2 && helpfulArray[startDataIndex + i + 1] <= 5) {
			classOfNeuron = 1;
			break;
		}
		if (helpfulArray[startDataIndex + i] >= 1 && helpfulArray[startDataIndex + i + 1] > 5 ) {
			classOfNeuron = 2;
			break;
		}
	}
	
	for (int i = 0; i < amountOfPeaks; i++) {
		if (intervals[startDataIndex + i] > a26 * (double)index_max && intervals[startDataIndex + i] <= a26 * (double)(index_max + 1)) {
			time_last_peak_0 = intervals[startDataIndex + i];
		}
		if (intervals[startDataIndex + i] > a26 * (double)(index_max + 1) && intervals[startDataIndex + i] <= a26 * (double)(index_max + 2)) {
			time_last_peak_1 = intervals[startDataIndex + i];
		}
		if (intervals[startDataIndex + i] > a26 * (double)(index_max + 2) && intervals[startDataIndex + i] <= a26 * (double)(index_max + 3)) {
			time_last_peak_2 = intervals[startDataIndex + i];
		}
	}

	if (((a26 * (double)(index_max + 1) - time_last_peak_0) >= a26 * 0.3f) && data[startDataIndex + (int)((double)a26 / (double)(2e-6f) * ((double)(index_max + 1))) - 10] >= 0.05f)
		classOfNeuron = 4;

	if (((a26 * (double)(index_max + 2) - time_last_peak_1) >= a26 * 0.3f) && data[startDataIndex + (int)((double)a26 / (double)(2e-6f) * ((double)(index_max + 2))) - 10] >= 0.05f)
		classOfNeuron = 4;

	if (((a26 * (double)(index_max + 3) - time_last_peak_2) >= a26 * 0.3f) && data[startDataIndex + (int)((double)a26 / (double)(2e-6f) * ((double)(index_max + 3))) - 10] >= 0.05f)
		classOfNeuron = 4;

	//if (maxFreq <= 15) {
	//	classOfNeuron = 1;
	//}
	//if (maxFreq <= 1) {
	//	classOfNeuron = 3;
	//}
	//if (maxFreq > 15) {
	//	classOfNeuron = 2;
	//}
	
	//for (int i = 0; i < index_write; i++) {
	//	if (maxFreq <= 20) {
	//		classOfNeuron = 3;
	//	}
	//}
	////return maxFreq;

	int delt_i = 0;
	for (int i = 0; i < index_max; i++) {
		if (helpfulArray[startDataIndex + i] >= 2) {
			delt_i = index_max - i + 1;
			break;
		}
	}

	//for (int i = 1; i < index_max; i++) {
	//	if (helpfulArray[startDataIndex + i-1] <= 4 && helpfulArray[startDataIndex + i] > 4 &&  helpfulArray[startDataIndex + i] <= 8 && maxFreq >= 8) {
	//		classOfNeuron = 1;
	//		//return classOfNeuron + 100 * delt_i;
	//	}
	//}

	//for (int i = 1; i < index_max; i++) {
	//	if (helpfulArray[startDataIndex + i - 1] <= 4 && helpfulArray[startDataIndex + i] > 8 || maxFreq >= 40) {
	//		classOfNeuron = 2;
	//		//return classOfNeuron + 100 * delt_i;
	//	}
	//}

	//if (classOfNeuron == 3)
	//	delt_i = 1;

	//return classOfNeuron+1000*delt_i;

	//return (int)((a26* (double)(index_max + 1) - time_last_peak)*10000) + 1000 * delt_i + 1000000 * classOfNeuron;

	return maxFreq + 1000 * delt_i + 1000000*classOfNeuron;
}

// ---------------------------------
// --- ���������� ������� DBSCAN ---
// ---------------------------------

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	//outData[idx] = idx;
	//return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == -1)
	{
		outData[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		outData[idx] = 0;
		return;
	}

	// --- ��������� �������� dbscan � ������ �������
	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, 
		idx, eps, outData);
}



__global__ void NeuronClassificationCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* valleys, double* TimeOfValleys,  double* helpfulArray,
	const double eps, int* outData)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	if (amountOfPeaks[idx] == -1)
	{
		outData[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		outData[idx] = 0;
		return;
	}

	// --- ��������� �������� dbscan � ������ �������
	outData[idx] = NeuronClassification(data, intervals, valleys, TimeOfValleys, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
	//  outData[idx] =               dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
}

// --------------------
// --- ���� ��� LLE ---
// --------------------
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	double zPower = 0;


	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0;
	}


	// --- �� ������ ������ -2, ��� ���������� ��� ������ ������� AND-TS
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		curand_init(seed+i, 0, 0, &state);
		z[i] = curand_uniform(&state)-0.5;
		//z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = z[i]/zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = x[i] + z[i] * eps;
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;
		double tempData2 = 0;

		// --- �� ������ ������ -2, ��� ���������� ��� ������ ������� AND-TS
		for (int l = 0; l < amountOfInitialConditions; ++l) {
			tempData2 = 1/eps;
			tempData2 = tempData2 * (x[l] - y[l]);
			tempData += tempData2 * tempData2;
		}

		tempData = sqrt(tempData);

		//for (int l = 0; l < amountOfInitialConditions; ++l) 
		//	tempData += (x[l] - y[l]) * (x[l] - y[l]);

		//tempData = sqrt(tempData) / eps;

		result += log(tempData);
		
		//if (tempData != 0)
			//tempData = (1 / tempData);

		// --- �� ������ ������ -2, ��� ���������� ��� ������ ������� AND-TS
		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) / tempData));
		}
		// --- � ��� �� ������ ������, ��� ��� ������ ������� AND-TS
		//y[2] = x[2];
		//y[3] = x[3];
	}

	resultArray[idx] = result / tMax;
}



// -------------------------
// --- ���� ��� LLE (IC) ---
// -------------------------
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	//printf("%f %f %f %f\n", localValues[0], localValues[1], localValues[2], localValues[3]);

	double zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		// z[i] = sinf(0.2171828 * (i + 1) + idx + (0.2171828 + i * idx)) * 0.5;
		z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l)
			tempData += (x[l] - y[l]) * (x[l] - y[l]);
		tempData = sqrt(tempData) / eps;

		result += log(tempData);

		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



//find projection operation (ab)
__device__ __host__ void projectionOperator(double* a, double* b, double* minuend, int amountOfValues)
{
	double numerator = 0;
	double denominator = 0;
	for (int i = 0; i < amountOfValues; ++i)
	{
		numerator += a[i] * b[i];
		denominator += b[i] * b[i];
	}

	double fraction = denominator == 0 ? 0 : numerator / denominator;

	for (int i = 0; i < amountOfValues; ++i)
		minuend[i] -= fraction * b[i];
}



__device__ __host__ void gramSchmidtProcess(double* a, double* b, int amountOfVectorsAndValuesInVector, double* denominators=nullptr/*They are is equale for our task*/)
{
	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[j + i * amountOfVectorsAndValuesInVector] = a[j + i * amountOfVectorsAndValuesInVector];

		for (int j = 0; j < i; ++j)
			projectionOperator(a + i * amountOfVectorsAndValuesInVector,
				b + j * amountOfVectorsAndValuesInVector,
				b + i * amountOfVectorsAndValuesInVector,
				amountOfVectorsAndValuesInVector);
	}

	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		double denominator = 0;
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			denominator += b[i * amountOfVectorsAndValuesInVector + j] * b[i * amountOfVectorsAndValuesInVector + j];
		denominator = sqrt(denominator);
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[i * amountOfVectorsAndValuesInVector + j] = denominator == 0 ? 0 : b[i * amountOfVectorsAndValuesInVector + j] / denominator;

		if (denominators != nullptr)
			denominators[i] = denominator;
	}
}



__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller,
	const int writableVar,
	const double maxValue,
	double* resultArray)
{
	extern __shared__ double s[];

	unsigned long long buferForMem = 0;
	double* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	double* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		double zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating


	gramSchmidtProcess(z, y, amountOfInitialConditions);


	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m ) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error;*/ }

		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints,
				amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error; */ }
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		// 
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (double)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}

// ------------------------------------------------------------------
// --- ���������� �������� �������� ����� � ���������� ���������� ---
// ------------------------------------------------------------------

__global__ void avgPeakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* systemCheker, double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// 1 - stability, -1 - fixed point, 0 - unbound solution
	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock-1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	//if (outAvgPeaks[idx] == -1)
	//{
	//	outAvgPeaks[idx] = NAN;
	//	AvgTimeOfPeaks[idx] = NAN;
	//	return;
	//}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	//__device__ __host__ int peakFinder(double* data, const int startDataIndex,
	//	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0) 
	{
		outAvgPeaks[idx] = 1000;
		AvgTimeOfPeaks[idx] = 1000;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}

	outAvgPeaks[idx] /= amountOfPeaks;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	return;
}

__global__ void avgPeakFinderCUDA_for2Dbif(double* data, const int sizeOfBlock, const int amountOfBlocks,
	double* outAvgPeaks, double* AvgTimeOfPeaks, double* outPeaks, double* timeOfPeaks, int* PeaksAmount, int* systemCheker, double h)
{
	// --- ��������� ������ ������, � ������� ��������� � ����� ������ ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// ���� ���������� ����� � ������� ��������, ��� ��������� - ����� ��������� ���
		return;

	// 1 - stability, -1 - fixed point, 0 - unbound solution
	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		PeaksAmount[idx] = 0;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock - 1];
		AvgTimeOfPeaks[idx] = -1.0;
		PeaksAmount[idx] = 0;
		return;
	}

	// --- ���� �� ���������� ������ ������� ��� �������� ��� "�����������", �� ���������� �� ---
	//if (outAvgPeaks[idx] == -1)
	//{
	//	outAvgPeaks[idx] = NAN;
	//	AvgTimeOfPeaks[idx] = NAN;
	//	return;
	//}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	//__device__ __host__ int peakFinder(double* data, const int startDataIndex,
	//	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0)
	{
		outAvgPeaks[idx] = 1000;
		AvgTimeOfPeaks[idx] = 1000;
		PeaksAmount[idx] = 0;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}
	PeaksAmount[idx] = amountOfPeaks;
	outAvgPeaks[idx] /= amountOfPeaks;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	return;
}

__global__ void CUDA_dbscan_kernel(double* data, double* intervals, int* labels,
	const int amountOfData, const double eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// ��������� ������� ������ ������
	if (idx >= amountOfData)								// ���� ������ ������ - ������������ �� ������
		return;

	//if (idx == 0 && labels[idxCurPoint] == 0)					// ���� � idxCurPoint ����� ��� ��� �������� - ����� ��� (������ ��� ������ ����� � ��������)
	//	labels[idxCurPoint] = atomicAdd(amountOfClusters, 1);

	labels[idxCurPoint] = amountOfClusters;

	if (labels[idx] != 0)									// � ����� ��� ���� ������� - ������������ �� ������
		return;

	if (idx == idxCurPoint)									// ���� ������������� ������� ����� - ������������ �� ������
		return;

	// ���� ���������� ����� ��������������� ������ idx � ������� ������ idxCurPoint <= eps - ����� ����� �������

	if (helpfulArray[idxCurPoint] == 0) {
		labels[idxCurPoint] = 0;
		return;
	}

	if (sqrt((data[idxCurPoint] - data[idx]) * (data[idxCurPoint] - data[idx]) + (intervals[idxCurPoint] - intervals[idx]) * (intervals[idxCurPoint] - intervals[idx])) <= eps)
	{
		labels[idx] = labels[idxCurPoint];						// ���� ����� �������. ������������, ��� � idxCurPoint �� ����� �� ���� ��������
		neighbors[atomicAdd(amountOfNeighbors, 1)] = idx;		// ��������� ������ ���������� ������ - ��� ���� �� �� ������
	}
}



__global__ void CUDA_dbscan_search_clear_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// ��������� ������� ������ ������
	if (idx >= amountOfData)								// ���� ������ ������ - ������������ �� ������
		return;

	if (labels[idx] == 0 && helpfulArray[idx] == 1)
	{
		*res = idx;
		return;
	}
}



__global__ void CUDA_dbscan_search_fixed_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// ��������� ������� ������ ������
	if (idx >= amountOfData)								// ���� ������ ������ - ������������ �� ������
		return;

	if (helpfulArray[idx] == -1 && labels[idx] == 0)
	{
		*res = idx;
		return;
	}
}

__global__ void CUDA_dbscan_search_unbound_points_kernel(double* data, double* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// ��������� ������� ������ ������
	if (idx >= amountOfData)								// ���� ������ ������ - ������������ �� ������
		return;

	if (helpfulArray[idx] == 0 && labels[idx] == 0)
	{
		*res = idx;
		return;
	}
}


//////int N = 3;
//////int i, j;
//////double x0, x1, y, K, b, flag, T2, X1[3], k[3][4], d0, m0, R, P, d, m, val, flin;
//////double eps = 2.2204e-16;
//////double x[2];

//////d0 = 1;
//////m0 = 1;

//////R = 2;
//////P = 1.25;

//////for (i = 0; i < N; i++) {
//////	X1[i] = X[i];
//////}

//////for (j = 0; j < 4; j++)
//////{

//////	d = d0;
//////	m = m0;
//////	y = X1[1];

//////	while (1) {
//////		if (abs(y) < d) {
//////			d = d / R;
//////			m = m / R;
//////			continue;
//////		}
//////		if (abs(y) > R * d) {
//////			d = d * R;
//////			m = m * R;
//////			continue;
//////		}
//////		break;
//////	}
//////	if (d > eps) {
//////		if (abs(y) < P * d) {
//////			x[1] = (((R * R + R) / (R * R)) * m - m) / (P * d - d);
//////			x[0] = m - d * x[1];

//////			val = x[1] * abs(y) + x[0];
//////		}
//////		else {
//////			x[1] = (R * m - ((R * R + R) / (R * R)) * m) / (R * d - P * d);
//////			x[0] = ((R * R + R) / (R * R)) * m - P * d * x[1];

//////			val = x[1] * abs(y) + x[0];
//////		}
//////	}
//////	else {
//////		val = 0;
//////	}

//////	k[0][j] = a[0] * X1[2];
//////	k[1][j] = a[1] * X1[1] + X1[2];
//////	k[2][j] = -X1[0] + X1[1] + a[2] * val;


//////	if (j == 3) {
//////		for (i = 0; i < N; i++) {
//////			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
//////		}
//////	}
//////	else if (j == 2) {
//////		for (i = 0; i < N; i++) {
//////			X1[i] = X[i] + h * k[i][j];
//////		}
//////	}
//////	else {
//////		for (i = 0; i < N; i++) {
//////			X1[i] = X[i] + 0.5 * h * k[i][j];
//////		}
//////	}
//////}


////Rossler 01_012 // 01_021
	//double h1 = 0.5*h + a[0];
	//double h2 = 0.5*h - a[0];
	//X[0] = X[0] - h1 * (X[1] + X[2]);
	//X[1] = X[1] + h1 * (X[0] + X[1] * a[1]);
	//X[2] = X[2] + h1 * (a[2] + X[2] * (X[0] - a[3]));

	//X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));
	//X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);
	//X[0] = X[0] - h2 * (X[1] + X[2]);

////Rossler 01_102
//	double h1 = 0.5 * h + a[0];
//	double h2 = 0.5 * h - a[0];
//	X[1] = X[1] + h1 * (X[0] + X[1] * a[1]);
//	X[0] = X[0] - h1 * (X[1] + X[2]);
//	X[2] = X[2] + h1 * (a[2] + X[2] * (X[0] - a[3]));

//	X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));
//	X[0] = X[0] - h2 * (X[1] + X[2]);
//	X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);

////Rossler 01_120 // 01_210
//	double h1 = 0.5 * h + a[0];
//	double h2 = 0.5 * h - a[0];
//	X[1] = X[1] + h1 * (X[0] + X[1] * a[1]);
//	X[2] = X[2] + h1 * (a[2] + X[2] * (X[0] - a[3]));
//	X[0] = X[0] - h1 * (X[1] + X[2]);

//	X[0] = X[0] - h2 * (X[1] + X[2]);
//	X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));
//	X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);

////Rossler 01_201 
//	double h1 = 0.5 * h + a[0];
//	double h2 = 0.5 * h - a[0];
//	X[2] = X[2] + h1 * (a[2] + X[2] * (X[0] - a[3]));
//	X[0] = X[0] - h1 * (X[1] + X[2]);
//	X[1] = X[1] + h1 * (X[0] + X[1] * a[1]);

//	X[1] = (X[1] + h2 * X[0]) / (1 - h2 * a[1]);
//	X[0] = X[0] - h2 * (X[1] + X[2]);
//	X[2] = (X[2] + h2 * a[2]) / (1 - h2 * (X[0] - a[3]));

////Rossler 10_012 // 10_021
//	double h1 = 0.5*h + a[0];
//	double h2 = 0.5*h - a[0];
//	X[0] = X[0] - h1 * (X[1] + X[2]);
//	X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[1]);
//	X[2] = (X[2] + h1 * a[2]) / (1 - h1 * (X[0] - a[3]));

//	X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));
//	X[1] = X[1] + h2 * (X[0] + X[1] * a[1]);
//	X[0] = X[0] - h2 * (X[1] + X[2]);

////Rossler 10_102
//	double h1 = 0.5 * h + a[0];
//	double h2 = 0.5 * h - a[0];
//	X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[1]);
//	X[0] = X[0] - h1 * (X[1] + X[2]);
//	X[2] = (X[2] + h1 * a[2]) / (1 - h1 * (X[0] - a[3]));

//	X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));
//	X[0] = X[0] - h2 * (X[1] + X[2]);
//	X[1] = X[1] + h2 * (X[0] + X[1] * a[1]);

////Rossler 10_120 // 10_210
//	double h1 = 0.5 * h + a[0];
//	double h2 = 0.5 * h - a[0];
//	X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[1]);
//	X[2] = (X[2] + h1 * a[2]) / (1 - h1 * (X[0] - a[3]));
//	X[0] = X[0] - h1 * (X[1] + X[2]);

//	X[0] = X[0] - h2 * (X[1] + X[2]);
//	X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));
//	X[1] = X[1] + h2 * (X[0] + X[1] * a[1]);

//Rossler 10_201
	//double h1 = 0.5 * h + a[0];
	//double h2 = 0.5 * h - a[0];
	//X[2] = (X[2] + h1 * a[2]) / (1 - h1 * (X[0] - a[3]));
	//X[0] = X[0] - h1 * (X[1] + X[2]);
	//X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[1]);

	//X[1] = X[1] + h2 * (X[0] + X[1] * a[1]);
	//X[0] = X[0] - h2 * (X[1] + X[2]);
	//X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));

//double x0, x1, y, K, b, flag, T2, X1[3], k[2][3];
//double m = 10;
//double c = 0.16;

//if (X[0] < 0)
//	x0 = -1;
//else if (X[0] > 0)
//	x0 = 1;
//else
//	x0 = 0;

//flag = 1;
//while (flag == 1) {
//	x1 = m * x0;
//	if ((X[0] > 0) && (X[0] > x1) || (X[0] < 0) && (X[0] < x1)) {
//		x0 = x0 * m;
//	}
//	else {
//		if ((X[0] >= x0) && (X[0] <= x1) || (X[0] <= x0) && (X[0] >= x1)) {
//			flag = 0;
//			T2 = (m - 1) * x0 / 2;
//			if ((X[0] >= 0 && X[0] < x0 + T2) || (X[0] < 0 && X[0] > x0 + T2)) {
//				K = (-3 - m) / (m - 1);
//				b = (1 - (-3 - m) / (m - 1)) * x0;
//			}
//			else {
//				K = (3 * m + 1) / (m - 1);
//				b = (1 - (3 * m + 1) / (m - 1)) * m * x0;
//			}
//			y = K * X[0] + b;
//		}
//		else {
//			if ((X[0] > 0) && (X[0] < x0) || (X[0] < 0) && (X[0] > x0))
//				x0 = x0 / m;
//		}
//	}
//}

//k[0][0] = a[0] * (X[1] + c * y);
//k[0][1] = X[0] - X[1] + X[2];
//k[0][2] = -a[1] * X[1] - a[2] * X[2];

//X1[0] = X[0] + h * k[0][0];
//X1[1] = X[1] + h * k[0][1];
//X1[2] = X[2] + h * k[0][2];

//if (X1[0] < 0)
//	x0 = -1;
//else if (X1[0] > 0)
//	x0 = 1;
//else
//	x0 = 0;

//flag = 1;
//while (flag == 1) {
//	x1 = m * x0;
//	if ((X1[0] > 0) && (X1[0] > x1) || (X1[0] < 0) && (X1[0] < x1)) {
//		x0 = x0 * m;
//	}
//	else {
//		if ((X1[0] >= x0) && (X1[0] <= x1) || (X1[0] <= x0) && (X1[0] >= x1)) {
//			flag = 0;
//			T2 = (m - 1) * x0 / 2;
//			if ((X1[0] >= 0 && X1[0] < x0 + T2) || (X1[0] < 0 && X1[0] > x0 + T2)) {
//				K = (-3 - m) / (m - 1);
//				b = (1 - (-3 - m) / (m - 1)) * x0;
//			}
//			else {
//				K = (3 * m + 1) / (m - 1);
//				b = (1 - (3 * m + 1) / (m - 1)) * m * x0;
//			}
//			y = K * X1[0] + b;
//		}
//		else {
//			if ((X1[0] > 0) && (X1[0] < x0) || (X1[0] < 0) && (X1[0] > x0))
//				x0 = x0 / m;
//		}
//	}
//}

//k[1][0] = a[0] * (X1[1] + c * y);
//k[1][1] = X1[0] - X1[1] + X1[2];
//k[1][2] = -a[1] * X1[1] - a[2] * X1[2];

//X[0] = X[0] + 0.5 * h * (k[0][0] + k[1][0]);
//X[1] = X[1] + 0.5 * h * (k[0][1] + k[1][1]);
//X[2] = X[2] + 0.5 * h * (k[0][2] + k[1][2]);


//////Bouali
	//CD
		//double k = 0.50928;
		//double h1 = h * k;
		//double h2 = h * (1 - k);

		//X[1] = X[1] / (1 + h1 * a[2] * (1 - X[0] * X[0]));
		//X[0] = (X[0] - h1 * a[1] * X[2]) / (1 - h1 * a[0] * (1 - X[1]));
		//X[2] = X[2] + h1 * a[3] * X[0];
		//X[2] = X[2] + h2 * a[3] * X[0];
		//X[0] = X[0] + h2 * (a[0] * X[0] * (1 - X[1]) - a[1] * X[2]);
		//X[1] = X[1] + h2 * (-a[2] * X[1] * (1 - X[0] * X[0]));
	//Euler
		//double X1[3];
		//X[0] = X[0] + h * (a[0] * X[0] * (1 - X[1]) - a[1] * X[2]);
		//X[1] = X[1] + h * (-a[2] * X[1] * (1 - X[0] * X[0]));
		//X[2] = X[2] + h * a[3] * X[0];
	//IMP
/*		double w[3][4], Z[3], A[3], Zn[3], dz, J[3][3];
		double tol;
		int i; int j;
		tol = 1e-14;
		int nnewtmAX = 10, nnewt;

		dz = 1e-13;
		nnewt = 0;
		J[0][2] = -a[1];
		J[1][2] = 0;
		J[2][0] = a[3];
		J[2][1] = 0;
		J[2][2] = 0;

		Z[0] = X[0]; Z[1] = X[1]; Z[2] = X[2];

		while ((dz > tol) && (nnewt < nnewtmAX)) {

			J[0][0] = a[0] * (1 - Z[1]);
			J[0][1] = -a[0] * Z[0];
			J[1][0] = 2 * Z[0] * a[2] * Z[1];
			J[1][1] = -a[2] * (1 - Z[0] * Z[0]);

			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					if (i == j)
						w[i][j] = 1 - 0.5 * h * J[i][j];
					else
						w[i][j] = -0.5 * h * J[i][j];
				}
			}

			A[0] = 0.5 * (X[0] + Z[0]);
			A[1] = 0.5 * (X[1] + Z[1]);
			A[2] = 0.5 * (X[2] + Z[2]);

			w[0][3] = X[0] - Z[0] + h * (a[0] * A[0] * (1 - A[1]) - a[1] * A[2]);
			w[1][3] = X[1] - Z[1] + h * (-a[2] * A[1] * (1 - A[0] * A[0]));
			w[2][3] = X[2] - Z[2] + h * a[3] * A[0];

			int HEIGHT = 3;
			int WIDTH = 4;
			int k; double t; double d;

			for (k = 0; k <= HEIGHT - 2; k++) {

				int l = k;

				for (i = k + 1; i <= HEIGHT - 1; i++) {
					if (abs(w[i][k]) > abs(w[l][k])) {
						l = i;
					}
				}
				if (l != k) {
					for (j = 0; j <= WIDTH - 1; j++) {
						if ((j == 0) || (j >= k)) {
							t = w[k][j];
							w[k][j] = w[l][j];
							w[l][j] = t;
						}
					}
				}

				d = 1.0 / w[k][k];
				for (i = (k + 1); i <= (HEIGHT - 1); i++) {
					if (w[i][k] == 0) {
						continue;
					}
					t = w[i][k] * d;
					for (j = k; j <= (WIDTH - 1); j++) {
						if (w[k][j] != 0) {
							w[i][j] = w[i][j] - t * w[k][j];
						}
					}
				}
			}

			for (i = (HEIGHT); i >= 2; i--) {
				for (j = 1; j <= i - 1; j++) {
					t = w[i - j - 1][i - 1] / w[i - 1][i - 1];
					w[i - j - 1][WIDTH - 1] = w[i - j - 1][WIDTH - 1] - t * w[i - 1][WIDTH - 1];
				}
				w[i - 1][WIDTH - 1] = w[i - 1][WIDTH - 1] / w[i - 1][i - 1];
			}
			w[0][WIDTH - 1] = w[0][WIDTH - 1] / w[0][0];
			Zn[0] = Z[0] + w[0][3];
			Zn[1] = Z[1] + w[1][3];
			Zn[2] = Z[2] + w[2][3];

			dz = sqrt((Zn[0] - Z[0]) * (Zn[0] - Z[0]) + (Zn[1] - Z[1]) * (Zn[1] - Z[1]) + (Zn[2] - Z[2]) * (Zn[2] - Z[2]));
			Z[0] = Zn[0];
			Z[1] = Zn[1];
			Z[2] = Zn[2];

			nnewt++;
		}
		X[0] = Zn[0];
		X[1] = Zn[1];
		X[2] = Zn[2];*/


		//EMP
			//double X1[3];
			//double h1 = h / 2;
			//X1[0] = X[0] + h1 * (a[0] * X[0] * (1 - X[1]) - a[1] * X[2]);
			//X1[1] = X[1] + h1 * (-a[2] * X[1] * (1 - X[0] * X[0]));
			//X1[2] = X[2] + h1 * a[3] * X[0];
			//
			//X[0] = X[0] + h * (a[0] * X1[0] * (1 - X1[1]) - a[1] * X1[2]);
			//X[1] = X[1] + h * (-a[2] * X1[1] * (1 - X1[0] * X1[0]));
			//X[2] = X[2] + h * a[3] * X1[0];
		//SIMP
			//double X1[3];
			//double h1 = h / 2;
			//X1[0] = X[0]; X1[1] = X[1]; X1[2] = X[2];
			//X1[0] = (X1[0] - h1 * a[1] * X1[2]) / (1 - h1 * a[0] * (1 - X1[1]));
			//X1[1] = X1[1] / (1 + h1 * a[2] * (1 - X1[0] * X1[0]));
			//X1[2] = X1[2] + h1 * a[3] * X1[0];
			//X[0] = X[0] + h * (a[0] * X1[0] * (1 - X1[1]) - a[1] * X1[2]);
			//X[1] = X[1] + h * (-a[2] * X1[1] * (1 - X1[0] * X1[0]));
			//X[2] = X[2] + h * a[3] * X1[0];

		//RK4
			//double k11, k21, k31, k12, k22, k32, k13, k23, k33, k14, k24, k34;
			//double X1[3];
			//
			//k11 = (a[0] * X[0] * (1 - X[1]) - a[1] * X[2]);
			//k21 = -a[2] * X[1] * (1 - X[0] * X[0]);
			//k31 = a[3] * X[0];
			//
			//X1[0] = X[0] + 0.5 * h * k11;
			//X1[1] = X[1] + 0.5 * h * k21;
			//X1[2] = X[2] + 0.5 * h * k31;
			//
			//k12 = (a[0] * X1[0] * (1 - X1[1]) - a[1] * X1[2]);
			//k22 = -a[2] * X1[1] * (1 - X1[0] * X1[0]);
			//k32 = a[3] * X1[0];
			//
			//X1[0] = X[0] + 0.5 * h * k12;
			//X1[1] = X[1] + 0.5 * h * k22;
			//X1[2] = X[2] + 0.5 * h * k32;
			//
			//k13 = (a[0] * X1[0] * (1 - X1[1]) - a[1] * X1[2]);
			//k23 = -a[2] * X1[1] * (1 - X1[0] * X1[0]);
			//k33 = a[3] * X1[0];
			//
			//X1[0] = X[0] + h * k13;
			//X1[1] = X[1] + h * k23;
			//X1[2] = X[2] + h * k33;
			//
			//k14 = (a[0] * X1[0] * (1 - X1[1]) - a[1] * X1[2]);
			//k24 = -a[2] * X1[1] * (1 - X1[0] * X1[0]);
			//k34 = a[3] * X1[0];
			//
			//X[0] = X[0] + h * (k11 + 2 * k12 + 2 * k13 + k14) / 6;
			//X[1] = X[1] + h * (k21 + 2 * k22 + 2 * k23 + k24) / 6;
			//X[2] = X[2] + h * (k31 + 2 * k32 + 2 * k33 + k34) / 6;



	////Generalized Sprott A
		//Explicit Euler
			//double X1[4];
			//X1[0] = X[0] + h * (X[1] + X[0] * X[2]);
			//X1[1] = X[1] + h * (-a[2] * X[0] + X[1] * X[2] + X[3]);
			//X1[2] = X[2] + h * (1 - X[0] * X[0] - X[1] * X[1]);
			//X1[3] = X[3] + h * (-a[1] * X[1]);
			//X[0] = X1[0]; X[1] = X1[1]; X[2] = X1[2]; X[3] = X1[3];
		//EC
			//X[0] = X[0] + h * (X[1] + X[0] * X[2]);
			//X[1] = X[1] + h * (-a[2] * X[0] + X[1] * X[2] + X[3]);
			//X[2] = X[2] + h * (1 - X[0] * X[0] - X[1] * X[1]);
			//X[3] = X[3] + h * (-a[1] * X[1]);
		//Implicit Euler


		//RK4	
			//double X1[4];
			//double k[4][4];
			//int N = 4;
			//int i, j;
			//for (i = 0; i < N; i++) {
			//	X1[i] = X[i];
			//}
			//for (j = 0; j < 4; j++) {
			//	k[0][j] = (X1[1] + X1[0] * X1[2]);
			//	k[1][j] = (-a[2] * X1[0] + X1[1] * X1[2] + X1[3]);
			//	k[2][j] = (1 - X1[0] * X1[0] - X1[1] * X1[1]);
			//	k[3][j] = (-a[1] * X1[1]);

			//	if (j == 3) {
			//		for (i = 0; i < N; i++) {
			//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
			//		}
			//	}
			//	else if (j == 2) {
			//		for (i = 0; i < N; i++) {
			//			X1[i] = X[i] + h * k[i][j];
			//		}
			//	}
			//	else {
			//		for (i = 0; i < N; i++) {
			//			X1[i] = X[i] + 0.5 * h * k[i][j];
			//		}
			//	}
			//}

	////////Dequan Li
		//Euler
			//double X0[3];
			//X0[0] = X[0] + h * (a[0] * (X[1] - X[0]) + a[2] * X[0] * X[2]);
			//X0[1] = X[1] + h * (a[4] * X[0] + a[5] * X[1] - X[0] * X[2]);
			//X0[2] = X[2] + h * (a[1] * X[2] + X[0] * X[1] - a[3] * X[0] * X[0]);
			//X[0] = X0[0]; X[1] = X0[1]; X[2] = X0[2];
		//Euler-Kromer
			//X[0] = X[0] + h * (a[0] * (X[1] - X[0]) + a[2] * X[0] * X[2]);
			//X[1] = X[1] + h * (a[4] * X[0] + a[5] * X[1] - X[0] * X[2]);
			//X[2] = X[2] + h * (a[1] * X[2] + X[0] * X[1] - a[3] * X[0] * X[0]);
		//Implicit Euler
			//double J[3][3], w[3][4], Z[3], A[3], Zn[3], dz, tol, t, d;

			//J[0][0] = -a[0] + a[2] * X[2];
			//J[0][1] = a[0];
			//J[0][2] = X[0] * a[2];

			//J[1][0] = a[4] - X[2];
			//J[1][1] = a[5];
			//J[1][2] = -X[0];

			//J[2][0] = X[1] - 2 * a[3] * X[0];
			//J[2][1] = X[0];
			//J[2][2] = a[1];


			//int k; int i; int j;
			//tol = 1e-14;
			//int nnewtmaX = 3, nnewt;
			//dz = 2e-13;
			//nnewt = 0;

			//Z[0] = X[0]; Z[1] = X[1]; Z[2] = X[2];
			//while ((dz > tol) && (nnewt < nnewtmaX)) {

			//	for (i = 0; i < 3; i++) {
			//		for (j = 0; j < 3; j++) {
			//			if (i == j)
			//				w[i][j] = 1 - 0.5 * h * J[i][j];
			//			else
			//				w[i][j] = -0.5 * h * J[i][j];
			//		}
			//	}

			//	A[0] = 0.5 * (X[0] + Z[0]);
			//	A[1] = 0.5 * (X[1] + Z[1]);
			//	A[2] = 0.5 * (X[2] + Z[2]);

			//	w[0][3] = -Z[0] + X[0] + h * (a[0] * (A[1] - A[0]) + a[2] * A[0] * A[2]);
			//	w[1][3] = -Z[1] + X[1] + h * (a[4] * A[0] + a[5] * A[1] - A[0] * A[2]);
			//	w[2][3] = -Z[2] + X[2] + h * (a[1] * A[2] + A[0] * A[1] - a[3] * A[0] * A[0]);


			//	for (k = 0; k <= 1; k++) {
			//		int l = k;

			//		for (i = k + 1; i <= 2; i++) {
			//			if (abs(w[i][k]) > abs(w[l][k])) {
			//				l = i;
			//			}
			//		}
			//		if (l != k) {
			//			for (j = 0; j <= 3; j++) {
			//				if ((j == 0) || (j >= k)) {
			//					t = w[k][j];
			//					w[k][j] = w[l][j];
			//					w[l][j] = t;
			//				}
			//			}
			//		}

			//		d = 1.0 / w[k][k];
			//		for (i = (k + 1); i <= (2); i++) {
			//			if (w[i][k] == 0) {
			//				continue;
			//			}
			//			t = w[i][k] * d;
			//			for (j = k; j <= (3); j++) {
			//				if (w[k][j] != 0) {
			//					w[i][j] = w[i][j] - t * w[k][j];
			//				}
			//			}
			//		}
			//	}
			//	for (i = (3); i >= 2; i--) {
			//		for (j = 1; j <= i - 1; j++) {
			//			t = w[i - j - 1][i - 1] / w[i - 1][i - 1];
			//			w[i - j - 1][3] = w[i - j - 1][3] - t * w[i - 1][3];
			//		}
			//		w[i - 1][3] = w[i - 1][3] / w[i - 1][i - 1];
			//	}
			//	w[0][3] = w[0][3] / w[0][0];
			//	Zn[0] = Z[0] + w[0][3];
			//	Zn[1] = Z[1] + w[1][3];
			//	Zn[2] = Z[2] + w[2][3];

			//	dz = sqrt(((Zn[0] - Z[0]) * (Zn[0] - Z[0]) + (Zn[1] - Z[1]) * (Zn[1] - Z[1]) + (Zn[2] - Z[2]) * (Zn[2] - Z[2])));
			//	Z[0] = Zn[0];
			//	Z[1] = Zn[1];
			//	Z[2] = Zn[2];

			//	nnewt++;
			//}

			//X[0] = Zn[0];
			//X[1] = Zn[1];
			//X[2] = Zn[2];



	//////
	//double h1 = h / 2;

	//X[0] = X[0] - h1 * (X[1] + X[2]);
	//X[1] = X[1] + h1 * (X[0] + X[1] * a[0]);
	//X[2] = X[2] + h1 * (a[1] + X[2] * (X[0] - a[2]));

	//X[2] = (X[2] + h1 * a[1]) / (1 - h1 * (X[0] - a[2]));
	//X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[0]);
	//X[0] = X[0] - h1 * (X[1] + X[2]);

	//////Matreshka CHUA
	/*double m, x0, x1, y, k, b, flag, T2, X1[3];
	m = 18;

	if (X[0] == 0)
		x0 = 0;
	else
		x0 = X[0]/abs(X[0]);

	flag = 1;
	while (flag == 1) {
		x1 = m * x0;
		if ((X[0] > 0) && (X[0] > x1) || (X[0] < 0) && (X[0] < x1)) {
			x0 = x0 * m;
		}
		else {
			if ((X[0] >= x0) && (X[0] <= x1) || (X[0] <= x0) && (X[0] >= x1)) {
				flag = 0;
				T2 = (m - 1) * x0 / 2;
				if ((X[0] >= 0 && X[0] < x0 + T2) || (X[0] < 0 && X[0] > x0 + T2)) {
					k = (-3 - m) / (m - 1);
					b = (1 - (-3 - m) / (m - 1)) * x0;
				}
				else {
					k = (3 * m + 1) / (m - 1);
					b = (1 - (3 * m + 1) / (m - 1)) * m * x0;
				}
				y = k * X[0] + b;
			}
			else {
				if ((X[0] > 0) && (X[0] < x0) || (X[0] < 0) && (X[0] > x0))
					x0 = x0 / m;
			}
		}
	}

	X[0] = X[0] + h * (a[0] * (X[1] + a[3] * y));
	X[1] = X[1] + h * (X[0] - X[1] + X[2]);
	X[2] = X[2] + h * (-a[1] * X[1] - a[2] * X[2]);
	X[0] = X1[0];
	X[1] = X1[1];
	X[2] = X1[2];*/

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	////Neron AND-TS ver 2.0
//	double X1[3], k[3][4], Im, Id, Iin;
//	int N = 3;
//	int i, j;
//
//	for (i = 0; i < N; i++) {
//		X1[i] = X[i];
//	}
//
//	for (j = 0; j < 4; j++) {
//
//
//	//	Id = a[4] * (expf((X[0] + a[6]) / a[9]) - expf(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * expf(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atanf(a[17] * (X[0] + a[6] - a[18])) + atanf(a[17] * (X[0] + a[6] + a[18])));
//	////  Id = a(4) * ( exp((X(1) + a(6)) / a(9)) -  exp(-(X(1) + a(6)) / a(9))) + (a[2] / a(10)) * (X(1) + a(6)) *  exp(-(X(1) + a(6) - a(10)) / a(10)) + a(3) * ( atan(a(17) * (X(1) + a(6) - a(18))) +  atan(a(17) * (X(1) + a(6) + a(18))));
//
//	//	if ((-X[0] + a[7]) > 0)
//	//		Im = (-X[0] + a[7]) * X[1] / a[19] + a[5];
//	//	else
//	//		Im = (-X[0] + a[7]) * X[1] / a[20] - a[5];
//
//	//	//Iin = a[25];
//	//	Iin = (a[25] + a[29] * floor((X[2] + a[28]) / a[26])) * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
//	//	//Iin = a[25] * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
//	//	//Iin = a[25] *  (fmod(X[2] + a[28], a[26])  )/a[26];		
//
//	//	k[0][j] = (Iin - Im - Id) / a[8];
//	//	k[1][j] = (1 / a[21]) * (1 / (1 + expf(-1 / (a[15] * a[15]) * ((-X[0] + a[7]) - a[11]) * ((-X[0] + a[7]) - a[13])))) * ((1 - 1 / (expf((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + expf(-1 / (a[16] * a[16]) * ((-X[0] + a[7]) - a[14]) * ((-X[0] + a[7]) - a[12])))) * ((1 - 1 / (expf((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]) + a[24]))));
//	// // //k(2, j) = (1 / a(21)) * (1 / (1 +  exp(-1 / (a(15) * a(15)) * ((-X(1) + a(7)) - a(11)) * ((-X(1) + a(7)) - a(13))))) * ((1 - 1 / ( exp((a[1] * X(2) + a(23))))) * (1 - X(2)) + X(2) * (1 - 1 / ( exp(a[1] * (1 - X(2)))))) - (1 / a(22)) * (1 - 1 / (1 +  exp(-1 / (a(16) * a(16)) * ((-X(1) + a(7)) - a(14)) * ((-X(1) + a(7)) - a(12))))) * ((1 - 1 / ( exp((a[1] * X(2))))) * (1 - X(2)) + X(2) * (1 - 1 / ( exp(a[1] * (1 - X(2)) + a(24)))));
//	//	k[2][j] = 1;
//
//		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//		Id = a[4] * (exp((X[0] + a[6]) / a[9]) - exp(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * exp(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atan(a[17] * (X[0] + a[6] - a[18])) + atan(a[17] * (X[0] + a[6] + a[18])));
//////    Id = a(4) * (exp((X(1) + a(6)) / a(9)) - exp(-(X(1) + a(6)) / a(9))) + (a[2] / a(10)) * (X(1) + a(6)) * exp(-(X(1) + a(6) - a(10)) / a(10)) + a(3) * (atan(a(17) * (X(1) + a(6) - a(18))) + atan(a(17) * (X(1) + a(6) + a(18))));
//
//		if ((-X[0] + a[7]) > 0)
//			Im = (-X[0] + a[7]) * X[1] / a[19] + a[5];
//		else
//			Im = (-X[0] + a[7]) * X[1] / a[20] - a[5];
//
//		//Iin   = a[25];
//		Iin = (a[25] + a[29] * floor((X[2] + a[28]) / a[26])) * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
//
//		//Iin = a[25] * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
//		//Iin = a[25] *  (fmod(X[2] + a[28], a[26])  )/a[26];		
//
//		k[0][j] = (Iin - Im - Id) / a[8];
//		k[1][j] = (1 / a[21]) * (1 / (1 + exp(-1 / (a[15] * a[15]) * ((-X[0] + a[7]) - a[11]) * ((-X[0] + a[7]) - a[13])))) * ((1 - 1 / (exp((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + exp(-1 / (a[16] * a[16]) * ((-X[0] + a[7]) - a[14]) * ((-X[0] + a[7]) - a[12])))) * ((1 - 1 / (exp((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]) + a[24]))));
//	  //k(2, j) = (1 / a(21)) * (1 / (1 + exp(-1 / (a(15) * a(15)) * ((-X(1) + a(7)) - a(11)) * ((-X(1) + a(7)) - a(13))))) * ((1 - 1 / (exp((a[1] * X(2) + a(23))))) * (1 - X(2)) + X(2) * (1 - 1 / (exp(a[1] * (1 - X(2)))))) - (1 / a(22)) * (1 - 1 / (1 + exp(-1 / (a(16) * a(16)) * ((-X(1) + a(7)) - a(14)) * ((-X(1) + a(7)) - a(12))))) * ((1 - 1 / (exp((a[1] * X(2))))) * (1 - X(2)) + X(2) * (1 - 1 / (exp(a[1] * (1 - X(2)) + a(24)))));
//		k[2][j] = 1;
//
//
//		if (j == 3) {
//			for (i = 0; i < N; i++) {
//				X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
//			}
//		}
//		else if (j == 2) {
//			for (i = 0; i < N; i++) {
//				X1[i] = X[i] + h * k[i][j];
//			}
//		}
//		else {
//			for (i = 0; i < N; i++) {
//				X1[i] = X[i] + 0.5 * h * k[i][j];
//			}
//		}
//	}
//	X[3] = Iin;
//
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 


//double X1[3], k[3][4], Im = 0, Id = 0, Iin = 0;
//double pi = 3.14159265359;
//int N = 3;
//int i, j;
//
//for (i = 0; i < N; i++) {
//	X1[i] = X[i];
//}
//
//for (j = 0; j < 4; j++) {
//
//
//	//Id = a[4] * (exp((X1[0] + a[6]) / a[9]) - exp(-(X1[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X1[0] + a[6]) * exp(-(X1[0] + a[6] - a[10]) / a[10]) + a[3] * (atan(a[17] * (X1[0] + a[6] - a[18])) + atan(a[17] * (X1[0] + a[6] + a[18])));
//	Id = a[4] * (expf((X1[0] + a[6]) / a[9]) - expf(-(X1[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X1[0] + a[6]) * expf(-(X1[0] + a[6] - a[10]) / a[10]) + a[3] * (atanf(a[17] * (X1[0] + a[6] - a[18])) + atanf(a[17] * (X1[0] + a[6] + a[18])));
//
//	if ((-X1[0] + a[7]) > 0)
//		Im = (-X1[0] + a[7]) * X1[1] / a[19] + a[5];
//	else
//		Im = (-X1[0] + a[7]) * X1[1] / a[20] - a[5];
//
//
//	Iin = a[28] + a[25] * sin(2 * pi * a[26] * X1[2] + a[27]);
//
//	k[0][j] = (Iin + Im - Id) / a[8];
//	//k[1][j] = (1 / a[21]) * (1 / (1 + exp(-1 / (a[15] * a[15]) * ((-X1[0] + a[7]) - a[11]) * ((-X1[0] + a[7]) - a[13])))) * ((1 - 1 / (exp((a[1] * X1[1] + a[23])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (exp(a[1] * (1 - X1[1]))))) - (1 / a[22]) * (1 - 1 / (1 + exp(-1 / (a[16] * a[16]) * ((-X1[0] + a[7]) - a[14]) * ((-X1[0] + a[7]) - a[12])))) * ((1 - 1 / (exp((a[1] * X1[1])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (exp(a[1] * (1 - X1[1]) + a[24]))));
//	k[1][j] = (1 / a[21]) * (1 / (1 + expf(-1 / (a[15] * a[15]) * ((-X1[0] + a[7]) - a[11]) * ((-X1[0] + a[7]) - a[13])))) * ((1 - 1 / (expf((a[1] * X1[1] + a[23])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (expf(a[1] * (1 - X1[1]))))) - (1 / a[22]) * (1 - 1 / (1 + expf(-1 / (a[16] * a[16]) * ((-X1[0] + a[7]) - a[14]) * ((-X1[0] + a[7]) - a[12])))) * ((1 - 1 / (expf((a[1] * X1[1])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (expf(a[1] * (1 - X1[1]) + a[24]))));
//	k[2][j] = 1;
//
//
//	if (j == 3) {
//		for (i = 0; i < N; i++) {
//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
//		}
//	}
//	else if (j == 2) {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + h * k[i][j];
//		}
//	}
//	else {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + 0.5 * h * k[i][j];
//		}
//	}
//}
//
//X[3] = Iin;

//Kopets ECG

//double aa[5], dx1, dx2, zna, cur, delQ, X1[4];
//double h1 = a[0] * h;
//double h2 = (1 - a[0]) * h;
//int i;
//double pi = 3.14159265358979323;
//
//double ag[5]{ 0.04319, 	-0.03,	 0.3955,	 -0.045, 0.055 };
//double bg[5]{ 	 -0.4, -0.045, 	 -0.009,	0.07096, 1.8   };
//double cg[5]{ 	0.063, 	0.032, 	0.02063, 	  0.016, 1.3 };
//
//
//
//for (i = 0; i < 5; i++)
//	aa[i] = 2 * ag[i] / (cg[i] * cg[i]);

////a[1] = 3.75;  //A  = 3.75;
////a[2] = 10;    //B  = 10;
////a[3] = 1;     //d  = 1;
////a[4] = -1;    //n  = - 1;
////a[5] = -0.33; //m0 = - 0.33;
////a[6] = 0.25;  //m1 = 0.25;
////a[7] = 5.5;   //T  = 5.5;
////a[8] = 10;    //k = 10;

//X[0] = X[0] + h1 * a[7] * (-a[1] * (X[2] + X[0] * (a[5] + a[6] * X[3] * X[3])));
//X[1] = X[1] + h1 * a[7] * (-a[2] * X[2]);
//X[2] = X[2] + h1 * a[7] * (-a[3] * (X[0] - X[1]));
//X[3] = X[3] + h1 * a[7] * (a[4] * X[0]);
//X[3] = X[3] + h2 * a[7] * (a[4] * X[0]);
//X[2] = X[2] + h2 * a[7] * (-a[3] * (X[0] - X[1]));
//X[1] = X[1] + h2 * a[7] * (-a[2] * X[2]);
//X[0] = (X[0] + h2 * a[7] * (-a[1] * (X[2]))) / (1 + h2 * a[7] * a[1] * (a[5] + a[6] * X[3] * X[3]));
//
//dx1 = a[7] * (-a[2] * X[2]);
//dx2 = a[7] * (-a[3] * (X[0] - X[1]));
//zna = (-X[2] * (dx1)+X[1] * dx2) / (X[1] * X[1] + X[2] * X[2]);
//
//cur = 0;
//for (i = 0; i < 5; i++) {
//	delQ = fmodf(atan2f(X[2], X[1]) - bg[i], 2 * pi);
//	cur = cur - aa[i] * delQ * expf(-((delQ * delQ) / (cg[i] * cg[i]))) * zna;
//}
//X[4] = X[4] + h * (cur - X[4] * a[8]);


//double X1[3], k[3][4], Im = 0, Id = 0, Iin = 0;
	//double pi = 3.14159265359;
	//int N = 3;
	//int i, j;
	//
	//for (i = 0; i < N; i++) {
	//	X1[i] = X[i];
	//}
	//
	//for (j = 0; j < 4; j++) {
	//
	//
	//	//Id = a[4] * (exp((X[0] + a[6]) / a[9]) - exp(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * exp(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atan(a[17] * (X[0] + a[6] - a[18])) + atan(a[17] * (X[0] + a[6] + a[18])));
	//
	//	//if ((-X[0] + a[7]) > 0)
	//	//	Im = (-X[0] + a[7]) * X[1] / a[19] + a[5];
	//	//else
	//	//	Im = (-X[0] + a[7]) * X[1] / a[20] - a[5];
	//
	//	////Iin   = a[25];
	//	//Iin = (a[25] + a[29] * floor((X[2] + a[28]) / a[26])) * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
	//	////Iin = a[25] *  (mod(X[2] + a[28], a[26])  )/a[26];	

	//	Iin = a[28] + a[25] * sin(2* pi * a[26] * X1[2] + a[27]);

	//	//k[0][j] = (Iin + Im - Id) / a[8];
	//	//k[1][j] = (1 / a[21]) * (1 / (1 + exp(-1 / (a[15] * a[15]) * ((-X[0] + a[7]) - a[11]) * ((-X[0] + a[7]) - a[13])))) * ((1 - 1 / (exp((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + exp(-1 / (a[16] * a[16]) * ((-X[0] + a[7]) - a[14]) * ((-X[0] + a[7]) - a[12])))) * ((1 - 1 / (exp((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]) + a[24]))));
	//	//k[2][j] = 1;
	//
	//	Id = a[4] * (expf((X1[0] + a[6]) / a[9]) - expf(-(X1[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X1[0] + a[6]) * expf(-(X1[0] + a[6] - a[10]) / a[10]) + a[3] * (atanf(a[17] * (X1[0] + a[6] - a[18])) + atanf(a[17] * (X1[0] + a[6] + a[18])));
	//	
	//	if ((-X1[0] + a[7]) > 0)
	//		Im = (-X1[0] + a[7]) * X1[1] / a[19] + a[5];
	//	else
	//		Im = (-X1[0] + a[7]) * X1[1] / a[20] - a[5];
	//
	//	//Iin   = a[25];
	//
	//	//if (fmod((X[2] + a[28]), a[26]) < a[27] && ((X[2] + a[28]) > 0))
	//	//	Iin = (a[25] + a[29] * floor(((X[2] + a[28])) / a[26]));
	//	//else
	//	//	Iin = 0;
	//		
	//
	//	//Iin = (a[25] + a[29] * floor((X[2] + a[28]) / a[26])) * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
	//	//Iin = a[25] *  (mod(X[2] + a[28], a[26])  )/a[26];		
	//
	//	k[0][j] = (Iin + Im - Id) / a[8];
	//	k[1][j] = (1 / a[21]) * (1 / (1 + expf(-1 / (a[15] * a[15]) * ((-X1[0] + a[7]) - a[11]) * ((-X1[0] + a[7]) - a[13])))) * ((1 - 1 / (expf((a[1] * X1[1] + a[23])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (expf(a[1] * (1 - X1[1]))))) - (1 / a[22]) * (1 - 1 / (1 + expf(-1 / (a[16] * a[16]) * ((-X1[0] + a[7]) - a[14]) * ((-X1[0] + a[7]) - a[12])))) * ((1 - 1 / (expf((a[1] * X1[1])))) * (1 - X1[1]) + X1[1] * (1 - 1 / (expf(a[1] * (1 - X1[1]) + a[24]))));
	//	k[2][j] = 1;
	//
	//	if (j == 3) {
	//		for (i = 0; i < N; i++) {
	//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
	//		}
	//	}
	//	else if (j == 2) {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + h * k[i][j];
	//		}
	//	}
	//	else {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + 0.5 * h * k[i][j];
	//		}
	//	}
	//}
	//
	//X[3] = Iin;






//Neron AND-TS ver 1.0
	//double X1[2], k[2][4], Im, Id;
	//int N = 2;
	//int i, j;

	//for (i = 0; i < N; i++) {
	//	X1[i] = X[i];
	//}

	//for (j = 0; j < 4; j++) {

	//	Im = (X1[0] + a[12]) * ((X1[1] / a[5]) + ((1 - X1[1]) / a[6]));
	//	Id = a[4] * (exp((X1[0] + a[11]) / a[16]) - exp(-(X1[0] + a[11]) / a[16])) + (a[2] / a[17]) * (X1[0] + a[11]) * exp(-(X1[0] + a[11] - a[17]) / a[17]) + a[3] * (atan(a[19] * (X1[0] + a[11] - a[20])) + atan(a[19] * (X1[0] + a[11] + a[20])));

	//	k[0][j] = (a[1] - Im - Id) / a[13];
	//	k[1][j] = (1 / a[14]) * ((1 - X1[1]) / (1 + exp(-(X1[0] + a[12] - a[7]) * (X1[0] + a[12] - a[9]) / (a[15] * a[18]))) - X1[1] * (1 - 1 / (1 + exp(-(X1[0] + a[12] - a[8]) * (X1[0] + a[12] - a[10]) / (a[15] * a[18])))));

	//	if (j == 3) {
	//		for (i = 0; i < N; i++) {
	//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
	//		}
	//	}
	//	else if (j == 2) {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + h * k[i][j];
	//		}
	//	}
	//	else {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + 0.5 * h * k[i][j];
	//		}
	//	}
	//}


	//////Neuron JJ-paper 25.02.2024
	//double F1, F2, Iin, h1, h2, a4, a5;

	//double a8, a9;
	//// **** F, eta_2 = f(eta_1)
	////a8 = 0.9 * a[7];
	////a9 = 0.5 * a[7];
	//// **** END ****
	////// *** *normal ***
	//a8 = a[8];
	//a9 = a[9];

	//a4 = 2 / (4 + a[6] * a[3]);
	//a5 = floor(a[5]);
	//h1 = h * a[0];
	//h2 = h * (1 - a[0]);

	//X[0] = X[0] + h1 * (X[3]);
	//X[1] = X[1] + h1 * (X[4]);
	//X[2] = X[2] + h1 * (X[5]);

	//Iin = a[10] * ((fmod(X[6] + a[13], a[11]) < a[12]) ? 1 : 0);
	//F1 = (a[1] + a[2] * a[3] * Iin);
	//F2 = (2 / a[6]) * (X[0] - X[1] - 2 * 3.14159265359 * (a5));

	//X[3] = X[3] + h1 * ((a4 / a[7]) * (F1 + F2 - a[3] * (X[0] + X[2])) - a9 * X[3] - sin(X[0]));
	////X[4] = X[4] + h1 * ((a4 / a[7]) * (F1 - F2 - a[3] * (X[1] + X[2])) - a[9] * X[4] - sin(X[1]));
	//X[4] = X[4] + h1 * ((a4 / a8) * (F1 - F2 - a[3] * (X[1] + X[2])) - a9 * X[4] - sin(X[1]));
	//X[5] = X[5] + h1 * (a4 * (2 * F1 - a[3] * (X[0] + X[1] + 2 * X[2])) - a9 * X[5] - sin(X[2]));
	//X[6] = X[6] + h1;


	//Iin = a[10] * ((fmod(X[6] + a[13], a[11]) < a[12]) ? 1 : 0);
	//F1 = (a[1] + a[2] * a[3] * Iin);

	//X[6] = X[6] + h2;
	//X[5] = (X[5] + h2 * (a4 * (2 * F1 - a[3] * (X[0] + X[1] + 2 * X[2])) - sin(X[2]))) / (1 + h2 * a9);
	//X[4] = (X[4] + h2 * ((a4 / a8) * (F1 - F2 - a[3] * (X[1] + X[2])) - sin(X[1]))) / (1 + h2 * a9);
	////X[4] = (X[4] + h2 * ((a4 / a[7]) * (F1 - F2 - a[3] * (X[1] + X[2])) - sin(X[1]))) / (1 + h2 * a[9]);
	//X[3] = (X[3] + h2 * ((a4 / a[7]) * (F1 + F2 - a[3] * (X[0] + X[2])) - sin(X[0]))) / (1 + h2 * a9);
	//X[2] = X[2] + h2 * (X[5]);
	//X[1] = X[1] + h2 * (X[4]);
	//X[0] = X[0] + h2 * (X[3]);
	//X[7] = X[5] + X[4];

	 //////Gokyildirim_2, IC = {1.5,3,6}, a = {0.5, 0.1, 0.7} - a[0] = symmetry	
	/////Rkalpha
	//double h1, alpha1, alpha2, X1[3], k[2][3];

	//h1 = a[0] * h;

	//alpha2 = 1 / (2 * a[0]);
	//alpha1 = (1 - alpha2);

	//k[0][0] = X[1] + X[0] * X[1] - X[1] * X[2];
	//k[0][1] = X[0] + a[1] * X[1] * X[2];
	//k[0][2] = a[2] * X[1] * X[1] + X[0] * X[1];

	//X1[0] = X[0] + h1 * k[0][0];
	//X1[1] = X[1] + h1 * k[0][1];
	//X1[2] = X[2] + h1 * k[0][2];

	//k[1][0] = X1[1] + X1[0] * X1[1] - X1[1] * X1[2];
	//k[1][1] = X1[0] + a[1] * X1[1] * X1[2];
	//k[1][2] = a[2] * X1[1] * X1[1] + X1[0] * X1[1];

	//X[0] = X[0] + h * (alpha1 * k[0][0] + alpha2 * k[1][0]);
	//X[1] = X[1] + h * (alpha1 * k[0][1] + alpha2 * k[1][1]);
	//X[2] = X[2] + h * (alpha1 * k[0][2] + alpha2 * k[1][2]);

	//////Rossler
	//double h1 = h * a[0];
	//double h2 = h * (1 - a[0]);
	//X[0] = X[0] + h1 * (-X[1] - X[2]);
	//X[1] = (X[1] + h1 * (X[0])) / (1 - a[1] * h1);
	//X[2] = (X[2] + h1 * a[2]) / (1 - h1 * (X[0] - a[3]));
	//X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));
	//X[1] = X[1] + h2 * (X[0] + a[1] * X[1]);
	//X[0] = X[0] + h2 * (-X[1] - X[2]);

	////Chen
	//double h1 = h * a[0];
	//double h2 = h * (1 - a[0]);

	//X[0] = (X[0] + h1 * a[1] * X[1]) / (1 + h1 * a[1]);
	//X[1] = (X[1] + h1 * X[0] * (a[3] - a[1] - X[2])) / (1 - h1 * a[3]);
	//X[2] = (X[2] + h1 * X[0] * X[1]) / (1 + h1 * a[2]);

	//X[2] = X[2] + h2 * (X[0] * X[1] - a[2] * X[2]);
	//X[1] = X[1] + h2 * (X[0] * (a[3] - a[1] - X[2]) + a[3] * X[1]);
	//X[0] = X[0] + h2 * (a[1] * (X[1] - X[0]));

	//////SBT Memristor
	//double X1[5];
	//double k[5][4];
	//int N = 5;
	//int i, j;

	//double A, B;
	//A = 0.0676;
	//B = 0.3682;

	//for (i = 0; i < N; i++) {
	//	X1[i] = X[i];
	//}

	//for (j = 0; j < 4; j++) {
	//	k[0][j] = (a[1] * ((-a[4] - (A + B * abs(X1[4]))) * X1[0] - (X1[0] - X1[1]) * ((A + B * abs(X1[3])) + a[5]) / (1 + a[6] * ((A + B * abs(X1[3])) + a[5]))));
	//	k[1][j] = (a[2] * (((A + B * abs(X1[3])) + a[5]) * (X1[0] - X1[1]) / (1 + a[6] * ((A + B * abs(X1[3])) + a[5])) + X1[2]));
	//	k[2][j] = (-a[3] * X1[1]);
	//	k[3][j] = ((X1[0] - X1[1]) / (1 + a[6] * ((A + B * abs(X1[3])) + a[5])));
	//	k[4][j] = (X1[0]);


	//	if (j == 3) {
	//		for (i = 0; i < N; i++) {
	//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
	//		}
	//	}
	//	else if (j == 2) {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + h * k[i][j];
	//		}
	//	}
	//	else {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + 0.5 * h * k[i][j];
	//		}
	//	}
	//}

	//////Henon Helies
	//double X1[4];
	//double k[4][4];
	//int N = 4;
	//int i, j;

	//for (i = 0; i < N; i++) {
	//	X1[i] = X[i];
	//}

	//for (j = 0; j < 4; j++) {
	//	k[0][j] = (X1[1]);
	//	k[1][j] = (-X1[0] - 2 * a[1] * X1[0] * X1[2]);
	//	k[2][j] = (X1[3]);
	//	k[3][j] = (-X1[2] - a[1] * (X1[0] * X1[0] - X1[2] * X1[2]));


	//	if (j == 3) {
	//		for (i = 0; i < N; i++) {
	//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
	//		}
	//	}
	//	else if (j == 2) {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + h * k[i][j];
	//		}
	//	}
	//	else {
	//		for (i = 0; i < N; i++) {
	//			X1[i] = X[i] + 0.5 * h * k[i][j];
	//		}
	//	}
	//}

	//////Sprott Case A, IC = {1,1,1}, a = {0.5, 1,1} - a[0] = symmetry
	//////CD
		//double h1, h2;
		//h1 = h * a[0];
		//h2 = h * (1 - a[0]);
		//
		//X[0] = X[0] + h1 * (X[1]);
		//X[1] = (X[1] - h1 * X[0]) / (1 - h1 * a[1] * X[2]);
		//X[2] = X[2] + h * (1 - a[2] * X[1] * X[1]);
		//X[1] = X[1] + h2 * (a[1] * X[1] * X[2] - X[0]);
		//X[0] = X[0] + h2 * (X[1]);

	//////Sine-CS, IC = {1,1,1E-6}, a = {0.5, 3.6,3.6} - a[0] = symmetry
	//////RK4
		//double X1[3];
		//double k[3][4];
		//int N = 3;
		//int i, j;
		//
		//for (i = 0; i < N; i++) {
		//	X1[i] = X[i];
		//}
		//
		//for (j = 0; j < 4; j++) {
		//	k[0][j] = X1[1] + X1[2] - a[1] * sin(X1[1]);
		//	k[1][j] = -X1[0] + X1[2];
		//	k[2][j] = -X1[0] - X1[2] + a[2] * sin(X1[0]);
		//
		//
		//	if (j == 3) {
		//		for (i = 0; i < N; i++) {
		//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
		//		}
		//	}
		//	else if (j == 2) {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + h * k[i][j];
		//		}
		//	}
		//	else {
		//		for (i = 0; i < N; i++) {
		//			X1[i] = X[i] + 0.5 * h * k[i][j];
		//		}
		//	}
		//}


	/*double k11, k21, k31, k12, k22, k32, k13, k23, k33, k14, k24, k34, k15, k25, k35, k16, k26, k36, k17, k27, k37, k18, k28, k38, k19, k29, k39;
	double k1A0, k2A0, k3A0, k1A1, k2A1, k3A1, k1A2, k2A2, k3A2, k1A3, k2A3, k3A3, x01, x02, x03;

	double B[2][13];
	B[0][0] = 0.04174749114153;
	B[0][1] = 0;
	B[0][2] = 0;
	B[0][3] = 0;
	B[0][4] = 0;
	B[0][5] = -0.05545232861124;
	B[0][6] = 0.2393128072012;
	B[0][7] = 0.7035106694034;
	B[0][8] = -0.7597596138145;
	B[0][9] = 0.6605630309223;
	B[0][10] = 0.1581874825101;
	B[0][11] = -0.2381095387529;
	B[0][12] = 0.25;

	B[1][0] = 0.02955321367635;
	B[1][1] = 0;
	B[1][2] = 0;
	B[1][3] = 0;
	B[1][4] = 0;
	B[1][5] = -0.8286062764878;
	B[1][6] = 0.3112409000511;
	B[1][7] = 2.4673451906;
	B[1][8] = -2.546941651842;
	B[1][9] = 1.443548583677;
	B[1][10] = 0.07941559588113;
	B[1][11] = 0.04444444444444;
	B[1][12] = 0;

	double M[13][12];
	M[0][0] = 0;
	M[0][1] = 0;
	M[0][2] = 0;
	M[0][3] = 0;
	M[0][4] = 0;
	M[0][5] = 0;
	M[0][6] = 0;
	M[0][7] = 0;
	M[0][8] = 0;
	M[0][9] = 0;
	M[0][10] = 0;
	M[0][11] = 0;

	M[1][0] = 0.05555555555556;
	M[1][1] = 0;
	M[1][2] = 0;
	M[1][3] = 0;
	M[1][4] = 0;
	M[1][5] = 0;
	M[1][6] = 0;
	M[1][7] = 0;
	M[1][8] = 0;
	M[1][9] = 0;
	M[1][10] = 0;
	M[1][11] = 0;

	M[2][0] = 0.02083333333333;
	M[2][1] = 0.0625;
	M[2][2] = 0;
	M[2][3] = 0;
	M[2][4] = 0;
	M[2][5] = 0;
	M[2][6] = 0;
	M[2][7] = 0;
	M[2][8] = 0;
	M[2][9] = 0;
	M[2][10] = 0;
	M[2][11] = 0;

	M[3][0] = 0.03125;
	M[3][1] = 0;
	M[3][2] = 0.09375;
	M[3][3] = 0;
	M[3][4] = 0;
	M[3][5] = 0;
	M[3][6] = 0;
	M[3][7] = 0;
	M[3][8] = 0;
	M[3][9] = 0;
	M[3][10] = 0;
	M[3][11] = 0;

	M[4][0] = 0.3125;
	M[4][1] = 0;
	M[4][2] = -1.171875;
	M[4][3] = 1.171875;
	M[4][4] = 0;
	M[4][5] = 0;
	M[4][6] = 0;
	M[4][7] = 0;
	M[4][8] = 0;
	M[4][9] = 0;
	M[4][10] = 0;
	M[4][11] = 0;

	M[5][0] = 0.0375;
	M[5][1] = 0;
	M[5][2] = 0;
	M[5][3] = 0.1875;
	M[5][4] = 0.15;
	M[5][5] = 0;
	M[5][6] = 0;
	M[5][7] = 0;
	M[5][8] = 0;
	M[5][9] = 0;
	M[5][10] = 0;
	M[5][11] = 0;

	M[6][0] = 0.04791013711111;
	M[6][1] = 0;
	M[6][2] = 0;
	M[6][3] = 0.1122487127778;
	M[6][4] = -0.02550567377778;
	M[6][5] = 0.01284682388889;
	M[6][6] = 0;
	M[6][7] = 0;
	M[6][8] = 0;
	M[6][9] = 0;
	M[6][10] = 0;
	M[6][11] = 0;

	M[7][0] = 0.01691798978729;
	M[7][1] = 0;
	M[7][2] = 0;
	M[7][3] = 0.387848278486;
	M[7][4] = 0.0359773698515;
	M[7][5] = 0.1969702142157;
	M[7][6] = -0.1727138523405;
	M[7][7] = 0;
	M[7][8] = 0;
	M[7][9] = 0;
	M[7][10] = 0;
	M[7][11] = 0;

	M[8][0] = 0.06909575335919;
	M[8][1] = 0;
	M[8][2] = 0;
	M[8][3] = -0.6342479767289;
	M[8][4] = -0.1611975752246;
	M[8][5] = 0.1386503094588;
	M[8][6] = 0.9409286140358;
	M[8][7] = 0.2116363264819;
	M[8][8] = 0;
	M[8][9] = 0;
	M[8][10] = 0;
	M[8][11] = 0;

	M[9][0] = 0.183556996839;
	M[9][1] = 0;
	M[9][2] = 0;
	M[9][3] = -2.468768084316;
	M[9][4] = -0.2912868878163;
	M[9][5] = -0.02647302023312;
	M[9][6] = 2.847838764193;
	M[9][7] = 0.2813873314699;
	M[9][8] = 0.1237448998633;
	M[9][9] = 0;
	M[9][10] = 0;
	M[9][11] = 0;

	M[10][0] = -1.215424817396;
	M[10][1] = 0;
	M[10][2] = 0;
	M[10][3] = 16.67260866595;
	M[10][4] = 0.9157418284168;
	M[10][5] = -6.056605804357;
	M[10][6] = -16.00357359416;
	M[10][7] = 14.8493030863;
	M[10][8] = -13.37157573529;
	M[10][9] = 5.13418264818;
	M[10][10] = 0;
	M[10][11] = 0;

	M[11][0] = 0.2588609164383;
	M[11][1] = 0;
	M[11][2] = 0;
	M[11][3] = -4.774485785489;
	M[11][4] = -0.435093013777;
	M[11][5] = -3.049483332072;
	M[11][6] = 5.577920039936;
	M[11][7] = 6.155831589861;
	M[11][8] = -5.062104586737;
	M[11][9] = 2.193926173181;
	M[11][10] = 0.1346279986593;
	M[11][11] = 0;

	M[12][0] = 0.8224275996265;
	M[12][1] = 0;
	M[12][2] = 0;
	M[12][3] = -11.65867325728;
	M[12][4] = -0.7576221166909;
	M[12][5] = 0.7139735881596;
	M[12][6] = 12.07577498689;
	M[12][7] = -2.12765911392;
	M[12][8] = 1.990166207049;
	M[12][9] = -0.234286471544;
	M[12][10] = 0.1758985777079;
	M[12][11] = 0;

	k11 = -x[1] - x[2];
	k21 = x[0] + a[0] * x[1];
	k31 = a[1] + x[2] * (x[0] - a[2]);


	x01 = x[0] + h * M[1][0] * k11;
	x02 = x[1] + h * M[1][0] * k21;
	x03 = x[2] + h * M[1][0] * k31;

	k12 = -x02 - x03;
	k22 = x01 + a[0] * x02;
	k32 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[2][0] * k11 + M[2][1] * k12);
	x02 = x[1] + h * (M[2][0] * k21 + M[2][1] * k22);
	x03 = x[2] + h * (M[2][0] * k31 + M[2][1] * k32);

	k13 = -x02 - x03;
	k23 = x01 + a[0] * x02;
	k33 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[3][0] * k11 + M[3][1] * k12 + M[3][2] * k13);
	x02 = x[1] + h * (M[3][0] * k21 + M[3][1] * k22 + M[3][2] * k23);
	x03 = x[2] + h * (M[3][0] * k31 + M[3][1] * k32 + M[3][2] * k33);

	k14 = -x02 - x03;
	k24 = x01 + a[0] * x02;
	k34 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[4][0] * k11 + M[4][1] * k12 + M[4][2] * k13 + M[4][3] * k14);
	x02 = x[1] + h * (M[4][0] * k21 + M[4][1] * k22 + M[4][2] * k23 + M[4][3] * k24);
	x03 = x[2] + h * (M[4][0] * k31 + M[4][1] * k32 + M[4][2] * k33 + M[4][3] * k34);

	k15 = -x02 - x03;
	k25 = x01 + a[0] * x02;
	k35 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[5][0] * k11 + M[5][1] * k12 + M[5][2] * k13 + M[5][3] * k14 + M[5][4] * k15);
	x02 = x[1] + h * (M[5][0] * k21 + M[5][1] * k22 + M[5][2] * k23 + M[5][3] * k24 + M[5][4] * k25);
	x03 = x[2] + h * (M[5][0] * k31 + M[5][1] * k32 + M[5][2] * k33 + M[5][3] * k34 + M[5][4] * k35);

	k16 = -x02 - x03;
	k26 = x01 + a[0] * x02;
	k36 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[6][0] * k11 + M[6][1] * k12 + M[6][2] * k13 + M[6][3] * k14 + M[6][4] * k15 + M[6][5] * k16);
	x02 = x[1] + h * (M[6][0] * k21 + M[6][1] * k22 + M[6][2] * k23 + M[6][3] * k24 + M[6][4] * k25 + M[6][5] * k26);
	x03 = x[2] + h * (M[6][0] * k31 + M[6][1] * k32 + M[6][2] * k33 + M[6][3] * k34 + M[6][4] * k35 + M[6][5] * k36);

	k17 = -x02 - x03;
	k27 = x01 + a[0] * x02;
	k37 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[7][0] * k11 + M[7][1] * k12 + M[7][2] * k13 + M[7][3] * k14 + M[7][4] * k15 + M[7][5] * k16 + M[7][6] * k17);
	x02 = x[1] + h * (M[7][0] * k21 + M[7][1] * k22 + M[7][2] * k23 + M[7][3] * k24 + M[7][4] * k25 + M[7][5] * k26 + M[7][6] * k27);
	x03 = x[2] + h * (M[7][0] * k31 + M[7][1] * k32 + M[7][2] * k33 + M[7][3] * k34 + M[7][4] * k35 + M[7][5] * k36 + M[7][6] * k37);

	k18 = -x02 - x03;
	k28 = x01 + a[0] * x02;
	k38 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[8][0] * k11 + M[8][1] * k12 + M[8][2] * k13 + M[8][3] * k14 + M[8][4] * k15 + M[8][5] * k16 + M[8][6] * k17 + M[8][7] * k18);
	x02 = x[1] + h * (M[8][0] * k21 + M[8][1] * k22 + M[8][2] * k23 + M[8][3] * k24 + M[8][4] * k25 + M[8][5] * k26 + M[8][6] * k27 + M[8][7] * k28);
	x03 = x[2] + h * (M[8][0] * k31 + M[8][1] * k32 + M[8][2] * k33 + M[8][3] * k34 + M[8][4] * k35 + M[8][5] * k36 + M[8][6] * k37 + M[8][7] * k38);

	k19 = -x02 - x03;
	k29 = x01 + a[0] * x02;
	k39 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[9][0] * k11 + M[9][1] * k12 + M[9][2] * k13 + M[9][3] * k14 + M[9][4] * k15 + M[9][5] * k16 + M[9][6] * k17 + M[9][7] * k18 + M[9][8] * k19);
	x02 = x[1] + h * (M[9][0] * k21 + M[9][1] * k22 + M[9][2] * k23 + M[9][3] * k24 + M[9][4] * k25 + M[9][5] * k26 + M[9][6] * k27 + M[9][7] * k28 + M[9][8] * k29);
	x03 = x[2] + h * (M[9][0] * k31 + M[9][1] * k32 + M[9][2] * k33 + M[9][3] * k34 + M[9][4] * k35 + M[9][5] * k36 + M[9][6] * k37 + M[9][7] * k38 + M[9][8] * k39);

	k1A0 = -x02 - x03;
	k2A0 = x01 + a[0] * x02;
	k3A0 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[10][0] * k11 + M[10][1] * k12 + M[10][2] * k13 + M[10][3] * k14 + M[10][4] * k15 + M[10][5] * k16 + M[10][6] * k17 + M[10][7] * k18 + M[10][8] * k19 + M[10][9] * k1A0);
	x02 = x[1] + h * (M[10][0] * k21 + M[10][1] * k22 + M[10][2] * k23 + M[10][3] * k24 + M[10][4] * k25 + M[10][5] * k26 + M[10][6] * k27 + M[10][7] * k28 + M[10][8] * k29 + M[10][9] * k2A0);
	x03 = x[2] + h * (M[10][0] * k31 + M[10][1] * k32 + M[10][2] * k33 + M[10][3] * k34 + M[10][4] * k35 + M[10][5] * k36 + M[10][6] * k37 + M[10][7] * k38 + M[10][8] * k39 + M[10][9] * k3A0);

	k1A1 = -x02 - x03;
	k2A1 = x01 + a[0] * x02;
	k3A1 = a[1] + x03 * (x01 - a[2]);


	x01 = x[0] + h * (M[11][0] * k11 + M[11][1] * k12 + M[11][2] * k13 + M[11][3] * k14 + M[11][4] * k15 + M[11][5] * k16 + M[11][6] * k17 + M[11][7] * k18 + M[11][8] * k19 + M[11][9] * k1A0 + M[11][10] * k1A1);
	x02 = x[1] + h * (M[11][0] * k21 + M[11][1] * k22 + M[11][2] * k23 + M[11][3] * k24 + M[11][4] * k25 + M[11][5] * k26 + M[11][6] * k27 + M[11][7] * k28 + M[11][8] * k29 + M[11][9] * k2A0 + M[11][10] * k2A1);
	x03 = x[2] + h * (M[11][0] * k31 + M[11][1] * k32 + M[11][2] * k33 + M[11][3] * k34 + M[11][4] * k35 + M[11][5] * k36 + M[11][6] * k37 + M[11][7] * k38 + M[11][8] * k39 + M[11][9] * k3A0 + M[11][10] * k3A1);

	k1A2 = -x02 - x03;
	k2A2 = x01 + a[0] * x02;
	k3A2 = a[1] + x03 * (x01 - a[2]);


	x01 = x[0] + h * (M[12][0] * k11 + M[12][1] * k12 + M[12][2] * k13 + M[12][3] * k14 + M[12][4] * k15 + M[12][5] * k16 + M[12][6] * k17 + M[12][7] * k18 + M[12][8] * k19 + M[12][9] * k1A0 + M[12][10] * k1A1 + M[12][11] * k1A2);
	x02 = x[1] + h * (M[12][0] * k21 + M[12][1] * k22 + M[12][2] * k23 + M[12][3] * k24 + M[12][4] * k25 + M[12][5] * k26 + M[12][6] * k27 + M[12][7] * k28 + M[12][8] * k29 + M[12][9] * k2A0 + M[12][10] * k2A1 + M[12][11] * k2A2);
	x03 = x[2] + h * (M[12][0] * k31 + M[12][1] * k32 + M[12][2] * k33 + M[12][3] * k34 + M[12][4] * k35 + M[12][5] * k36 + M[12][6] * k37 + M[12][7] * k38 + M[12][8] * k39 + M[12][9] * k3A0 + M[12][10] * k3A1 + M[12][11] * k3A2);

	k1A3 = -x02 - x03;
	k2A3 = x01 + a[0] * x02;
	k3A3 = a[1] + x03 * (x01 - a[2]);


	x[0] = x[0] + h * (B[0][0] * k11 + B[0][1] * k12 + B[0][2] * k13 + B[0][3] * k14 + B[0][4] * k15 + B[0][5] * k16 + B[0][6] * k17 + B[0][7] * k18 + B[0][8] * k19 + B[0][9] * k1A0 + B[0][10] * k1A1 + B[0][11] * k1A2 + B[0][12] * k1A3);
	x[1] = x[1] + h * (B[0][0] * k21 + B[0][1] * k22 + B[0][2] * k23 + B[0][3] * k24 + B[0][4] * k25 + B[0][5] * k26 + B[0][6] * k27 + B[0][7] * k28 + B[0][8] * k29 + B[0][9] * k2A0 + B[0][10] * k2A1 + B[0][11] * k2A2 + B[0][12] * k2A3);
	x[2] = x[2] + h * (B[0][0] * k31 + B[0][1] * k32 + B[0][2] * k33 + B[0][3] * k34 + B[0][4] * k35 + B[0][5] * k36 + B[0][6] * k37 + B[0][7] * k38 + B[0][8] * k39 + B[0][9] * k3A0 + B[0][10] * k3A1 + B[0][11] * k3A2 + B[0][12] * k3A3);*/

	//z[0] = x[0] + h * (B[1][0] * k11 + B[1][1] * k12 + B[1][2] * k13 + B[1][3] * k14 + B[1][4] * k15 + B[1][5] * k16 + B[1][6] * k17 + B[1][7] * k18 + B[1][8] * k19 + B[1][9] * k1A0 + B[1][10] * k1A1 + B[1][11] * k1A2 + B[1][12] * k1A3);
	//z[1] = x[1] + h * (B[1][0] * k21 + B[1][1] * k22 + B[1][2] * k23 + B[1][3] * k24 + B[1][4] * k25 + B[1][5] * k26 + B[1][6] * k27 + B[1][7] * k28 + B[1][8] * k29 + B[1][9] * k2A0 + B[1][10] * k2A1 + B[1][11] * k2A2 + B[1][12] * k2A3);
	//z[2] = x[2] + h * (B[1][0] * k31 + B[1][1] * k32 + B[1][2] * k33 + B[1][3] * k34 + B[1][4] * k35 + B[1][5] * k36 + B[1][6] * k37 + B[1][7] * k38 + B[1][8] * k39 + B[1][9] * k3A0 + B[1][10] * k3A1 + B[1][11] * k3A2 + B[1][12] * k3A3);

	//int i;
	//double h1;
	//double b[17];
	//b[0] = 0.1302024830889;
	//b[1] = 0.5611629817751;
	//b[2] = -0.3894749626448;
	//b[3] = 0.1588419065552;
	//b[4] = -0.3959038941332;
	//b[5] = 0.1845396409783;
	//b[6] = 0.2583743876863;
	//b[7] = 0.2950117236093;
	//b[8] = -0.60550853383;
	//b[9] = 0.2950117236093;
	//b[10] = 0.2583743876863;
	//b[11] = 0.1845396409783;
	//b[12] = -0.3959038941332;
	//b[13] = 0.1588419065552;
	//b[14] = -0.3894749626448;
	//b[15] = 0.5611629817751;
	//b[16] = 0.1302024830889;

	//for (i = 0; i < 17; ++i)
	//{
	//	h1 = h * 0.5 * b[i];
	//	x[0] = x[0] + h1 * (-x[1] - x[2]);
	//	x[1] = (x[1] + h1 * x[0]) / (1 - a[0] * h1);
	//	x[2] = (x[2] + h1 * a[1]) / (1 - h1 * (x[0] - a[2]));
	//	x[2] = x[2] + h1 * (a[1] + x[2] * (x[0] - a[2]));
	//	x[1] = x[1] + h1 * (x[0] + a[0] * x[1]);
	//	x[0] = x[0] + h1 * (-x[1] - x[2]);
	//}


//double X1[3], k[3][4], Im, Idiode, Iin, Itunnel, Iex, Id, V;
//double pi = 3.14159265359;

//Gi305a
//double Is = 3e-6;
//double B = 19.5;
//double Ip = 0.01;
//double Vp = 0.04;
//double Iv = 0.0003;
//double D = 20;
//double E = 0.09;
//double Cp = 1e-15;

//Gi401a
//double Is = 1.15e-7;
//double B = 15;
//double Ip = 2.1e-5;
//double Vp = 0.09;
//double Iv = -3e-6;
//double D = 26;
//double E = 0.14;
//double Cp = 1e-15;
//
//int N = 3;
//int i, j;
//
//for (i = 0; i < N; i++) {
//	X1[i] = X[i];
//}
//
//for (j = 0; j < 4; j++) {
//
//
//	V = X1[0] + a[7];
//	Idiode = Is * (exp(B * V) - exp(-B * V));
//	Itunnel = Ip / Vp * V * exp(-(V - Vp) / Vp);
//	Iex = Iv * (atan(D * (V - E)) + atan(D * (V + E)));
//
//	Id = Idiode + Itunnel + Iex;
//
//	Iin = a[5] * sin(2 * pi * a[6] * X1[2]);
//
//	//k[0][j] = (1/a[1])*(X1[1] -Iin - Id);
//	//k[1][j] = (1/a[2])*(-X1[0]-a[3]*X1[1]+a[4]);
//	//k[2][j] = 1;
//	k[0][j] = (1 / a[1]) * (-X1[1] + Iin - Id);
//	k[1][j] = (1 / a[2]) * (X1[0] - a[3] * X1[1] + a[4]);
//	k[2][j] = 1;
//
//
//	if (j == 3) {
//		for (i = 0; i < N; i++) {
//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
//		}
//	}
//	else if (j == 2) {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + h * k[i][j];
//		}
//	}
//	else {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + 0.5 * h * k[i][j];
//		}
//	}
//}


//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//
//X[0] = X[0] + h1 * (X[1] + X[0] * X[2]);
//X[1] = X[1] + h1 * (-X[0] * (a[2] - X[3]) + X[1] * X[2]);
//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[3] = X[3] + h1 * (-a[1] * X[1] * X[3]);
//X[3] = (X[3]) / (1 + h2 * a[1] * X[1]);
//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[1] = (X[1] + h2 * (-X[0] * (a[2] - X[3]))) / (1 - h2 * X[2]);
//X[0] = (X[0] + h2 * (X[1])) / (1 - h2 * X[2]);

//double X1[3], h1, h2;
//
//X1[0] = X[0] + h * (X[1]);
//X1[1] = X[1] + h * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
//X1[2] = X[2] + h * ((1 / a[1]) * (X[1] - X[2]));
//
//X[0] = X1[0];
//X[1] = X1[1];
//X[2] = X1[2];

/////Makarov
//double X1[3];
//X1[0] = X[0] + 0.5 * h * X[1];
//X1[1] = X[1] + 0.5 * h * (-X[0] - X[1] * X[2]);
//X1[2] = X[2] + 0.5 * h * (cosh(X[1]) - 1 - a[2] * cos(X[0] * X[0]) - a[1] * cos(X[1]));
//
//X[0] = X[0] + h * X1[1];
//X[1] = X[1] + h * (-X1[0] - X1[1] * X1[2]);
//X[2] = X[2] + h * (cosh(X1[1]) - 1 - a[2] * cos(X1[0] * X1[0]) - a[1] * cos(X1[1]));

//////Babkin_1
//double h1, h2;
//h1 = h * a[0];
//h2 = h * (1 - a[0]);
//
//X[0] = X[0] + h1 * (a[1] * X[1] * X[3]);
//X[1] = X[1] + h1 * (a[6] * X[1] * X[2]);
//X[2] = X[2] + h1 * (-a[2] * X[1] * X[1] + a[3] * X[3]);
//X[3] = X[3] + h1 * (-a[4] * X[0] * X[1] - a[5] * X[2] * X[2] * X[2]);
//X[3] = X[3] + h2 * (-a[4] * X[0] * X[1] - a[5] * X[2] * X[2] * X[2]);
//X[2] = X[2] + h2 * (-a[2] * X[1] * X[1] + a[3] * X[3]);
//X[1] = (X[1]) / (1 - h2 * a[6] * X[2]);
//X[0] = X[0] + h2 * (a[1] * X[1] * X[3]);


//double h1 = h * a[0];
//double h2 = h * (1 - a[0]);
//
//X[0] = X[0] + h1 * (a[3] * X[1] + X[0] * X[2]);
//X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3] * X[3]);
//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[3] = X[3] + h1 * (-a[1] * X[1]);
//
//X[3] = X[3] + h2 * (-a[1] * X[1]);
//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[1] = (X[1] + h2 * (X[3] * X[3] - a[2] * X[0])) / (1 - h2 * X[2]);
//X[0] = (X[0] + h2 * a[3] * X[1]) / (1 - h2 * X[2]);

//X[0] = X[0] + h1 * (a[3] * X[2] + X[0] * X[2]);
//X[1] = X[1] + h1 * (-a[2] * X[0] + X[1] * X[2] + X[3] * X[3]);
//X[2] = X[2] + h1 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[3] = X[3] + h1 * (-a[1] * X[1]);
//
//X[3] = X[3] + h2 * (-a[1] * X[1]);
//X[2] = X[2] + h2 * (1 - X[0] * X[0] - X[1] * X[1]);
//X[1] = (X[1] + h2 * (X[3] * X[3] - a[2] * X[0])) / (1 - h2 * X[2]);
//X[0] = (X[0] + h2 * a[3] * X[2]) / (1 - h2 * X[2]);


//double X1[4];
//X1[0] = X[0] + 0.5 * h * (a[3] * X[1] + X[0] * X[2]);
//X1[1] = X[1] + 0.5 * h * (-a[2] * X[0] + X[1] * X[2] + X[3] * X[3]);
//X1[2] = X[2] + 0.5 * h * (1 - X[0] * X[0] - X[1] * X[1]);
//X1[3] = X[3] + 0.5 * h * (-a[1] * X[1]);
//
//X[0] = X[0] + h * (a[3] * X1[1] + X1[0] * X1[2]);
//X[1] = X[1] + h * (-a[2] * X1[0] + X1[1] * X1[2] + X1[3] * X1[3]);
//X[2] = X[2] + h * (1 - X1[0] * X1[0] - X1[1] * X1[1]);
//X[3] = X[3] + h * (-a[1] * X1[1]);


// RLCs-JJ
//h1 = h * a[0];
//h2 = h * (1 - a[0]);
//
//X[0] = X[0] + h1 * (X[1]);
//X[1] = X[1] + h1 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
//X[2] = X[2] + h1 * ((1 / a[1]) * (X[1] - X[2]));
//
//X1[0] = X[0];
//X1[1] = X[1];
//X1[2] = X[2];
//
//X[2] = (X1[2] + h2 * (1 / a[1]) * X[1]) / (1 + h2 * (1 / a[1]));
//X[1] = X1[1] + h2 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
//X[1] = X1[1] + h2 * ((1 / a[2]) * (a[3] - ((X[1] > a[4]) ? a[5] : a[6]) * X[1] - sin(X[0]) - X[2]));
//X[0] = X1[0] + h2 * (X[1]);


////Timur Chinease
//double h1, h2;
//h1 = h * a[0];
//h2 = h * (1 - a[0]);
//
//X[0] = X[0] + h1 * (a[1] * X[1]);
//X[1] = X[1] + h1 * (a[2] * X[0] + a[3] * X[1] + a[4] * cos(a[5] * X[2]));
//X[2] = X[2] + h1 * (a[6] * X[1] + a[7] * X[2]);
//X[2] = ( X[2] + h2 * (a[6] * X[1]) ) / (1 - h2 * a[7]);
//X[1] = ( X[1] + h2 * (a[2] * X[0] + a[4] * cos(a[5] * X[2])) ) / (1 - a[3] * h2);
//X[0] = X[0] + h2 * (a[1] * X[1]);



//double X1[3];
//double k[3][4];
//
//int N = 3;
//int i, j;
//
//
//for (i = 0; i < N; i++) {
//	X1[i] = X[i];
//}
//
//for (j = 0; j < 4; j++) {
//
//	k[0][j] = a[1] * X1[1];
//	k[1][j] = a[2] * X1[2];
//	k[2][j] = a[3] * X1[1] + a[4] * X1[0] * X1[2] + a[5] * X1[1] * X1[1] + a[6] * X1[2] * X1[2] + a[7] * X1[0] * X1[0] * X1[2] + a[8] * X1[0] * X1[2] * X1[2] + a[9] * X1[1] * X1[2] * X1[2];
//
//
//
//	if (j == 3) {
//		for (i = 0; i < N; i++) {
//			X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
//		}
//	}
//	else if (j == 2) {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + h * k[i][j];
//		}
//	}
//	else {
//		for (i = 0; i < N; i++) {
//			X1[i] = X[i] + 0.5 * h * k[i][j];
//		}
//	}
//}

//double h1, h2, z;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (a[1] * X[1]);
//X[1] = X[1] + h1 * (a[2] * X[2]);
//X[2] = X[2] + h1 * (a[3] * X[1] + a[4] * X[0] * X[2] + a[5] * X[1] * X[1] + a[6] * X[2] * X[2] + a[7] * X[0] * X[0] * X[2] + a[8] * X[0] * X[2] * X[2] + a[9] * X[1] * X[2] * X[2]);
//z = X[2];
//X[2] = z + h2 * (a[3] * X[1] + a[4] * X[0] * X[2] + a[5] * X[1] * X[1] + a[6] * X[2] * X[2] + a[7] * X[0] * X[0] * X[2] + a[8] * X[0] * X[2] * X[2] + a[9] * X[1] * X[2] * X[2]);
//X[2] = z + h2 * (a[3] * X[1] + a[4] * X[0] * X[2] + a[5] * X[1] * X[1] + a[6] * X[2] * X[2] + a[7] * X[0] * X[0] * X[2] + a[8] * X[0] * X[2] * X[2] + a[9] * X[1] * X[2] * X[2]);
//X[1] = X[1] + h2 * (a[2] * X[2]);
//X[0] = X[0] + h2 * (a[1] * X[1]);


//// --- Case B ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (X[1] * X[2]);
//X[1] = X[1] + h1 * (X[0] - X[1]);
//X[2] = X[2] + h * (1 - X[0] * X[1]);
//X[1] = (X[1] + h2 * (X[0])) / (1 + h2);
//X[0] = X[0] + h2 * (X[1] * X[2]);

//// --- Case C ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (X[1] * X[2]);
//X[1] = X[1] + h1 * (X[0] - X[1]);
//X[2] = X[2] + h * (1 - X[0] * X[0]);
//X[1] = (X[1] + h2 * (X[0])) / (1 + h2);
//X[0] = X[0] + h2 * (X[1] * X[2]);
//
//// --- Case F ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (X[1] + X[2]);
//X[1] = X[1] + h1 * (-X[0] + 0.5 * X[1]);
//X[2] = X[2] + h1 * (X[0] * X[0] - X[2]);
//X[2] = (X[2] + h2 * (X[0] * X[0])) / (1 + h2);
//X[1] = (X[1] + h2 * (-X[0])) / (1 - 0.5 * h2);
//X[0] = X[0] + h2 * (X[1] + X[2]);

//// --- Case G ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (0.4 * X[0] + X[2]);
//X[1] = X[1] + h1 * (X[0] * X[2] - X[1]);
//X[2] = X[2] + h * (-X[0] + X[1]);
//X[1] = (X[1] + h2 * (X[0] * X[2])) / (1 + h2);
//X[0] = (X[0] + h2 * (X[2])) / (1 - 0.4 * h2);

//// --- Case H ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (-X[1] + X[2] * X[2]);
//X[1] = X[1] + h1 * (X[0] + 0.5 * X[1]);
//X[2] = X[2] + h1 * (X[0] - X[2]);
//X[2] = (X[2] + h2 * (X[0])) / (1 + h2);
//X[1] = (X[1] + h2 * (X[0])) / (1 - 0.5 * h2);
//X[0] = X[0] + h2 * (-X[1] + X[2] * X[2]);

//// --- Case N ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * (-2 * X[1]);
//X[1] = X[1] + h1 * (X[0] + X[2] * X[2]);
//X[2] = X[2] + h1 * (1 + X[1] - 2 * X[2]);
//X[2] = (X[2] + h2 * (1 + X[1])) / (1 + 2 * h2);
//X[1] = X[1] + h2 * (X[0] + X[2] * X[2]);
//X[0] = X[0] + h2 * (-2 * X[1]);


// --- Bo Sang New Chaotic system ---
//double h1, h2;
//
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//
//X[0] = X[0] + h1 * (a[1] * X[1]);
//X[1] = X[1] + h1 * (a[2] * X[2] + a[3] * X[0]);
//X[2] = X[2] + h1 * (a[4] + a[5] * X[2] + a[6] * cos(a[7] * X[1]));
//X[2] = (X[2] + h2 * (a[4] + a[6] * cos(a[7] * X[1]))) / (1 - a[5] * h2);
//X[1] = X[1] + h2 * (a[2] * X[2] + a[3] * X[0]);
//X[0] = X[0] + h2 * (a[1] * X[1]);

//// --- Lorenz CD ---
//double h1, h2;
//h1 = h * a[0];
//h2 = h * (1 - a[0]);
//X[0] = X[0] + h1 * (a[1] * (X[1] - X[0]));
//X[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - X[1]);
//X[2] = X[2] + h1 * (X[0] * X[1] - a[3] * X[2]);
//X[2] = (X[2] + h2 * (X[0] * X[1])
//) / (1 + h2 * a[3]);
//X[1] = (X[1] + h2 * (X[0] * (a[2] - X[2]))) / (1 + h2);
//X[0] = (X[0] + h2 * (a[1] * (X[1]))) / (1 + a[1] * h2);

//// --- DissipCons CD ---
//double h1, h2;
//h1 = a[0] * h;
//h2 = (1 - a[0]) * h;
//X[0] = X[0] + h1 * ((X[2] * X[2] - a[3]) * X[0] - a[5] * X[1]);
//X[1] = X[1] + h1 * (a[5] * X[0] + (X[2] * X[2] - a[3]) * X[1]);
//X[2] = X[2] + h1 * ((a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]) * X[2] + a[4] * X[0]);
//
//X[2] = (X[2] + h2 * (a[4] * X[0])) / (1 - h2 * (a[1] + a[2] - X[0] * X[0] - a[2] * X[1] * X[1]));
//X[1] = (X[1] + h2 * (a[5] * X[0])) / (1 - h2 * (X[2] * X[2] - a[3]));
//X[0] = (X[0] + h2 * (-a[5] * X[1])) / (1 - h2 * (X[2] * X[2] - a[3]));

} // old_library