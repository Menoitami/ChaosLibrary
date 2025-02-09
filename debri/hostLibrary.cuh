#pragma once

// -----------------------
// --- ���������� CUDA ---
// -----------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -----------------------

// --------------------------------------------
// --- KiShiVi ���������� ��� ������ � CUDA ---
// --------------------------------------------

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"

// --------------------------------------------

// -----------------------------
// --- ���������� ���������� ---
// -----------------------------

#include <iomanip>
#include <string>

// -----------------------------



/**
 * �������, ��� ������� ���������� �������������� ���������.
 */
__host__ void distributedSystemSimulation(
	const double	tMax,							// ����� ������������� �������
	const double	h,								// ��� ��������������
	const double	hSpecial,						// ��� �������� ����� ��������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);				// ���������� ����������				



/**
 * �������, ��� ������� ���������� �������������� ���������.
 */
__host__ void bifurcation1D(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// �������� ��������� ����������
	const int*		indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH);					// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)


__host__ void TimeDomainCalculation(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,				// ������ � ���������� ���������
	const double* ranges,							// ��������� ��������� ����������
	const int* indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH);						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ���������� �������������� ��������� �� ����.
 */
__host__ void bifurcation1DForH(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// �������� ��������� ����
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH);					// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ���������� �������������� ���������. (�� ��������� ��������)
 */
__host__ void bifurcation1DIC(
	const double	tMax,							  // ����� ������������� �������
	const int		nPts,							  // ���������� ���������
	const double	h,								  // ��� ��������������
	const int		amountOfInitialConditions,		  // ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				  // ������ � ���������� ���������
	const double*	ranges,							  // �������� ��������� ���������� �������
	const int*		indicesOfMutVars,				  // ������ ����������� ���������� �������
	const int		writableVar,					  // ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						  // ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					  // �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							  // ���������
	const int		amountOfValues,					  // ���������� ����������
	const int		preScaller,
	std::string		OUT_FILE_PATH);					  // ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN)
 */
__host__ void bifurcation2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH);								// ������� ��� ��������� DBSCAN 

__host__ void neuronClasterization2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH);								// ������� ��� ��������� DBSCAN 

__host__ void neuronClasterization2D_2(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH);								// ������� ��� ��������� DBSCAN 

__host__ void FastSynchro(
	const double	tMax,								// ����� ������������� �������
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double	NTime,								// ����� ������� �� �������� ����� ����������� �������������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const double	h,									// ��� ��������������
	const double*	kForward,							// ������ ������������� ������������� ������
	const double*	kBackward,							// ������ ������������� ������������� �����
	const double*	initialConditionsMaster,			// ������ � ���������� ��������� �������
	const double*   initialConditionsSlave,				// ������ � ���������� ��������� ������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������
	const int		iterOfSynchr,						// ����� �������� �������������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	std::string		OUT_FILE_PATH);						// ������� ��� ��������� DBSCAN 

__host__ void FastSynchro_2(
	const double	NTime,								// ����� ������� �� �������� ����� ����������� �������������
	const int		nPts,							// ���������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const double	h,									// ��� ��������������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const double*	kForward,							// ������ ������������� ������������� ������
	const double*	kBackward,							// ������ ������������� ������������� �����
	const double*	initialConditions,			// ������ � ���������� ��������� �������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������
	const int		iterOfSynchr,						// ����� �������� �������������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	std::string		OUT_FILE_PATH);						// ������� ��� ��������� DBSCAN 


/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN) (for initial conditions)
 */
__host__ void bifurcation2DIC(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH);								// ������� ��� ��������� DBSCAN 



/**
 * ���������� 1D LLE ���������
 */
__host__ void LLE1D(
	const double	tMax,								// ����� ������������� �������
	const double	NT,									// ����� ������������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const double	eps,								// ������� ��� LLE
	const double*	initialConditions,					// ������ � ���������� ���������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);					// ���������� ����������



/**
 * ���������� 1D LLE ��������� (IC)
 */
__host__ void LLE1DIC(			
	const double tMax,									// ����� ������������� �������
	const double NT,									// ����� ������������
	const int nPts,										// ���������� ���������
	const double h,										// ��� ��������������
	const double eps,									// ������� ��� LLE
	const double* initialConditions,					// ������ � ���������� ���������
	const int amountOfInitialConditions,				// ���������� ��������� ������� ( ��������� � ������� )
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,						// ������� ���������� ����������
	const int writableVar,								// ������ ���������, �� �������� ����� ������� ���������
	const double maxValue,								// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double transientTime,							// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int amountOfValues,
	std::string		OUT_FILE_PATH);							// ���������� ����������



/**
 * Construction of a 2D LLE diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE2D(
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
	std::string		OUT_FILE_PATH);



/**
 * Construction of a 2D LLE diagram (for initial conditions)
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE2DIC(
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
	std::string		OUT_FILE_PATH);



/**
 * Construction of a 1D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LS1D(
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
	std::string		OUT_FILE_PATH);




/**
 * Construction of a 2D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LS2D(
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
	std::string		OUT_FILE_PATH);

void CUDA_dbscan(double* data, double* intervals, int* labels, int* helpfulArray, const int amountOfData, const double eps);

/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN) (for initial conditions)
 */
__host__ void basinsOfAttraction(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,								// ������� ��� ��������� DBSCAN 
	std::string		OUT_FILE_PATH);		

/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN) (for initial conditions)
 */
__host__ void basinsOfAttraction_2(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double* initialConditions,					// ������ � ���������� ���������
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps,
	std::string		OUT_FILE_PATH);								// ������� ��� ��������� DBSCAN 




