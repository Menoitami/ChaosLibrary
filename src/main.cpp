#include "bifurcationHOST.h"
#include "LLEHost.h"
#include "hostLibrary.cuh"
#include "LLECUDA.cuh"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <conio.h>
#include <chrono>
//using namespace old_library;

template <typename Func, typename... Args>
void measureExecutionTime(Func&& func, const std::string& fileName, Args&&... args) {

    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::string timingFileName = fileName + "_timing.txt";
    std::ofstream outFile(timingFileName);
    if (!outFile) {
        std::cerr << "Ошибка: Не удалось открыть файл для записи времени!" << std::endl;
        return;
    }
    outFile << "Время выполнения: " << duration << " мс\n";
    outFile.close();
    std::cout << "Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
}

int main()
{


	size_t startTime = std::clock();



	//double params[5]{ 0.5, 10, 28, (double)8 / 3, 1 };
	////double init[4]{ 0.1, 0.1, 25, 0 };
	//double init[3]{ 0.1, 0.1, 20 };
	////double params[6]{ 0.5, 0.25, 0.25, 0.5, 0.45, 0.15 };
	////double init[3]{ 1.57, 0.25, 0.5 };
	//double h = (double)0.001;

	//	basinsOfAttraction_2(
	//		300,							//const double	tMax,						// Время моделирования системы
	//		200,							//const int		nPts,						// Разрешение диаграммы
	//		h,							//const double	h,							// Шаг интегрирования
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//		init,							//const double* initialConditions,			// Массив с начальными условиями
	//		new double[4]{ -5, 5, -5, 5 },	//const double* ranges,					// Диапазоны изменения параметров
	//		new int[2]{ 0, 1 },				//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//		0,								//const int		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
	//		100000,							//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//		500,							//const double	transientTime,				// Время, которое будет промоделировано перед расчетом диаграммы
	//		params,							//const double* values,						// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//		1,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		0.25,							//const double	eps,						// Эпсион для алгоритма DBSCAN
	//		"D:\\CUDAresults\\Basins_LorenzDiss_1.csv"	//std::string		OUT_FILE_PATH
	//);

		//bifurcation1D(
		//	2000,							//const double	tMax,							// Время моделирования системы
		//	2000,							//const int		nPts,							// Разрешение диаграммы
		//	h,							//const double	h,								// Шаг интегрирования
		//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
		//	init,							//const double* initialConditions,				// Массив с начальными условиями
		//	new double[2]{-0.8, 0.8 },		//const double* ranges,							// Диаппазон изменения переменной
		//	new int[1]{ 4 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
		//	1,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
		//	1000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	2500,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
		//	params,								//const double* values,							// Параметры
		//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
		//	1,								//const int		preScaller,
		//	"D:\\CUDAresults\\Bif1D_DissipCons_0.csv"
		//);

	//FastSynchro_2(
	//	0.15,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//	201,	//const int		nPts,							// Разрешение диаграммы
	//	params,	//const double* values,								// Параметры
	//	sizeof(params) / sizeof(double),	//const int		amountOfValues,						// Количество параметров
	//	h,		//const double	h,									// Шаг интегрирования
	//	new double[4]{ -200,	200,	-200, 200 },	//const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 0,	1 },				//const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	new double[3]{ 0,	40,	0},//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//	new double[3]{ 0,	40,	0},//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//	init,//const double* initialConditions,			// Массив с начальными условиями мастера
	//	sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	1000000,							//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//	100,									//const int		iterOfSynchr,						// Число итераций синхронизации
	//	1,									//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	"D:\\CUDAresults\\FS2_Lorenz_0.csv"//std::string		OUT_FILE_PATH
	//);

		//FastSynchro(
		//500,	//const double	tMax,								// Время моделирования системы
		//0,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
		//0.15,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
		//params,	//const double* values,								// Параметры
		//sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
		//h,	//const double	h,									// Шаг интегрирования
		//new double[3]{ 0, 40, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
		//new double[3]{ 0, 40, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
		//new double[3]{ 0.1, 0.1, 20 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
		//new double[3]{ 0.1, 0.1, 20 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
		//sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
		//2000000,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
		//200,	//const int		iterOfSynchr,						// Число итераций синхронизации
		//1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//"D:\\CUDAresults\\FS3_Lorenz_2_0_40_0_and_0_40_0.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
		//);

	//FastSynchro_2(
	//	0.15,							//const double	NTime,						// Длина отрезка по которому будет проводиться синхронизация
	//	201,							//const int		nPts,						// Разрешение диаграммы
	//	params,							//const double* values,						// Параметры
	//	sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//	h,								//const double	h,							// Шаг интегрирования
	//	new double[4]{ -1000,	1000,	-1000, 1000 },	//const double* ranges,		// Диапазоны изменения параметров
	//	new int[2]{ 0,	1 },			//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//	new double[3]{ 0,	0,	0 },	//const double* kForward,					// Массив коэффициентов синхронизации вперед
	//	new double[3]{ 0,	0,	0 },	//const double* kBackward,					// Массив коэффициентов синхронизации назад
	//	init,							//const double* initialConditions,			// Массив с начальными условиями мастера
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//	1000000,						//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//	15,							//const int		iterOfSynchr,				// Число итераций синхронизации
	//	1,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	"D:\\CUDAresults\\FS2_Lorenz_CD_.csv"						//std::string		OUT_FILE_PATH
	//);

	//std::string str = "D:\\CUDAresults\\FS3_Lorenz_CD_0_40_0_and_0_40_0_es4_";
	//std::string path;

	//for (int i = 0; i < 51; i++) {
	//	path = str + std::to_string(i) + ".csv";
	//	params[4] =  0.0 + i * 0.02;
	//	//params[3] = 1.0 + i * 0.1;
	//	//params[2] = 25 + i * 1;
	//	//params[0] =  0.0 + i * 0.02;

	//	FastSynchro_2(
	//		0.15,							//const double	NTime,						// Длина отрезка по которому будет проводиться синхронизация
	//		201,							//const int		nPts,						// Разрешение диаграммы
	//		params,							//const double* values,						// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//		h,								//const double	h,							// Шаг интегрирования
	//		new double[4]{ -200,	200,	-200, 200 },	//const double* ranges,		// Диапазоны изменения параметров
	//		new int[2]{ 0,	1 },			//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//		new double[3]{ 0,	0,	0 },	//const double* kForward,					// Массив коэффициентов синхронизации вперед
	//		new double[3]{ 0,	0,	0 },	//const double* kBackward,					// Массив коэффициентов синхронизации назад
	//		init,							//const double* initialConditions,			// Массив с начальными условиями мастера
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//		1000000,						//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//		15,							//const int		iterOfSynchr,				// Число итераций синхронизации
	//		1,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		path							//std::string		OUT_FILE_PATH
	//	);
	//}

	//	FastSynchro(
	//		500,	//const double	tMax,								// Время моделирования системы
	//		1000,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//		0.15,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//		params,	//const double* values,								// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
	//		h,	//const double	h,									// Шаг интегрирования
	//		new double[3]{ 0, -40, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//		new double[3]{ 0, 0, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//		new double[3]{ 1.57, 0.25, 0.5 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//		new double[3]{ 1.56, 0.26, 0.45 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//		200,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//		200,	//const int		iterOfSynchr,						// Число итераций синхронизации
	//		1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		"D:\\CUDAresults\\FS_Lorenz_2_0_-40_0_and_0_0_0.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
	//	);

	//	FastSynchro(
	//		500,	//const double	tMax,								// Время моделирования системы
	//		1000,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//		0.15,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//		params,	//const double* values,								// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
	//		h,	//const double	h,									// Шаг интегрирования
	//		new double[3]{ 40, 0, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//		new double[3]{ 0, 0, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//		new double[3]{ 1.57, 0.25, 0.5 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//		new double[3]{ 1.56, 0.26, 0.45 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//		200,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//		200,	//const int		iterOfSynchr,						// Число итераций синхронизации
	//		1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		"D:\\CUDAresults\\FS_Lorenz_1_40_0_0_and_0_0_0.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
	//	);

	//	FastSynchro(
	//		500,	//const double	tMax,								// Время моделирования системы
	//		1000,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//		0.15,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//		params,	//const double* values,								// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
	//		h,	//const double	h,									// Шаг интегрирования
	//		new double[3]{ -40, 0, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//		new double[3]{ 0, 0, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//		new double[3]{ 1.57, 0.25, 0.5 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//		new double[3]{ 1.56, 0.26, 0.45 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//		200,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//		200,	//const int		iterOfSynchr,						// Число итераций синхронизации
	//		1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		"D:\\CUDAresults\\FS_Lorenz_1_-40_0_0_and_0_0_0.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
	//	);

	//std::string str = "D:\\CUDAresults\\FS_DissipCons_0";
	//std::string path;


//for (int i = 0; i < 81; i++) {
//	path = str + std::to_string(i) + ".csv";
//	
//	params[4] = -0.8 + i * 0.02;
//	FastSynchro(
//		2000,	//const double	tMax,								// Время моделирования системы
//		300,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		0.2,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		params,	//const double* values,								// Параметры
//		sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
//		h,	//const double	h,									// Шаг интегрирования
//		new double[3]{ 1, 1, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
//		new double[3]{ 1, 1, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
//		new double[3]{ 1.57, 0.25, 0.5 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
//		new double[3]{ 1.56, 0.26, 0.45 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		10000,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		500,	//const int		iterOfSynchr,						// Число итераций синхронизации
//		1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		path	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
//	);
//}


	//double params[8]{ 0.5, -1, 3, 1, -1, -1, 1, 1.54 };
	// double params[8]{ 0.5, -1, 3, 1, -1, -1, 1, 1.53 };
	// double init[3]{ 1, 2, 0};

	//basinsOfAttraction_2(
	//		500,							//const double	tMax,						// Время моделирования системы
	//		200,							//const int		nPts,						// Разрешение диаграммы
	//		0.01,							//const double	h,							// Шаг интегрирования
	//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//		init,							//const double* initialConditions,			// Массив с начальными условиями
	//		new double[4]{ -0.0001, 0.0001, -0.0001, 0.0001 },	//const double* ranges,					// Диапазоны изменения параметров
	//		new int[2]{ 0, 1 },				//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//		1,								//const int		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
	//		100000,							//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//		5000,							//const double	transientTime,				// Время, которое будет промоделировано перед расчетом диаграммы
	//		params,							//const double* values,						// Параметры
	//		sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//		2,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		0.25,							//const double	eps,						// Эпсион для алгоритма DBSCAN
	//		"D:\\CUDAresults\\Basins_Sprott_1.csv"	//std::string		OUT_FILE_PATH
	//);

	// old_library::bifurcation2D(
	// 		300, // const double tMax,
	// 		200, // const int nPts,
	// 		0.01, // const double h,
	// 		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	// 		init, // const double* initialConditions,
	// 		new double[4]{ -50, 50, -100, 100 }, // const double* ranges,
	// 		new int[2]{ 6, 7 }, // const int* indicesOfMutVars,
	// 		0, // const int writableVar,
	// 		10000, // const double maxValue,
	// 		2000, // const double transientTime,
	// 		params, // const double* values,
	// 		sizeof(params) / sizeof(double), // const int amountOfValues,
	// 		1, // const int preScaller,
	// 		0.05, //eps
	// 		std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_diagram_debri.csv"  // Используем BIFURCATION_OUTPUT_PATH
	// );

	//double params[6]{ 0.5, 3, 2.7, 4.7, 2, 9 };
	//double init[3]{ 1, 1, 0};
	//double h = (double)1 / 100251;

	//FastSynchro(
	//(double)100000 * h,	//const double	tMax,								// Время моделирования системы
	//(double)1000 * h,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//(double)100 * h,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//params,	//const double* values,								// Параметры
	//sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
	//h,	//const double	h,									// Шаг интегрирования
	//new double[3]{ 1e5, 0, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//new double[3]{ 0, 1e5, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//new double[3]{ 0.1, 0.1, 0.3 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//new double[3]{ 0.1, 0.1, 0 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//10000,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//200,	//const int		iterOfSynchr,						// Число итераций синхронизации
	//1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//"D:\\CUDAresults\\FastSynchro_analogChua_01.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
	//);


	//double params[6]{ 0.5, 3, 2.7, 4.7, 2, 9 };
	//double init[3]{ 1, 1, 0};

	//FastSynchro(
	//2500,	//const double	tMax,								// Время моделирования системы
	//500,	//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//0.35,	//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//params,	//const double* values,								// Параметры
	//sizeof(params) / sizeof(double),//const int		amountOfValues,						// Количество параметров
	//0.005,	//const double	h,									// Шаг интегрирования
	//new double[3]{ 0, 5.4, 0 },	//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//new double[3]{ 0, 5.4, 0 },	//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//new double[3]{ 1, 1, 0.3 },	//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//new double[3]{ 1, 1, 0.5 },	//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//10000,	//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//200,	//const int		iterOfSynchr,						// Число итераций синхронизации
	//1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//"D:\\CUDAresults\\FastSynchro_DadrasMimeni_02.csv"	//std::string		OUT_FILE_PATH						// Эпсилон для алгоритма DBSCAN 
	//);

	//double params[14];
	//params[0] = 0.5;		//params[0] = 0.5;
	//params[1] = 1;			//params[1] = 1;
	//params[2] = 0.78035;	//params[2] = 0.78035;
	//params[3] = 7.0772;		//params[3] = 7.0772;
	//params[4] = 22.7511;	//params[4] = 22.7511;
	//params[5] = -5.4068;	//params[5] = -5.4068;
	//params[6] = -50.59;		//params[6] = -50.59;
	//params[7] = -447.1443;	//params[7] = -447.1443;
	//params[8] = 18.8077;	//params[8] = 18.8077;
	//params[9] = 49;			//params[9] = -24.729;
	//params[10] = 7;			//params[10] = -5.0877;
	//params[11] = -579.5771;	//params[11] = -579.5771;
	//params[12] = 0;
	//params[13] = 0;

	//double params[10];
	//params[0] = 0.5;
	//params[1] = 1;
	//params[2] = 1;
	//params[3] = -0.1025;
	//params[4] = -2.3458;
	//params[5] = -2.6769;
	//params[6] = 34.0395;
	//params[7] = 31.6344;
	//params[8] = 347.3255;
	//params[9] = -798.794;


	////double init[3]{ 0.0, 0.01, 0.01 };
	//double init[3]{ -0.053798, -0.0268235, 0.00259754 };

	//	bifurcation1D(
	//	2500,							//const double	tMax,							// Время моделирования системы
	//	5000,							//const int		nPts,							// Разрешение диаграммы
	//	0.05,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 0.2, 2 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 0 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	2,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	1000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	50000,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\Bif1D_BZh_jerk_0.csv"
	//);

//					   b     a  m  w   mu
double params[7]{ 0.5, 3, 0, 2, 1, 18, 1 };
//double params[6]{ 0.5, 3, -1, 1, 1, 1.53 };
double init[3]{ 3, 3, 0 };
double h = (double)0.01;
double TT = 500000;
double CT = 10000;

	//bifurcation1D(
	//	CT,							//const double	tMax,							// Время моделирования системы
	//	3000,							//const int		nPts,							// Разрешение диаграммы
	//	h,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 1.5, 3 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 3 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	0,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	10000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_chameleon02_a_1.csv"
	//);

	//bifurcation1D(
	//	CT,							//const double	tMax,							// Время моделирования системы
	//	3000,							//const int		nPts,							// Разрешение диаграммы
	//	h,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 2, 4 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 1 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	0,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	10000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_chameleon02_b_1.csv"
	//);

	//bifurcation1D(
	//	CT,							//const double	tMax,							// Время моделирования системы
	//	3000,							//const int		nPts,							// Разрешение диаграммы
	//	h,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 0.8, 1.4 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 4 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	0,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	10000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_chameleon02_m_1.csv"
	 //);

	//bifurcation1D(
	//	CT,							//const double	tMax,							// Время моделирования системы
	//	3000,							//const int		nPts,							// Разрешение диаграммы
	//	h,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 10, 20 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 5 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	0,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	10000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_chameleon02_w.csv"
	//);

	//bifurcation1D(
	//	CT,							//const double	tMax,							// Время моделирования системы
	//	3000,							//const int		nPts,							// Разрешение диаграммы
	//	h,							//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ 0.5, 2.5 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 6 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	//	0,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	10000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,								//const double* values,							// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
	//	1,								//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_chameleon02_mu.csv"
	//);

	// --- ---

	//basinsOfAttraction_2(
	//	3000,							//const double	tMax,						// Время моделирования системы
	//	201,							//const int		nPts,						// Разрешение диаграммы
	//	0.01,							//const double	h,							// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,			// Массив с начальными условиями
	//	new double[4]{ -5, 5, -5, 5 },	//const double* ranges,					// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },				//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//	0,								//const int		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
	//	100000,							//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	2000,							//const double	transientTime,				// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,							//const double* values,						// Параметры
	//	sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//	10,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,							//const double	eps,						// Эпсион для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_chameleon02_3.csv"	//std::string		OUT_FILE_PATH
	//);



	// Bifurcation::bifurcation2D(
	// 	400, // const double tMax,
	// 	1000, // const int nPts,
	// 	h, // const double h,
	// 	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	// 	init, // const double* initialConditions,
	// 	//new double[4]{ -5, 5, -5, 5 }, // const double* ranges,
	// 	new double[4]{ 0, 5, 0, 20 }, // const double* ranges,
	// 	new int[2]{ 3, 5 }, // const int* indicesOfMutVars,
	// 	0, // const int writableVar,
	// 	10000, // const double maxValue,
	// 	5000, // const double transientTime,
	// 	params, // const double* values,
	// 	sizeof(params) / sizeof(double), // const int amountOfValues,
	// 	1, // const int preScaller,
	// 	0.001, //eps
	// 	std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_1.csv"
	// );

//LS1D(
//	1000,			//const double tMax,
//	0.5,			//const double NT,
//	1000,			//const int nPts,
//	h,				//const double h,
//	1e-6,			//const double eps,
//	init,			//const double* initialConditions,
//	sizeof(init) / sizeof(double), //const int amountOfInitialConditions,
//	new double[2]{ 0, 5 },	//const double* ranges,
//	new int[1]{ 5 },		//const int* indicesOfMutVars,
//	0,						//const int writableVar,
//	10000,					//const double maxValue,
//	5000,					//const double transientTime,
//	params,					//const double* values,
//	sizeof(params) / sizeof(double), //const int amountOfValues,
//	"D:\\CUDAresults\\LS1D_chameleon02_5.csv"
//);
//	new double[4]{ 0, 5, 0, 20 },

	// printf("Start_func\n");
	// LLE_constants::LLE2D(
	// 		500,		//const double tMax,
	// 		0.5,		//const double NT,
	// 		h,			//const double h,
	// 		1e-6,		//eps
	// 		1000,		//const trans time,
	// 		init,		// init
	// 		3,		//ammount inti,
	// 		params,	//params
	// 		7,	// amount params,
	// 		new double[3] {0,5,100},
	// 		new double[3] {0,20,100},
	// 		new int[2]{ 3, 5 },		
	// 		"LLE2D_my.csv");
	
		old_library::LLE2D(
			500,		//const double tMax,
			0.5,		//const double NT,
			100,		//const int nPts,
			h,			//const double h,
			1e-6,		//const double eps,
			init,		//const double* initialConditions,
			sizeof(init) / sizeof(double),		//const int amountOfInitialConditions,
			new double[4]{ 0, 5, 0, 20 },		//const double* ranges,
			new int[2]{ 3, 5 },					//const int* indicesOfMutVars,
			0,			//const int writableVar,
			10000,		//const double maxValue,
			1000,		//const double transientTime,
			params,		//const double* values,
			sizeof(params) / sizeof(double),	//const int amountOfValues,
			std::string(LLE_OUTPUT_PATH) + "/lle_1.csv"
		);

		//printf(" --- Time of runnig: %zu ms", std::clock() - startTime);


// measureExecutionTime(
// 		LLE2D,
// 		"LLE2D_chameleon02_3_5_TT300000_4.csv",
// 			5000,		//const double tMax,
// 			0.5,		//const double NT,
// 			800,		//const int nPts,
// 			h,			//const double h,
// 			1e-6,		//const double eps,
// 			init,		//const double* initialConditions,
// 			sizeof(init) / sizeof(double),		//const int amountOfInitialConditions,
// 			new double[4]{ 0, 5, 0, 20 },		//const double* ranges,
// 			new int[2]{ 3, 5 },					//const int* indicesOfMutVars,
// 			0,			//const int writableVar,
// 			10000,		//const double maxValue,
// 			300000,		//const double transientTime,
// 			params,		//const double* values,
// 			sizeof(params) / sizeof(double),	//const int amountOfValues,
// 			"LLE2D_chameleon02_3_5_TT300000_4.csv"
// 	);
		// old_library::LLE2D(
		// 	100,		//const double tMax,
		// 	0.5,		//const double NT,
		// 	100,		//const int nPts,
		// 	h,			//const double h,
		// 	1e-6,		//const double eps,
		// 	init,		//const double* initialConditions,
		// 	sizeof(init) / sizeof(double),		//const int amountOfInitialConditions,
		// 	new double[4]{ 0, 5, 0, 20 },		//const double* ranges,
		// 	new int[2]{ 3, 5 },					//const int* indicesOfMutVars,
		// 	0,			//const int writableVar,
		// 	10000,		//const double maxValue,
		// 	300000,		//const double transientTime,
		// 	params,		//const double* values,
		// 	sizeof(params) / sizeof(double),	//const int amountOfValues,
		// 	std::string(LLE_OUTPUT_PATH) + "/lle_1.csv"
		// );


	//basinsOfAttraction_2(
	//	1500,							//const double	tMax,						// Время моделирования системы
	//	201,							//const int		nPts,						// Разрешение диаграммы
	//	0.1,							//const double	h,							// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,			// Массив с начальными условиями
	//	new double[4]{ -2.2, 1, -0.3, 0.3 },	//const double* ranges,					// Диапазоны изменения параметров
	//	new int[2]{ 0, 2 },				//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//	0,								//const int		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
	//	100000,							//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	15000,							//const double	transientTime,				// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,							//const double* values,						// Параметры
	//	sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//	1,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.15,							//const double	eps,						// Эпсион для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_BZh_1.csv"	//std::string		OUT_FILE_PATH
	//);





	//double params[8]{ 0.5, -1, 1, 0, 1, 1, -7, -1.04 };
							
	//double params[8]{ 0.5, -1, 1, 0, 1, 2, -7, -1 };
	//double params[8]{ 0.5, -1, 1, 0, 1, 1.0, -7, -0.45 };
	//double params[8]{ 0.5, -1, 1, 0, 1, 1.74, -7, -0.8 };
	//double init[3]{ -4, 0, 0};


	//basinsOfAttraction_2(
	//	300,							//const double	tMax,						// Время моделирования системы
	//	700,							//const int		nPts,						// Разрешение диаграммы
	//	0.01,							//const double	h,							// Шаг интегрирования
	//	sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,	// Количество начальных условий ( уравнений в системе )
	//	init,							//const double* initialConditions,			// Массив с начальными условиями
	//	new double[4]{ -0.1, 0.1, 5.8, 9.2 },	//const double* ranges,					// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },				//const int* indicesOfMutVars,				// Индексы изменяемых параметров
	//	0,								//const int		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
	//	100000,							//const double	maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	1500,							//const double	transientTime,				// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,							//const double* values,						// Параметры
	//	sizeof(params) / sizeof(double),//const int		amountOfValues,				// Количество параметров
	//	5,								//const int		preScaller,					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.15,							//const double	eps,						// Эпсион для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Timur_Chinease_b0.80_w1.74_1.csv"	//std::string		OUT_FILE_PATH
	//);

//		bifurcation2D(
//		300, // const double tMax,
//		300, // const int nPts,
//		1.0e-2, // const double h,
//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
//		init, // const double* initialConditions,
//		new double[4]{ -0.15, 0.01, -20, -13 }, // const double* ranges,
//		new int[2]{ 3, 6 }, // const int* indicesOfMutVars,
//		0, // const int writableVar,
//		100000, // const double maxValue,
//		1500, // const double transientTime,
//		params, // const double* values,
//		sizeof(params) / sizeof(double), // const int amountOfValues,
//		5, // const int preScaller,
//		0.02, //eps
//		"D:\\CUDAresults\\Bif2D_Timur_Chinease_01.csv"
//);


	//a[1] = 3.75;  //A  = 3.75;
//a[2] = 10;    //B  = 10;
//a[3] = 1;     //d  = 1;
//a[4] = -1;    //n  = - 1;
//a[5] = -0.33; //m0 = - 0.33;
//a[6] = 0.25;  //m1 = 0.25;
//a[7] = 5.5;   //T  = 5.5;
//a[8] = 10;    //k = 10;


	//double params[9]{ 0.5, 3.75, 10, 1, -1, -0.33, 0.25, 5.5, 10};
	//double init[5]{ 0.01, 0.1, 0, 0, 0};
	//double CT = 150;
	//double TT = 300;
	//double ress = 1200;
	//double h = 0.0001;

//	// a
//	bifurcation1D(
//		CT,							//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,							//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ 3.2, 4 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1]{ 1 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_a_HR.csv"
//	);
////
////	// b
//	bifurcation1D(
//		CT,								//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2] { 9, 12 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1] { 2 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_b_HR.csv"
//);
////
////	// d
//	bifurcation1D(
//		CT,								//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ 0.6, 2.0 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1]{ 3 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_d_HR.csv"
//	);
////
////	// n
//	bifurcation1D(
//		CT,								//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ -1.3, 0.3 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1]{ 4 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_n_HR.csv"
//	);
////
////	// m0
//	bifurcation1D(
//		CT,								//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ -0.45, -0.2 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1]{ 5 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_m0_HR.csv"
//	);
//
//	//m1
//	bifurcation1D(
//		CT,								//const double	tMax,							// Время моделирования системы
//		ress,							//const int		nPts,							// Разрешение диаграммы
//		h,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),	//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,							//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ 0, 4 },		//const double* ranges,							// Диаппазон изменения переменной
//		new int[1]{ 6 },				//const int* indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
//		4,								//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000000000,					//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,							//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,								//const double* values,							// Параметры
//		sizeof(params) / sizeof(double),		//const int		amountOfValues,					// Количество параметров
//		1,								//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_ECG_model_m1_HR.csv"
//	);

//	//a b
//bifurcation2D(
//	100, // const double tMax,
//	800, // const int nPts,
//	1.0e-4, // const double h,
//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
//	init, // const double* initialConditions,
//	new double[4]{ 3, 4, 7.5, 11 }, // const double* ranges,
//	new int[2]{ 1, 2 }, // const int* indicesOfMutVars,
//	4, // const int writableVar,
//	100000, // const double maxValue,
//	50, // const double transientTime,
//	params, // const double* values,
//	sizeof(params) / sizeof(double), // const int amountOfValues,
//	10, // const int preScaller,
//	0.001, //eps
//	"D:\\CUDAresults\\bif2D_ECG_model_ab.csv"
//);
//
//	//a d
//	bifurcation2D(
//		100, // const double tMax,
//		800, // const int nPts,
//		1.0e-4, // const double h,
//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
//		init, // const double* initialConditions,
//		new double[4]{1.5, 4.5, 0, 1.4 }, // const double* ranges,
//		new int[2]{ 1, 3 }, // const int* indicesOfMutVars,
//		4, // const int writableVar,
//		100000, // const double maxValue,
//		50, // const double transientTime,
//		params, // const double* values,
//		sizeof(params) / sizeof(double), // const int amountOfValues,
//		10, // const int preScaller,
//		0.001, //eps
//		"D:\\CUDAresults\\bif2D_ECG_model_ad.csv"
//);
//
//	//d n
//	bifurcation2D(
//		100, // const double tMax,
//		800, // const int nPts,
//		1.0e-4, // const double h,
//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
//		init, // const double* initialConditions,
//		new double[4]{ 0.4, 1.6, -2.5, 2.5 }, // const double* ranges,
//		new int[2]{ 3, 4 }, // const int* indicesOfMutVars,
//		4, // const int writableVar,
//		100000, // const double maxValue,
//		50, // const double transientTime,
//		params, // const double* values,
//		sizeof(params) / sizeof(double), // const int amountOfValues,
//		10, // const int preScaller,
//		0.001, //eps
//		"D:\\CUDAresults\\bif2D_ECG_model_dn.csv"
//	);
//
//	//m0 m1
//	bifurcation2D(
//		100, // const double tMax,
//		800, // const int nPts,
//		1.0e-4, // const double h,
//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
//		init, // const double* initialConditions,
//		new double[4]{ -0.5, -0.25, 0, 6 }, // const double* ranges,
//		new int[2]{ 5, 6 }, // const int* indicesOfMutVars,
//		4, // const int writableVar,
//		100000, // const double maxValue,
//		50, // const double transientTime,
//		params, // const double* values,
//		sizeof(params) / sizeof(double), // const int amountOfValues,
//		10, // const int preScaller,
//		0.001, //eps
//		"D:\\CUDAresults\\bif2D_ECG_model_m0m1.csv"
//	);

// measureExecutionTime(
// 		LLE2D,
// 		"LLE2D_sem_02.csv",
// 		1000, // const double tMax,
// 		0.4, // const double NT,
// 		200, // const int nPts,
// 		1e-2, // const double h,
// 		1e-8, // const double eps,
// 		new double[4]{ 0.009, 0.003, 0, 0 }, // const double* initialConditions,
// 		4, // const int amountOfInitialConditions,
// 		new double[4]{ -0.015, 0.015, -180, 180}, // const double* ranges,
// 		new int[2]{ 1, 2 }, // const int* indicesOfMutVars,
// 		0, // const int writableVar,
// 		10000, // const double maxValue,
// 		20000, // const double transientTime,
// 		new double[4]{ 0.5, -0.015, -180, 0.1 }, // const double* values,
// 		4,
// 		"LLE2D_sem_02.csv"); // const int amountOfValues);

	//LLE2D(
	//	1000, // const double tMax,
	//	0.4, // const double NT,
	//	200, // const int nPts,
	//	1e-2, // const double h,
	//	1e-8, // const double eps,
	//	new double[4]{ 0.009, 0.003, 0, 0 }, // const double* initialConditions,
	//	4, // const int amountOfInitialConditions,
	//	new double[4]{ -0.015, 0.015, -180, 180}, // const double* ranges,
	//	new int[2]{ 1, 2 }, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	10000, // const double maxValue,
	//	20000, // const double transientTime,
	//	new double[4]{ 0.5, -0.015, -180, 0.1 }, // const double* values,
	//	4,
	//	"D:\\CUDAresults\\LLE2D_sem_02.csv"); // const int amountOfValues);


	// --- Lorenz ---
	//double params[4]{ 0.5, 10, 28, 2.666666666666666};
	//double Xm[3]{ 3, -3, 0 };
	//double Xs[3]{ 1, 1 , 20};
	//double K_Forward[3]	{ 0, 40, 0 };
	//double K_Backward[3]{ 0, 40, 0 };

	//// --- Nose-Hoover ---
	//double params[4]{ 0.5, 1, 1, 1 };
	//double Xm[3]{ 3, -3, 0 };
	//double Xs[3]{ 3.1, 2.9 , 0 };
	//double K_Forward[3]{ 0, 5, 0 };
	//double K_Backward[3]{ 0, 5, 0 };

	// --- Rossler ---
	//double params[4]{ 0.5, 0.2, 0.2, 5.7 };
	//double Xm[3]{ 3, -3, 0 };
	//double Xs[3]{ 3.1, 2.9 , 0 };
	//double K_Forward[3]{ 0, 2, 0 };
	//double K_Backward[3]{ 0, 10, 0 };

	//// --- Case B ---
	//double params[4]{ 0.5, 1, 1, 1};
	//double Xm[3]{ -1, -1, -1 };
	//double Xs[3]{ 1, 1 , 1};
	//double K_Forward[3]	{ 0, 20, 0 };
	//double K_Backward[3]{ 0, 7, 0 };

	//FastSynchro(
	//	2000,									//const double	tMax,								// Время моделирования системы
	//	300,									//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	10.00,									//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//	params,									//const double* values,								// Параметры
	//	sizeof(params) / sizeof(double),		//const int		amountOfValues,						// Количество параметров
	//	0.01,									//const double	h,									// Шаг интегрирования
	//	K_Forward,								//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//	K_Backward,								//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//	Xm,										//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//	Xs,										//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//	sizeof(Xm) / sizeof(double),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	1000000,								//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//	20,									//const int		iterOfSynchr,						// Число итераций синхронизации
	//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	"D:\\CUDAresults\\FS_test_002.csv"		//std::string		OUT_FILE_PATH
	//);		


		//double param[7]{ 0.5, 5.8, 3.7, 2, 0.9, 1, 1.5 };
		//double init[4]{ 0.98, 1.9 , 0.98, -0.98};
		
		//	bifurcation1D(
		//	200,								//const double	tMax,							// Время моделирования системы
		//	50000,								//const int		nPts,							// Разрешение диаграммы
		//	0.01,								//const double	h,								// Шаг интегрирования
		//	sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
		//	init,								//const double* initialConditions,				// Массив с начальными условиями
		//	new double[2]{ 6, 6.2 },			//const double* ranges,							// Диапазон изменения переменной
		//	new int[1]{ 1 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
		//	0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
		//	100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	100,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
		//	param,									//const double* values,							// Параметры
		//	sizeof(param) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
		//	1,									//const int		preScaller,
		//	"D:\\CUDAresults\\Babkin_1dBif.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//);

		//bifurcation2D(
		//	500, // const double tMax,
		//	1000, // const int nPts,
		//	0.01, // const double h,
		//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
		//	init, // const double* initialConditions,
		//	new double[4]{ 0, 100, 0, 100 }, // const double* ranges,
		//	new int[2]{ 4, 5 }, // const int* indicesOfMutVars,
		//	1, // const int writableVar,
		//	100000, // const double maxValue,
		//	1000, // const double transientTime,
		//	param, // const double* values,
		//	sizeof(param) / sizeof(double), // const int amountOfValues,
		//	1, // const int preScaller,
		//	0.5, //eps
		//	"D:\\CUDAresults\\Babkin_2dBif.csv"
		//);


				//double param[3]{ 0.5, 0.3, 1 };
		//double init[3]{ 1, 1 , 1 };

	//		bifurcation2D(
	//		300, // const double tMax,
	//		300, // const int nPts,
	//		0.01, // const double h,
	//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//		init, // const double* initialConditions,
	//		new double[4]{ 0, 2, 0, 2 }, // const double* ranges,
	//		new int[2]{ 1, 2 }, // const int* indicesOfMutVars,
	//		0, // const int writableVar,
	//		100000000000, // const double maxValue,
	//		2000, // const double transientTime,
	//		param, // const double* values,
	//		sizeof(param) / sizeof(double), // const int amountOfValues,
	//		1, // const int preScaller,
	//		0.1, //eps
	//		"D:\\CUDAresults\\Makarov_2dBif.csv"
	//);




	//std::string str = "D:\\CUDAresults\\FS_CaseB_000";
	//std::string path;
	//int ress = 200;
	//double NT_start = 1.0;
	//double NT_delta = 0.1;

	//for (int i = 0; i < ress + 1; i++) {
	//	path = str + std::to_string(i) + ".csv";

	//	FastSynchro(
	//		500,													//const double	tMax,								// Время моделирования системы
	//		300,													//const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//		NT_start + i* NT_delta,									//const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	//		params,													//const double* values,								// Параметры
	//		sizeof(params) / sizeof(double),						//const int		amountOfValues,						// Количество параметров
	//		0.01,													//const double	h,									// Шаг интегрирования
	//		K_Forward,												//const double* kForward,							// Массив коэффициентов синхронизации вперед
	//		K_Backward,												//const double* kBackward,							// Массив коэффициентов синхронизации назад
	//		Xm,														//const double* initialConditionsMaster,			// Массив с начальными условиями мастера
	//		Xs,														//const double* initialConditionsSlave,				// Массив с начальными условиями слейва
	//		sizeof(Xm) / sizeof(double),							//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//		1000000,												//const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	//		150,													//const int		iterOfSynchr,						// Число итераций синхронизации
	//		2,					//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//		path//std::string		OUT_FILE_PATH
	//	);
	//	
	//}

	//double param[7]{ 0.5, 29.215, 0.707, 1.25, 6.9, 0.367, 0.0478 };
	//double init[3]{ 1, 0 , 0 };

	//basinsOfAttraction_2(
	//300,									// Время моделирования системы
	//100,									// Разрешение диаграммы
	//0.001,									// Шаг интегрирования
	//sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//init,									// Массив с начальными условиями
	//new double[4]{ 0.2, 0.3, 0.3, 0.34 },	// Диапазоны изменения параметров
	//new int[2]{ 1, 2 },						// Индексы изменяемых параметров
	//1,										// Индекс уравнения, по которому будем строить диаграмму
	//100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//200000,									// Время, которое будет промоделировано перед расчетом диаграммы
	//param,									// Параметры
	//sizeof(param) / sizeof(double),		// Количество параметров
	//100,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//0.1,									// Эпсилон для алгоритма DBSCAN
	//"D:\\CUDAresults\\Basins_nhe1.csv");


	//double param[8]{ 0, 4.5E-9, 0.92E-3, 144, 0.02, 0.011, 325000, 0};
	//double param[8]{ 0, 5.5E-9, 0.5E-3, 15000,  0.01, 0.005, 1025000, 0 };
	//double init[3]{ 0, 0 , 0 };

	//bifurcation2D(
	//0.0005, // const double tMax,
	//100, // const int nPts,
	//1e-8, // const double h,
	//sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//init, // const double* initialConditions,
	//new double[4]{ 100, 5000, 200000, 2000000 }, // const double* ranges,
	//new int[2]{ 3, 6 }, // const int* indicesOfMutVars,
	//0, // const int writableVar,
	//100000000000, // const double maxValue,
	//0.0005, // const double transientTime,
	//param, // const double* values,
	//sizeof(param) / sizeof(double), // const int amountOfValues,
	//3, // const int preScaller,
	//0.01, //eps
	//"D:\\CUDAresults\\FHN_GI305a_2Dbif.csv"
	//);



//	double a[4]{ 0, 0.2, 0.2, 5.7};
//	double init[3]{ 0.01, 0 , 0 };
//
//	bifurcation1D(
//	500,								//const double	tMax,							// Время моделирования системы
//	2000,								//const int		nPts,							// Разрешение диаграммы
//	0.01,								//const double	h,								// Шаг интегрирования
//	sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	init,								//const double* initialConditions,				// Массив с начальными условиями
//	new double[2]{ -0.3, 0.3 },			//const double* ranges,							// Диапазон изменения переменной
//	new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	15000,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	a,									//const double* values,							// Параметры
//	sizeof(a) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
//	1,									//const int		preScaller,
//	"D:\\CUDAresults\\bif1D_Rossler_01_h=0.05.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//);
//
//	bifurcation1D(
//		500,								//const double	tMax,							// Время моделирования системы
//		2000,								//const int		nPts,							// Разрешение диаграммы
//		0.01,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,								//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ -0.3, 0.3 },			//const double* ranges,							// Диапазон изменения переменной
//		new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		15000,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		a,									//const double* values,							// Параметры
//		sizeof(a) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
//		1,									//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_Rossler_01_h=0.01.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	);
//
//	bifurcation1D(
//		500,								//const double	tMax,							// Время моделирования системы
//		2000,								//const int		nPts,							// Разрешение диаграммы
//		0.005,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,								//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ -0.3, 0.3 },			//const double* ranges,							// Диапазон изменения переменной
//		new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		15000,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		a,									//const double* values,							// Параметры
//		sizeof(a) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
//		2,									//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_Rossler_01_h=0.005.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	);
//
//	bifurcation1D(
//		500,								//const double	tMax,							// Время моделирования системы
//		2000,								//const int		nPts,							// Разрешение диаграммы
//		0.001,								//const double	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,								//const double* initialConditions,				// Массив с начальными условиями
//		new double[2]{ -0.3, 0.3 },			//const double* ranges,							// Диапазон изменения переменной
//		new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		15000,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		a,									//const double* values,							// Параметры
//		sizeof(a) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
//		10,									//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_Rossler_01_h=0.001.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	);
	//double params[4]{ 3, 2.2, 1, 0.001};
	//double init[3]{ 1.2, 1.0 , -0.02 };


//	double params[3]{ 1.9, -1.8, 3.9 };
//	double init[3]{ -0.05, 0,0 };
//
//
//basinsOfAttraction_2(
//	2000,									// Время моделирования системы
//	100,									// Разрешение диаграммы
//	0.01,									// Шаг интегрирования
//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
//	init,									// Массив с начальными условиями
//	new double[4]{ -1, 0, -1, 0 },	// Диапазоны изменения параметров
//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
//	0,										// Индекс уравнения, по которому будем строить диаграмму
//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	2000,									// Время, которое будет промоделировано перед расчетом диаграммы
//	params,									// Параметры
//	sizeof(params) / sizeof(double),		// Количество параметров
//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	1.1,									// Эпсилон для алгоритма DBSCAN
//	"D:\\CUDAresults\\Basins_CaseJTreshka_1.csv");


	//double init[3]{ 0.2, 0, 0 };
	//double a[3]{0.2, 0.2, 5.7};

		//neuronClasterization2D(
		//	50, // const double tMax,
		//	50, // const int nPts,
		//	0.01, // const double h,
		//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
		//	init, // const double* initialConditions,

		//	new double[4]{ 0.05, 0.35, -0.05, 0.5 }, // const double* ranges,
		//	new int[2]{ 0, 1 }, // const int* indicesOfMutVars,
		//	0, // const int writableVar,
		//	100000000, // const double maxValue,
		//	//3600, // const int maxAmountOfPeaks,
		//	5000, // const double transientTime,
		//	a, // const double* values,
		//	sizeof(a) / sizeof(double), // const int amountOfValues,
		//	10, // const int preScaller,
		//	0.01, //eps
		//	"D:\\CUDAresults\\Rossler_2Dclassification.csv"
		//);

	//TimeDomainCalculation(
	//300,								//const double	tMax,							// Время моделирования системы
	//30,								//const int		nPts,							// Разрешение диаграммы1
	//0.01,								//const double	h,								// Шаг интегрирования
	//sizeof(init) / sizeof(double),		// const int amountOfInitialConditions,
	//init,								// const double* initialConditions,
	//new double[2]{ 0.34, 0.35 },		//const double* ranges,							// Диаппазон изменения переменной
	//new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	//0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//500,								//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//a,									// const double* values,
	//sizeof(a) / sizeof(double),			// const int amountOfValues,
	//1,									//const int		preScaller,
	//"D:\\CUDAresults\\Rossler_TimeDomain.csv"	//std::string		OUT_FILE_PATH
	//);						

	
	double a[30];	
		/* GI403a	*/								
	/* Symmetry =	*/ a[0] = 0.5;                          
	/* A =			*/ a[1] = 2;
	/* Ip =			*/ a[2] = 6.3e-5;
	/* Iv =			*/ a[3] = 6e-6;
	/* Is =			*/ a[4] = 1.1E-7; 
	/* Ilk =		*/ a[5] = 1E-12;
	/* U1 =			*/ a[6] = 0; //0.005
	/* U2 =			*/ a[7] = 0.04; //0.0
	/* C1 =			*/ a[8] = 2.2e-8;
	/* Vt =			*/ a[9] = 0.0588235;
	/* Vp =			*/ a[10] = 0.038; 
	/* Vth_p =		*/ a[11] = 0.267;
	/* Vh_p =		*/ a[12] = 0.08;
	/* Vth_n =		*/ a[13] = -0.119;
	/* Vh_n =		*/ a[14] = -0.006;
	/* VTset =		*/ a[15] = 0.0099; 
	/* VTreset =	*/ a[16] = 0.0175;
	/* D =			*/ a[17] = 20;
	/* E =			*/ a[18] = 0.09;
	/* Ron_p =		*/ a[19] = 806;
	/* Ron_n =		*/ a[20] = 1434;
	/* TAUset =		*/ a[21] = 1.2E-7;
	/* TAUreset =	*/ a[22] = 1.3E-7;
	/* Dset =		*/ a[23] = 0.05;
	/* Dreset =		*/ a[24] = 0.5;
	/* Ampl_Iin =	*/ a[25] = 6.3e-5;
	/* T_Iin =		*/ a[26] = 0.05;
	/* Pulse_Iin =	*/ a[27] = 0.048;
	/* Offset =		*/ a[28] =-0.002;
	/* RampStep =	*/ a[29] = 4.0E-7;

		/* GI401a	*/ //
	/* Ip =			*/ a[2] = 2.1e-5;
	/* Iv =			*/ a[3] = -3e-6;
	/* Is =			*/ a[4] = 1.15E-7;
	/* Vt =			*/ a[9] = 0.066666666667;
	/* Vp =			*/ a[10] = 0.09;
	/* D =			*/ a[17] = 26;
	/* E =			*/ a[18] = 0.14;
	/* Ampl_Iin =	*/ //a[25] = 1.95e-5;
	/* T_Iin =		*/ //a[26] = 0.052;
	/* Pulse_Iin =	*/ //a[27] = 0.05;
	/* Offset =		*/ //a[28] = -0.002;
	/* RampStep =	*/ //a[29] = 1.2E-7;

		/* BD4	*/ //
	/* Ip =			*/ //a[2] = 4.8e-5;
	/* Iv =			*/ //a[3] = 2.0e-6;
	/* Is =			*/ //a[4] = 1.0E-8;
	/* Vt =			*/ //a[9] = 0.047619048;
	/* Vp =			*/ //a[10] = 0.04;
	/* D =			*/ //a[17] = 24;
	/* E =			*/ //a[18] = 0.15;
	/* Ampl_Iin =	*/ //a[25] = 4.8e-5;
	/* T_Iin =		*/ //a[26] = 0.052;
	/* Pulse_Iin =	*/ //a[27] = 0.05;
	/* Offset =		*/ //a[28] = -0.002;
	/* RampStep =	*/ //a[29] = 2.0E-7;

			/* BD5	*/ //
	/* Ip =			*/ //a[2] = 1.48e-5;
	/* Iv =			*/ //a[3] = 1.0e-6;
	/* Is =			*/ //a[4] = 1.0E-8;
	/* Vt =			*/ //a[9] = 0.0490196;
	/* Vp =			*/ //a[10] = 0.033;
	/* D =			*/ //a[17] = 13;
	/* E =			*/ //a[18] = 0.07;
	/* Ampl_Iin =	*/ //a[25] = 1.47e-5;
	/* T_Iin =		*/ //a[26] = 0.052;
	/* Pulse_Iin =	*/ //a[27] = 0.05;
	/* Offset =		*/ //a[28] = -0.002;
	/* RampStep =	*/ //a[29] = 1.5E-7;

	//double init[8]{ 0, 0, 0, a[11], a[12], 0, 0, 0 };

//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
//NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE //NEURON AND_TS HERE 
	
	//If input is sin, Iin = a[28] + a[25] * sin(2 * pi* a[26] * X1[2] + a[27]);
//	a[25] = 0.5e-6;
//	a[26] = 1000;
//	a[27] = 0;
//	a[28] = 21e-6;
//
//
//double init[4]{ 0, 0, 1e-9, 0 };


	   
//	a[8] = 1.12632e-7;

//	TimeDomainCalculation(
//	0.10,								//const double	tMax,							// Время моделирования системы
//	2,									//const int		nPts,							// Разрешение диаграммы1
//	1e-7,								//const double	h,								// Шаг интегрирования
//	sizeof(init) / sizeof(double),		// const int amountOfInitialConditions,
//	init,								// const double* initialConditions,
//	new double[2]{ 0.04, 0.05 },		//const double* ranges,							// Диаппазон изменения переменной
//	new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	100000000000,						//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	0,								//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	a,									// const double* values,
//	sizeof(a) / sizeof(double),			// const int amountOfValues,
//	200,									//const int		preScaller,
//	"D:\\CUDAresults\\TimeDomain_neuron.csv"	//std::string		OUT_FILE_PATH
//);



	//bifurcation2D(
	//0.1, // const double tMax,
	//600, // const int nPts,
	//5.0e-7, // const double h,
	//sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//init, // const double* initialConditions,
	//new double[4]{ 0, 2.0e-6, 100, 2200 }, // const double* ranges,
	//new int[2]{ 25, 26 }, // const int* indicesOfMutVars,
	//0, // const int writableVar,
	//100000000000, // const double maxValue,
	//0.1, // const double transientTime,
	//a, // const double* values,
	//sizeof(a) / sizeof(double), // const int amountOfValues,
	//25, // const int preScaller,
	//0.05, //eps
	//"D:\\CUDAresults\\LLE2D_AND_TS_neuron2.csv"
	//);

// measureExecutionTime(
// 		LLE2D,
// 		"LLE2D_AND_TS_neuron3.csv",
// 		0.5, // const double tMax,
// 		5e-4, // const double NT,
// 		1000, // const int nPts,
// 		5e-7, // const double h,
// 		1e-5, // const double eps,
// 		init, // const double* initialConditions,
// 		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
// 		new double[4]{ 0, 2.0e-6, 100, 2200 }, // const double* ranges,
// 		new int[2]{ 25, 26 }, // const int* indicesOfMutVars,
// 		0, // const int writableVar,
// 		10000, // const double maxValue,
// 		0.1, // const double transientTime,
// 		a, // const double* values,
// 		sizeof(a) / sizeof(double),
// 		"LLE2D_AND_TS_neuron3.csv"); 
	//LLE2D(
	//0.5, // const double tMax,
	//5e-4, // const double NT,
	//1000, // const int nPts,
	//5e-7, // const double h,
	//1e-5, // const double eps,
	//init, // const double* initialConditions,
	//sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//new double[4]{ 0, 2.0e-6, 100, 2200 }, // const double* ranges,
	//new int[2]{ 25, 26 }, // const int* indicesOfMutVars,
	//0, // const int writableVar,
	//10000, // const double maxValue,
	//0.1, // const double transientTime,
	//a, // const double* values,
	//sizeof(a) / sizeof(double),
	//"D:\\CUDAresults\\LLE2D_AND_TS_neuron3.csv"); 



	//neuronClasterization2D_2(
	//	1,											// const double	tMax,								// Время моделирования системы
	//	20,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	new double[4]{ -0.01, 0.05, 1.0E-8, 4.0E-7},	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	//new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_GI403a_U1C1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	170,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	//new double[4]{ -0.01, 0.05, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_GI403a_U1U2.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

				/* GI401a	*/ //
		/* Ip =			*/ a[2] = 2.1e-5;
		/* Iv =			*/ a[3] = -3e-6;
		/* Is =			*/ a[4] = 1.15E-7;
		/* Vt =			*/ a[9] = 0.066666666667;
		/* Vp =			*/ a[10] = 0.09;
		/* D =			*/ a[17] = 26;
		/* E =			*/ a[18] = 0.14;
		/* Ampl_Iin =	*/ a[25] = 1.8e-5;
		/* T_Iin =		*/ a[26] = 0.052;
		/* Pulse_Iin =	*/ a[27] = 0.05;
		/* Offset =		*/ a[28] = -0.002;
		/* RampStep =	*/ a[29] = 1.8E-7;

		//neuronClasterization2D_2(
		//	2,											// const double	tMax,								// Время моделирования системы
		//	170,											// const int		nPts,								// Разрешение диаграммы
		//	1e-7,										// const double	h,									// Шаг интегрирования
		//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
		//	init,										// const double* initialConditions,					// Массив с начальными условиями
		////// U1 - C1
		//	new double[4]{ 0.025, 0.125, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
		//	new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		////// U1 - U2
		//	//new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
		//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
		//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
		//	a,											// const double* values,								// Параметры
		//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
		//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//	0.000001,										// const double	eps,
		//	"D:\\CUDAresults\\neuron2D_GI401a_U1C1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
		//);

		//neuronClasterization2D_2(
		//	2,											// const double	tMax,								// Время моделирования системы
		//	170,											// const int		nPts,								// Разрешение диаграммы
		//	1e-7,										// const double	h,									// Шаг интегрирования
		//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
		//	init,										// const double* initialConditions,					// Массив с начальными условиями
		////// U1 - C1
		//	//new double[4]{ -0.01, 0.05, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
		//	//new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		////// U1 - U2
		//	new double[4]{ 0.0, 0.06, 0, 0.09 },	// const double* ranges,								// Диапазоны изменения параметров
		//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
		//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
		//	a,											// const double* values,								// Параметры
		//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
		//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//	0.000001,										// const double	eps,
		//	"D:\\CUDAresults\\neuron2D_GI401a_U1U2.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
		//);

		/* BD4	*/ //
	/* Ip =			*/   a[2] = 4.8e-5;
	/* Iv =			*/   a[3] = 2.0e-6;
	/* Is =			*/   a[4] = 1.0E-8;
	/* Vt =			*/   a[9] = 0.047619048;
	/* Vp =			*/   a[10] = 0.04;
	/* D =			*/   a[17] = 24;
	/* E =			*/   a[18] = 0.15;
	/* Ampl_Iin =	*/   a[25] = 4.6e-5;
	/* T_Iin =		*/   a[26] = 0.052;
	/* Pulse_Iin =	*/   a[27] = 0.05;
	/* Offset =		*/   a[28] = -0.002;
	/* RampStep =	*/   a[29] = 3.0E-7;

		//neuronClasterization2D_2(
		//	2,											// const double	tMax,								// Время моделирования системы
		//	170,											// const int		nPts,								// Разрешение диаграммы
		//	1e-7,										// const double	h,									// Шаг интегрирования
		//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
		//	init,										// const double* initialConditions,					// Массив с начальными условиями
		////// U1 - C1
		//	new double[4]{ -0.015, 0.06, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
		//	new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		////// U1 - U2
		//	//new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
		//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
		//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
		//	a,											// const double* values,								// Параметры
		//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
		//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//	0.000001,										// const double	eps,
		//	"D:\\CUDAresults\\neuron2D_BD4_U1C1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
		//);

		//neuronClasterization2D_2(
		//	2,											// const double	tMax,								// Время моделирования системы
		//	170,											// const int		nPts,								// Разрешение диаграммы
		//	1e-7,										// const double	h,									// Шаг интегрирования
		//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
		//	init,										// const double* initialConditions,					// Массив с начальными условиями
		////// U1 - C1
		//	//new double[4]{ -0.01, 0.05, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
		//	//new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		////// U1 - U2
		//	new double[4]{ 0.03, 0.07, -0.06, 0.0 },	// const double* ranges,								// Диапазоны изменения параметров
		//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
		//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
		//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
		//	a,											// const double* values,								// Параметры
		//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
		//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		//	0.000001,										// const double	eps,
		//	"D:\\CUDAresults\\neuron2D_BD4_U1U2.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
		//);

			/* BD5	*/   
	/* Ip =			*/   a[2] = 1.48e-5;
	/* Iv =			*/   a[3] = 1.0e-6;
	/* Is =			*/   a[4] = 1.0E-8;
	/* Vt =			*/   a[9] = 0.0490196;
	/* Vp =			*/   a[10] = 0.033;
	/* D =			*/   a[17] = 13;
	/* E =			*/   a[18] = 0.07;
	/* Ampl_Iin =	*/   a[25] = 1.3e-5;
	/* T_Iin =		*/   a[26] = 0.052;
	/* Pulse_Iin =	*/   a[27] = 0.05;
	/* Offset =		*/   a[28] = -0.002;
	/* RampStep =	*/   a[29] = 2.1E-7;

	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	170,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	new double[4]{ -0.02, 0.05, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	//new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_BD5_U1C1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	170,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	//new double[4]{ -0.01, 0.05, 1.0E-8, 4.0E-7 },	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,8 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	new double[4]{ 0, 0.04, -0.03, 0.02 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_BD5_U1U2.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);
















	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	200,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	//new double[4]{ -0.02, 0.04, 1.0E-8, 3.5E-8},	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	new double[4]{ -0.08, -0.02, 0.04, 0.10 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_U1U2_2.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	28,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	//new double[4]{ -0.02, 0.04, 1.0E-8, 3.5E-8},	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	new double[4]{ 0.04, 0.10, -0.08, -0.02 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_U1U2_3_1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

	//neuronClasterization2D_2(
	//	2,											// const double	tMax,								// Время моделирования системы
	//	200,											// const int		nPts,								// Разрешение диаграммы
	//	1e-7,										// const double	h,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),				// const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	//	init,										// const double* initialConditions,					// Массив с начальными условиями
	////// U1 - C1
	//	//new double[4]{ -0.02, 0.04, 1.0E-8, 3.5E-8},	// const double* ranges,								// Диапазоны изменения параметров
	//	//new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	////// U1 - U2
	//	new double[4]{ -0.02, 0.04, -0.02, 0.04 },	// const double* ranges,								// Диапазоны изменения параметров
	//	new int[2]{ 6,7 },							// const int* indicesOfMutVars,					// Индексы изменяемых параметров
	//	0,											// const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	//	100000,										// const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,											// const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,											// const double* values,								// Параметры
	//	sizeof(a) / sizeof(double),					// const int		amountOfValues,						// Количество параметров
	//	20,											// const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.000001,										// const double	eps,
	//	"D:\\CUDAresults\\neuron2D_U1U2_1.csv"				// std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
	//);

	//bifurcation1D(
	//	0.04,								//const double	tMax,							// Время моделирования системы
	//	1000,								//const int		nPts,							// Разрешение диаграммы
	//	5e-8,								//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),		//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	//	init,								//const double* initialConditions,				// Массив с начальными условиями
	//	new double[2]{ -0.01, 0.02 },			//const double* ranges,							// Диапазон изменения переменной
	//	new int[1]{ 6 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	//	0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,									//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,									//const double* values,							// Параметры
	//	sizeof(a) / sizeof(double),			//const int		amountOfValues,					// Количество параметров
	//	10,									//const int		preScaller,
	//	"D:\\CUDAresults\\bif1D_neuron.csv"//std::string		OUT_FILE_PATH					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//);

		//TimeDomainCalculation(
	//	1,								//const double	tMax,							// Время моделирования системы
	//	70,									//const int		nPts,							// Разрешение диаграммы1
	//	1e-7,								//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),		// const int amountOfInitialConditions,
	//	init,								// const double* initialConditions,
	//	new double[2]{ -0.02, 0.01 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 6 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	//	0,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,								//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,									// const double* values,
	//	sizeof(a) / sizeof(double),			// const int amountOfValues,
	//	5,									//const int		preScaller,
	//	"D:\\CUDAresults\\TimeDomain_neuron.csv"	//std::string		OUT_FILE_PATH
	//);

	//TimeDomainCalculation(
	//	100000,								//const double	tMax,							// Время моделирования системы
	//	1,									//const int		nPts,							// Разрешение диаграммы1
	//	1,								//const double	h,								// Шаг интегрирования
	//	sizeof(init) / sizeof(double),		// const int amountOfInitialConditions,
	//	init,								// const double* initialConditions,
	//	new double[2]{ 0.49, 0.5 },		//const double* ranges,							// Диаппазон изменения переменной
	//	new int[1]{ 0 },					//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	//	1,									//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	//	100000,								//const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	0,								//const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	//	a,									// const double* values,
	//	sizeof(a) / sizeof(double),			// const int amountOfValues,
	//	1,									//const int		preScaller,
	//	"D:\\CUDAresults\\TimeDomain_0.csv"	//std::string		OUT_FILE_PATH
	//);
	
	//a[7] = 0.5;
	//neuronClasterization2D(
	//	0.01, // const double tMax,
	//	50, // const int nPts,
	//	5e-8, // const double h,
	//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//	init, // const double* initialConditions,
	//	new double[4]{ -0.5, 0.5, 1e-5, 20e-4 }, // const double* ranges,
	//	new int[2]{ 6, 25 }, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	100000000, // const double maxValue,
	//	//3600, // const int maxAmountOfPeaks,
	//	0.01, // const double transientTime,
	//	a, // const double* values,
	//	sizeof(a) / sizeof(double), // const int amountOfValues,
	//	50, // const int preScaller,
	//	0.00005, //eps
	//	"D:\\CUDAresults\\Neuron_AND-TS_GI401a_2Dclassification2.csv"
	//);

	//std::string str = "D:\\CUDAresults\\Neuron_AND-TS_GI401a_2Dclassification_";
	//std::string path;
	//int ress = 25;

	//for (int i = 0; i < ress; i++) {

	//	path = str + std::to_string(i) + ".csv";

	//	a[7] = 0 + (double)i*0.02;
	//	neuronClasterization2D(
	//		0.01, // const double tMax,
	//		50, // const int nPts,
	//		5e-8, // const double h,
	//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//		init, // const double* initialConditions,
	//		new double[4]{ -0.25, 0.11, 1e-5, 2.5e-4 }, // const double* ranges,
	//		new int[2]{ 6, 25 }, // const int* indicesOfMutVars,
	//		0, // const int writableVar,
	//		100000000, // const double maxValue,
	//		//3600, // const int maxAmountOfPeaks,
	//		0.01, // const double transientTime,
	//		a, // const double* values,
	//		sizeof(a) / sizeof(double), // const int amountOfValues,
	//		50, // const int preScaller,
	//		0.00005, //eps
	//		path
	//		//"D:\\CUDAresults\\Neuron_AND-TS_GI401a_2Dclassification2.csv"
	//	);
	//}
	//	bifurcation2D(
	//		0.01, // const double tMax,
	//		100, // const int nPts,
	//		5e-8, // const double h,
	//		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//		init, // const double* initialConditions,
	//		new double[4]{ -0.5, 0.5, -0.5, 0.5 }, // const double* ranges,
	//		new int[2]{ 6, 7 }, // const int* indicesOfMutVars,
	//		0, // const int writableVar,
	//		100000000000, // const double maxValue,
	//		0.001, // const double transientTime,
	//		a, // const double* values,
	//		sizeof(a) / sizeof(double), // const int amountOfValues,
	//		20, // const int preScaller,
	//		0.01, //eps
	//		"D:\\CUDAresults\\Neuron_AND-TS_GI403a_2Dbif.csv"
	//);


		//double init[2]{ 0, 0 };
		//double a[21];


	///*  Sym   */ a[0] = 0.5;
	///*  Iin   */ a[1]  = 6.6e-5;
	///*  Ip    */ a[2]  = 6.2e-5;
	///*  Iv    */ a[3]  = 6e-6;
	///*  Is    */ a[4]  = 1.1e-7;
	///*  Ron   */ a[5]  = 2e3;
	///*  Roff  */ a[6]  = 1e6;
	///*  Von1  */ a[7]  = 0.28;
	///*  Voff1 */ a[8]  = 0.14;
	///*  Von2  */ a[9]  = -0.12;
	///*  Voff2 */ a[10] = -0.02;
	///*  U1    */ a[11] = 0;
	///*  U2    */ a[12] = 0.1;
	///*  C1    */ a[13] = 2e-8;
	///*  Tay   */ a[14] = 1e-7;
	///*  T     */ a[15] = 0.5;
	///*  Vt    */ a[16] = double(1) / 17;
	///*  Vp    */ a[17] = 0.039;// 0.037;
	///*  R     */ a[18] = 8.617333262145179e-05; //R = boltz/echarge;
	///*  D     */ a[19] = 20;
	///*  E     */ a[20] = 0.09;

	//bifurcation2D(
	//0.01, // const double tMax,
	//100, // const int nPts,
	//1e-7, // const double h,
	//sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//init, // const double* initialConditions,
	//new double[4]{ 1e-8, 1e-6, 0, 0.2 }, // const double* ranges,
	//new int[2]{ 13, 12 }, // const int* indicesOfMutVars,
	//0, // const int writableVar,
	//100000000000, // const double maxValue,
	//0.01, // const double transientTime,
	//a, // const double* values,
	//sizeof(a) / sizeof(double), // const int amountOfValues,
	//20, // const int preScaller,
	//0.05, //eps
	//"D:\\CUDAresults\\Neuron_ORLOVSKII_001_2Dbif.csv"
	//);



	//double params[14]{ 0.5, 1, 3, 0.5, 0, 2, 8, 1, 1, 0.76, 1, 140, 20, 0 };
	////double params[14]{ 0.5, 1, 3, 0.5, 0, 2, 8, 0.6, 1.18, 0.76, 1, 140, 20, 0 };

	////double params[14]{ 0.5, 1, 3, 0.5, 0, 10, 8, 1, 2, 2, 1, 140, 20, 0 };
	////LEGEND: sym., Ib, l, lam, lam_1, phi, l_e, eta_1, eta_2, G, amp., per., pulse, shift

	//// ********** PARAMS FOR LARGE phi_e, G = 0.5 **********
	////double params[14]{ 0.5, 1, 3, 0.5, 0, 10, 8, 1, 5, 2, 0.5, 140, 20, 0 };
	////double params[14]{ 0.5, 1, 3, 0.5, 0, 10, 8, 1, 10, 4, 0.5, 140, 20, 0 };
	////LEGEND: sym., Ib, l, lam, lam_1, phi, l_e, eta_1, eta_2, G, amp., per., pulse, shift
	//double init[8]{ 0,0,0,0,0,0,0,0 };



	//neuronClasterization2D(
	//	360, // const double tMax,
	//	50, // const int nPts,
	//	0.01, // const double h,
	//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//	init, // const double* initialConditions,

	//	// ************* ETA1-GAMMA ***********
	//	//new double[4] { 0, 10, 0, 10 }, // const double* ranges,
	//	//new int[2] { 7, 9 }, // const int* indicesOfMutVars,

	//	// ************* ETA1-ETA2 ***********
	//	//new double[4] { 0, 2, 0, 2 }, // const double* ranges,
	//	//new int[2] { 7, 8 }, // const int* indicesOfMutVars,

	//	//
	//	// ************* ETA-Ib ***********
	//	//new double[4] { 0, 5, 0, 5 }, // const double* ranges,
	//	//new int[2] { 7, 1 }, // const int* indicesOfMutVars,

	//	// ************* ETA-Iin ***********
	//	//new double[4] { 0, 5, 0, 20 }, // const double* ranges,
	//	//new int[2] { 7, 10 }, // const int* indicesOfMutVars,

	//	// ************* phi-ETA ***********
	//	new double[4]{ 0, 50, 0.1, 20 }, // const double* ranges,
	//	new int[2]{ 5, 7 }, // const int* indicesOfMutVars,

	//	// ************* phi-GAMMA ***********
	//	//new double[4] { 0, 100, 0, 50 }, // const double* ranges,
	//	//new int[2] { 5, 9 }, // const int* indicesOfMutVars,

	//	// ************* phi-lambda *********** DO NOT USE IT, NON-PHYSICAL
	//	//new double[4] { 0, 100, 0.4, 0.6 }, // const double* ranges,
	//	//new int[2] { 5, 3 }, // const int* indicesOfMutVars,

	//	// ************* phi-l ***********
	//	//new double[4] { 0, 200, 0, 200 }, // const double* ranges,
	//	//new int[2] { 5, 2 }, // const int* indicesOfMutVars,

	//	// ************* eta-l ***********
	//	//new double[4] { 0, 10, 0, 100 }, // const double* ranges,
	//	//new int[2] { 7, 2 }, // const int* indicesOfMutVars,

	//	// ************* phi-Ib ***********
	//	//new double[4] { 0, 200, 0, 10 }, // const double* ranges,
	//	//new int[2] { 5, 1 }, // const int* indicesOfMutVars,

	//	7, // const int writableVar,
	//	100000000, // const double maxValue,
	//	//3600, // const int maxAmountOfPeaks,
	//	360, // const double transientTime,
	//	params, // const double* values,
	//	sizeof(params) / sizeof(double), // const int amountOfValues,
	//	1, // const int preScaller,
	//	0.01, //eps
	//	"D:\\CUDAresults\\3JJti_2Dbif_01.csv"
	//	//"C:\\Users\\elektroraum\\YandexDisk\\LETI2\\2024\\cuda_projects\\3JJti_2Dbif_01.csv"
	//);





	//double params[7]{ 0.5, 22.34, 2.75, 13, -0.885, 0.1, 0.025 };
	//double init[5]{ 0, 0.00001, 0, 0, 0 };
	//double params[3]{ 0.5, 3, 3 };
	//double init[3]{ 0, 0, 1E-6 };


	//basinsOfAttraction_2(
	//	200,									// Время моделирования системы
	//	100,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ -12, 12, -12, 12 },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	1,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	500,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	5,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_SineSC4_1.csv");

	//	double params[4]{ 0.7736, 40, 3,28 };
	//	double init[3]{ 0, 0, 18 };

	//basinsOfAttraction_2(
	//	1000,									// Время моделирования системы
	//	200,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ -16, 16, -16, 16 },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	500,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.004,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Сhen_1.csv");

	//double params[3]{ 0.5, 0.065, 0.45 };
	//double init[3]{ 0, 0, 0 };

	//basinsOfAttraction_2(
	//	500,									// Время моделирования системы
	//	500,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ -30, 5, -5, 7 },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	500,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	5,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Gokyl_1.csv");


//	double params[3]{ 0.5, 0.1,0.7 }; //0.210028, 0.190272
////	double init[3]{ 1.5,3,6 };
//	double init[3]{ 0.001,0,0 };
//	params[2] = 1.25;

	//bifurcation2D(
	//	300, // const double tMax,
	//	300, // const int nPts,
	//	0.01, // const double h,
	//	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	//	init, // const double* initialConditions,
	//	new double[4]{ -1000, 16000, 0, 0.17 }, // const double* ranges,
	//	new int[2]{ 0, 1 }, // const int* indicesOfMutVars,
	//	1, // const int writableVar,
	//	100000000, // const double maxValue,
	//	4000, // const double transientTime,
	//	params, // const double* values,
	//	sizeof(params) / sizeof(double), // const int amountOfValues,
	//	1, // const int preScaller,
	//	0.05, //eps
	//	"D:\\CUDAresults\\RKalpha_BabkinPaper_alpha_vs_a_var_b_1.25_1.csv"
	//);

	//double params[4]{ 0.5, 0.29, 0.14, 4.52}; 
	//double init[3]{ 0.1, 0.1, 15.55517 };
	//double ranges[4]{ -14, 24, -16, 12 };

	//double params[4]{ 0.7719, 40, 3, 28 };
	//double init[3]{ 0.1, 0.1, 18 };

	//basinsOfAttraction(
	//1000,									// Время моделирования системы
	//200,									// Разрешение диаграммы
	//0.01,									// Шаг интегрирования
	//sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//init,									// Массив с начальными условиями
	//new double[4]{ -16, 16, -16, 16 },	// Диапазоны изменения параметров
	//new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//0,										// Индекс уравнения, по которому будем строить диаграмму
	//100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//300,									// Время, которое будет промоделировано перед расчетом диаграммы
	//params,									// Параметры
	//sizeof(params) / sizeof(double),		// Количество параметров
	//1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//0.05,									// Эпсилон для алгоритма DBSCAN
	//"D:\\CUDAresults\\Basins_Chen_01.csv");

	//basinsOfAttraction(
	//	300,									// Время моделирования системы
	//	250,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ ranges[0], ranges[1]/2, ranges[2], ranges[3]/2 },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	300,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Rossler_01.csv");

	//basinsOfAttraction(
	//	300,									// Время моделирования системы
	//	250,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ ranges[0]/2, ranges[1], ranges[2], ranges[3] / 2 },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	300,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Rossler_02.csv");

	//basinsOfAttraction(
	//	300,									// Время моделирования системы
	//	250,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ ranges[0], ranges[1] / 2, ranges[2]/2, ranges[3] },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	300,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Rossler_03.csv");

	//basinsOfAttraction(
	//	300,									// Время моделирования системы
	//	250,									// Разрешение диаграммы
	//	0.01,									// Шаг интегрирования
	//	sizeof(init) / sizeof(double),			// Количество начальных условий ( уравнений в системе )
	//	init,									// Массив с начальными условиями
	//	new double[4]{ ranges[0] / 2, ranges[1], ranges[2]/2, ranges[3] },	// Диапазоны изменения параметров
	//	new int[2]{ 0, 1 },						// Индексы изменяемых параметров
	//	0,										// Индекс уравнения, по которому будем строить диаграмму
	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	//	300,									// Время, которое будет промоделировано перед расчетом диаграммы
	//	params,									// Параметры
	//	sizeof(params) / sizeof(double),		// Количество параметров
	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	//	0.05,									// Эпсилон для алгоритма DBSCAN
	//	"D:\\CUDAresults\\Basins_Rossler_04.csv");



	
	printf(" --- Time of runnig: %zu ms", std::clock() - startTime);
    return 0;
}