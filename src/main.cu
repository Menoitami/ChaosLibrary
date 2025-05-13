#include "bifurcationHOST.h"
#include "basinsHOST.h"
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
#include <string>
#include <fstream>
#include <vector>
#include <systems.cuh>
#include <map>
#include <sstream>
//using namespace old_library;


//#define USE_LORENZ_MODEL

// Структура для хранения результатов тестов

void runPerformanceTests() {
	std::ofstream resultsFile("performance_results.csv");
	if (!resultsFile) {
		std::cerr << "Ошибка: Не удалось открыть файл для записи результатов!" << std::endl;
		return;
	}
	
	// Заголовок CSV файла
	resultsFile << "Library,Resolution,ModelingTime,ExecutionTime_ms" << std::endl;
	
	// Параметры для тестов (уменьшаем количество для ускорения)
	std::vector<double> resolutions = {100, 300, 500};
	std::vector<double> modelingTimeTrans = {1000, 3000, 5000};
	std::vector<int> modelingTimes = {1000, 3000, 5000};
	
	double params[5]{ 0.5, 0.1, 1.4, 15.552, 2 };
	double init[3]{ 0, 0, 0 };
	double ranges[4]{ -6, 6, -6, 6 };
	int indicesOfMutVars[2]{ 0, 1 };
	
	// Тесты для Basins::basinsOfAttraction_2
	std::cout << "Starting tests for Basins::basinsOfAttraction_2..." << std::endl;
	for (int resolution : resolutions) {
		for (int modelingTime : modelingTimes) {
			std::cout << "Testing Basins::basinsOfAttraction_2 with resolution=" << resolution 
					  << ", modelingTime=" << modelingTime << std::endl;
			
			auto start = std::chrono::high_resolution_clock::now();
			
			try {
				Basins::basinsOfAttraction_2(
					500,    // Время моделирования системы
					resolution,      // Разрешение диаграммы
					0.01,           // Шаг интегрирования
					sizeof(init) / sizeof(double),   // Количество начальных условий
					init,           // Массив с начальными условиями
					ranges,
					indicesOfMutVars,
					1,              // Индекс уравнения для диаграммы
					100000000,      // Максимальное значение
					modelingTime,           // Время для промоделирования
					params,         // Параметры
					sizeof(params) / sizeof(double),  // Количество параметров
					1,              // Множитель
					0.05,           // Эпсилон для DBSCAN
					std::string(BASINS_OUTPUT_PATH) + "/basins_perf_test_" + 
						std::to_string(resolution) + "_" + std::to_string(modelingTime) + ".csv"
				);
				
				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				
				// Запись результатов в CSV
				resultsFile << "Basins," << resolution << "," << modelingTime << "," << duration << std::endl;
				resultsFile.flush(); // Сбрасываем буфер после каждого теста
				
				std::cout << "  Time taken: " << duration << " milliseconds" << std::endl;
			}
			catch (const std::exception& e) {
				std::cerr << "Error in Basins::basinsOfAttraction_2: " << e.what() << std::endl;
				resultsFile << "Basins," << resolution << "," << modelingTime << ",ERROR" << std::endl;
			}
		}
	}
	
	// Тесты для old_library::basinsOfAttraction_2
	std::cout << "\nStarting tests for old_library::basinsOfAttraction_2..." << std::endl;
	for (int resolution : resolutions) {
		for (int modelingTime : modelingTimes) {
			std::cout << "Testing old_library::basinsOfAttraction_2 with resolution=" << resolution 
					  << ", modelingTime=" << modelingTime << std::endl;
			
			auto start = std::chrono::high_resolution_clock::now();
			
			try {
				old_library::basinsOfAttraction_2(
					500,    // Время моделирования системы
					resolution,      // Разрешение диаграммы
					0.01,           // Шаг интегрирования
					sizeof(init) / sizeof(double),   // Количество начальных условий
					init,           // Массив с начальными условиями
					ranges,
					indicesOfMutVars,
					1,              // Индекс уравнения для диаграммы
					100000000,      // Максимальное значение
					modelingTime,           // Время для промоделирования
					params,         // Параметры
					sizeof(params) / sizeof(double),  // Количество параметров
					1,              // Множитель
					0.05,           // Эпсилон для DBSCAN
					std::string(BASINS_OUTPUT_PATH) + "/old_basins_perf_test_" + 
						std::to_string(resolution) + "_" + std::to_string(modelingTime) + ".csv"
				);
				
				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				
				// Запись результатов в CSV
				resultsFile << "old_library," << resolution << "," << modelingTime << "," << duration << std::endl;
				resultsFile.flush(); // Сбрасываем буфер после каждого теста
				
				std::cout << "  Time taken: " << duration << " milliseconds" << std::endl;
			}
			catch (const std::exception& e) {
				std::cerr << "Error in old_library::basinsOfAttraction_2: " << e.what() << std::endl;
				resultsFile << "old_library," << resolution << "," << modelingTime << ",ERROR" << std::endl;
			}
		}
	}
	
	resultsFile.close();
	std::cout << "Performance tests completed. Results saved to performance_results.csv" << std::endl;
}

int main()
{
	size_t startTime = std::clock();
	double h = (double)0.01;

#ifdef USE_CHAMELEON_MODEL
	double a[7]{ 0.5, 3, 0, 2, 1, 18, 1 };
	double init[3]{ 3, 3, 0 };
	double ranges[4]{ 0, 5, 0, 20 };
	int indicesOfMutVars[2]{ 3, 5 };
	int writableVar = 0;
	double maxValue = 10000;
	double eps_bif = 0.001;
	double eps_lle = 1e-6;
	int preScaller = 1;
	double NT_lle = 0.5;
#endif

#ifdef USE_ROSSLER_MODEL
	double a[4]{ 0, 0.2, 0.2, 5.7};
	double init[3]{ 0.01, 0 , 0 };
	double ranges[4]{ -0, 0.6, 0, 0.6 };
	int indicesOfMutVars[2]{ 1, 2 };
	int writableVar = 0;
	double maxValue = 10000;
	double eps_bif = 0.001;
	double eps_lle = 1e-6;
	int preScaller = 1;
	double NT_lle = 0.5;
#endif

#ifdef USE_SYSTEM_FOR_BASINS
	double params[5]{ 0.5, 0.1665, 1.4,  15.552, 2 };
	double init[3]{ 0, 0, 0,};
	double ranges[4]{ -6, 6, -6, 6};
	int indicesOfMutVars[2]{ 0, 1 };
	int writableVar = 0;
	double maxValue = 10000;
	double eps_bif = 0.001;
	double eps_lle = 1e-6;
	int preScaller = 1;
	double NT_lle = 0.5;

	{
		// //double h = 0.01;
		
		// std::cout << "Start basins" << std::endl;
		// auto start = std::chrono::high_resolution_clock::now();
		
		// Basins::basinsOfAttraction_2(
		// 	200,         // Время моделирования системы
		// 	500,         // Разрешение диаграммы
		// 	0.01,         // Шаг интегрирования
		// 	sizeof(init) / sizeof(double),   // Количество начальных условий ( уравнений в системе )
		// 	init,         // Массив с начальными условиями
		// 	ranges,
		// 	indicesOfMutVars,
		// 	1,          // Индекс уравнения, по которому будем строить диаграмму
		// 	100000000,        // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		// 	2000,         // Время, которое будет промоделировано перед расчетом диаграммы
		// 	params,         // Параметры
		// 	sizeof(params) / sizeof(double),  // Количество параметров
		// 	1,          // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		// 	0.05,         // Эпсилон для алгоритма DBSCAN
		// 	std::string(BASINS_OUTPUT_PATH) + "/basinsOfAttraction_system_for_basins_graph.csv"
		// );
		// auto end = std::chrono::high_resolution_clock::now();
		// auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		// std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
	}
	runPerformanceTests();
#endif

#ifdef USE_SYSTEM_FOR_BASINS_2
	double params[5]{ 0.5, 0.1, 1.4,  15.552, 2 };
	double init[3]{ 0, 0, 0,};
	
	{
		//double h = 0.01;
		
		std::cout << "Start basins" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		
		Basins::basinsOfAttraction_2(
			200,         // Время моделирования системы
			500,         // Разрешение диаграммы
			0.01,         // Шаг интегрирования
			sizeof(init) / sizeof(double),   // Количество начальных условий ( уравнений в системе )
			init,         // Массив с начальными условиями
			new double[4]{ -200, 200, -60, 60},
			new int[2] { 0, 1 },      // Индексы изменяемых параметров
			1,          // Индекс уравнения, по которому будем строить диаграмму
			100000000,        // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
			2000,         // Время, которое будет промоделировано перед расчетом диаграммы
			params,         // Параметры
			sizeof(params) / sizeof(double),  // Количество параметров
			1,          // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
			0.05,         // Эпсилон для алгоритма DBSCAN
			std::string(BASINS_OUTPUT_PATH) + "/basinsOfAttraction_system_for_basins_graph_2.csv"
		);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
	}
#endif

	{
		// double params_old[5]{ 0.5, 0.1665, 1.4,  15.552, 2 };
 		// double init_old[3]{ 0, 0, 0,};

		// std::cout << "Start old_library basins" << std::endl;
		// auto start_old = std::chrono::high_resolution_clock::now();

		// old_library::basinsOfAttraction_2(
		// 	500,         // Время моделирования системы
		// 	500,         // Разрешение диаграммы
		// 	0.01,         // Шаг интегрирования
		// 	sizeof(init_old) / sizeof(double),   // Количество начальных условий ( уравнений в системе )
		// 	init_old,         // Массив с начальными условиями
		// 	new double[4]{ -6, 6, -6, 6},
		// 	new int[2] { 0, 1 },      // Индексы изменяемых параметров
		// 	1,          // Индекс уравнения, по которому будем строить диаграмму
		// 	100000000,        // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		// 	1000,         // Время, которое будет промоделировано перед расчетом диаграммы
		// 	params_old,         // Параметры
		// 	sizeof(params_old) / sizeof(double),  // Количество параметров
		// 	1,          // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
		// 	0.05,         // Эпсилон для алгоритма DBSCAN
		// 	std::string(BASINS_OUTPUT_PATH) + "/basinsOfAttraction_test_old.csv"
		// );

		// auto end_old = std::chrono::high_resolution_clock::now();
		// auto duration_old = std::chrono::duration_cast<std::chrono::milliseconds>(end_old - start_old).count();
		// std::cout << "Time taken: " << duration_old << " milliseconds" << std::endl;
	}



	//auto start = std::chrono::high_resolution_clock::now();
	// // --- Bifurcation: Легкий запуск ---

	{
		// {
		// 	double CT = 1000;
		// 	int nPts = 10000;
		// 	double TT = 10000;
			
		// 	auto start = std::chrono::high_resolution_clock::now();
		// 	Bifurcation::bifurcation1D(
		// 		CT, nPts, h,
		// 		sizeof(init) / sizeof(double), init,
		// 		ranges, indicesOfMutVars, writableVar,
		// 		maxValue, TT,
		// 		a, sizeof(a) / sizeof(double),
		// 		preScaller,
		// 		std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_Graph_chameleon.csv"
		// 	);
		// 	auto end = std::chrono::high_resolution_clock::now();
		// 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		// 	std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_Graph_chameleon_timing.txt";
		// 	std::ofstream outFile(timingFileName);
		// 	if (outFile) {
		// 		outFile << duration;
		// 		outFile.close();
		// 		std::cout << "Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
		// 	} else {
		// 		std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Test)!" << std::endl;
		// 	}
		// }
	}

	// {
	// 	double CT = 400;
	// 	int nPts = 500;
	// 	double TT = 2000;
	// 	auto start = std::chrono::high_resolution_clock::now();
	// 	Bifurcation::bifurcation2D(
	// 		CT, nPts, h,
	// 		sizeof(init) / sizeof(double), init,
	// 		ranges, indicesOfMutVars, writableVar,
	// 		maxValue, TT,
	// 		a, sizeof(a) / sizeof(double),
	// 		preScaller, eps_bif,
	// 		std::string(BIFURCATION_OUTPUT_PATH) + "/Bfurcation_graph_2D_rossler.csv" // Другой файл для результатов
	// 	);
	// 	auto end = std::chrono::high_resolution_clock::now();
	// 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 	std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bfurcation_graph_2D_rossler_timing.txt"; // Другой файл для времени
	// 	std::ofstream outFile(timingFileName);
	// 	if (outFile) {
	// 		outFile << duration;
	// 		outFile.close();
	// 		std::cout << "old_library::Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 	} else {
	// 		std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Test)!" << std::endl;
	// 	}
	// }
	



// {
// 	//LLe
// 	{
// 		double CT = 400;
// 		int nPts = 500;
// 		double TT = 2000;
// 		auto start = std::chrono::high_resolution_clock::now();
// 		LLE::LLE2D(
// 			CT, NT_lle, nPts, h, eps_lle,
// 			init, sizeof(init) / sizeof(double), 
// 			ranges, indicesOfMutVars, writableVar,
// 			maxValue, TT,
// 			a, sizeof(a) / sizeof(double),
// 			std::string(LLE_OUTPUT_PATH) + "/LLE_2D_graph_chameleon.csv"
// 		);	
// 		auto end = std::chrono::high_resolution_clock::now();
// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// 		std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_2D_graph_chameleon_timing.txt";
// 		std::ofstream outFile(timingFileName);
// 		if (outFile) {
// 			outFile << duration;
// 			outFile.close();
// 			std::cout << "LLE (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
// 		}
// 		else {
// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени LLE (Test)!" << std::endl;
// 		}	
// 	}
// }




	// Bifurcation 1D
// --- Bifurcation: Легкий запуск ---
	// {
	// 	double CT = 500;
	// 	int nPts = 1000;
	// 	double TT = 2000;

	// 	// --- Новый Bifurcation ---
	// 	{

	// 		Bifurcation::bifurcation1D(
	// 			CT, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, TT,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation1D_test.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation1D_test_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Test)!" << std::endl;
	// 		}
	// 	}

	// 	// --- Старый Bifurcation (old_library) ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		old_library::bifurcation1D(
	// 			CT, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, TT,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation1D_test_old.csv" // Другой файл для результатов
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation1D_test_old_timing.txt"; // Другой файл для времени
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "old_library::Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Test)!" << std::endl;
	// 		}
	// 	}






	// {
	// 	double CT = 100;
	// 	int nPts = 100;
	// 	double TT = 1000;

	// 	// --- Новый Bifurcation ---

	// 	{
	// 		old_library::basinsOfAttraction_2(
	// 			CT, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, TT,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller, eps_bif,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/basinsOfAttraction_test.csv"
	// 		);
	// 	}
	//  }
	// Запуск тестов производительности
	

	std::cout << "Общее время выполнения: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	return 0;
} 