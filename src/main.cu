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

void runBasinsPerformanceTests() {
	std::ofstream resultsFile("performance_results.csv");
	if (!resultsFile) {
		std::cerr << "Ошибка: Не удалось открыть файл для записи результатов!" << std::endl;
		return;
	}
	
	// Заголовок CSV файла
	resultsFile << "Test,Parameter,Value,Library,ExecutionTime_ms" << std::endl;
	
	// Параметры для тестов
	std::vector<int> resolutions = {100, 300, 500};  // Тест по разрешению
	std::vector<int> modelingTimes = {1000, 3000, 5000};  // Тест по времени моделирования
	
	double params[5]{ 0.5, 0.1, 1.4, 15.552, 2 };
	double init[3]{ 0, 0, 0 };
	double ranges[4]{ -6, 6, -6, 6 };
	int indicesOfMutVars[2]{ 0, 1 };
	
	// Тест 1: Влияние разрешения (фиксированное время моделирования)
	std::cout << "\n===== Тест 1: Влияние разрешения на время выполнения =====\n";
	int fixedModelingTime = 4000; // Фиксированное время моделирования
	
	for (int resolution : resolutions) {
		std::cout << "\nТестирование с разрешением = " << resolution << std::endl;
		
		// Запуск Basins::basinsOfAttraction_2
		std::cout << "  Запуск Basins::basinsOfAttraction_2..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			Basins::basinsOfAttraction_2(
				500,                // Время моделирования системы
				resolution,         // Разрешение диаграммы
				0.01,               // Шаг интегрирования
				sizeof(init) / sizeof(double),   // Количество начальных условий
				init,               // Массив с начальными условиями
				ranges,
				indicesOfMutVars,
				1,                  // Индекс уравнения для диаграммы
				100000000,          // Максимальное значение
				fixedModelingTime,  // Время для промоделирования
				params,             // Параметры
				sizeof(params) / sizeof(double),  // Количество параметров
				1,                  // Множитель
				0.05,               // Эпсилон для DBSCAN
				std::string(BASINS_OUTPUT_PATH) + "/basins_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "Resolution," << resolution << ",Basins," << duration1 << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",Basins,ERROR" << std::endl;
		}
		
		// Запуск old_library::basinsOfAttraction_2
		std::cout << "  Запуск old_library::basinsOfAttraction_2..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::basinsOfAttraction_2(
				500,                // Время моделирования системы
				resolution,         // Разрешение диаграммы
				0.01,               // Шаг интегрирования
				sizeof(init) / sizeof(double),   // Количество начальных условий
				init,               // Массив с начальными условиями
				ranges,
				indicesOfMutVars,
				1,                  // Индекс уравнения для диаграммы
				100000000,          // Максимальное значение
				fixedModelingTime,  // Время для промоделирования
				params,             // Параметры
				sizeof(params) / sizeof(double),  // Количество параметров
				1,                  // Множитель
				0.05,               // Эпсилон для DBSCAN
				std::string(BASINS_OUTPUT_PATH) + "/old_basins_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "Resolution," << resolution << ",old_library," << duration2 << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration2 / duration1;
				std::cout << "    Ускорение: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",old_library,ERROR" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	// Тест 2: Влияние времени моделирования (фиксированное разрешение)
	std::cout << "\n===== Тест 2: Влияние времени моделирования на время выполнения =====\n";
	int fixedResolution = 300; // Фиксированное разрешение
	
	for (int modelingTime : modelingTimes) {
		std::cout << "\nТестирование с временем моделирования = " << modelingTime << std::endl;
		
		// Запуск Basins::basinsOfAttraction_2
		std::cout << "  Запуск Basins::basinsOfAttraction_2..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			Basins::basinsOfAttraction_2(
				500,                // Время моделирования системы
				fixedResolution,    // Разрешение диаграммы
				0.01,               // Шаг интегрирования
				sizeof(init) / sizeof(double),   // Количество начальных условий
				init,               // Массив с начальными условиями
				ranges,
				indicesOfMutVars,
				1,                  // Индекс уравнения для диаграммы
				100000000,          // Максимальное значение
				modelingTime,       // Время для промоделирования
				params,             // Параметры
				sizeof(params) / sizeof(double),  // Количество параметров
				1,                  // Множитель
				0.05,               // Эпсилон для DBSCAN
				std::string(BASINS_OUTPUT_PATH) + "/basins_time_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",Basins," << duration1 << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",Basins,ERROR" << std::endl;
		}
		
		// Запуск old_library::basinsOfAttraction_2
		std::cout << "  Запуск old_library::basinsOfAttraction_2..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::basinsOfAttraction_2(
				500,                // Время моделирования системы
				fixedResolution,    // Разрешение диаграммы
				0.01,               // Шаг интегрирования
				sizeof(init) / sizeof(double),   // Количество начальных условий
				init,               // Массив с начальными условиями
				ranges,
				indicesOfMutVars,
				1,                  // Индекс уравнения для диаграммы
				100000000,          // Максимальное значение
				modelingTime,       // Время для промоделирования
				params,             // Параметры
				sizeof(params) / sizeof(double),  // Количество параметров
				1,                  // Множитель
				0.05,               // Эпсилон для DBSCAN
				std::string(BASINS_OUTPUT_PATH) + "/old_basins_time_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",old_library," << duration2 << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration2 / duration1;
				std::cout << "    Ускорение: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",old_library,ERROR" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	resultsFile.close();
	std::cout << "\nТесты производительности завершены. Результаты сохранены в performance_results.csv" << std::endl;
}

void runLLEPerformanceTests() {
	std::ofstream resultsFile("lle_performance_results.csv");
	if (!resultsFile) {
		std::cerr << "Ошибка: Не удалось открыть файл для записи результатов LLE!" << std::endl;
		return;
	}
	
	// Заголовок CSV файла
	resultsFile << "Test,Parameter,Value,Library,ExecutionTime_ms" << std::endl;
	
	// Параметры для тестов
	std::vector<int> resolutions = {100, 300, 500};  // Тест по разрешению
	std::vector<double> ntValues = {0.3, 0.5, 0.7};  // Тест по значению NT
	
	double h = 0.01;  // Шаг интегрирования
	double eps_lle = 1e-6;  // Эпсилон для LLE
	
	// Используем параметры из модели Chameleon
	double params[7]{ 0.5, 3, 0, 2, 1, 18, 1 };
	double init[3]{ 3, 3, 0 };
	double ranges[4]{ 0, 5, 0, 20 };
	int indicesOfMutVars[2]{ 3, 5 };
	int writableVar = 0;
	double maxValue = 10000;
	
	// Тест 1: Влияние разрешения (фиксированные остальные параметры)
	std::cout << "\n===== Тест 1: Влияние разрешения на время выполнения LLE =====\n";
	double fixedNT = 0.5;  // Фиксированное значение NT
	double fixedCT = 400;  // Фиксированное значение CT
	double fixedTT = 2000;  // Фиксированное время моделирования
	
	for (int resolution : resolutions) {
		std::cout << "\nТестирование с разрешением = " << resolution << std::endl;
		
		// Запуск LLE::LLE2D
		std::cout << "  Запуск LLE::LLE2D..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			LLE::LLE2D(
				fixedCT, fixedNT, resolution, h, eps_lle,
				init, sizeof(init) / sizeof(double), 
				ranges, indicesOfMutVars, writableVar,
				maxValue, fixedTT,
				params, sizeof(params) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "Resolution," << resolution << ",LLE," << duration1 << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",LLE,ERROR" << std::endl;
		}
		
		// Запуск CUDA версии LLE, если она доступна
		std::cout << "  Запуск old_library::LLE2D..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::LLE2D(
				fixedCT, fixedNT, resolution, h, eps_lle,
				init, sizeof(init) / sizeof(double), 
				ranges, indicesOfMutVars, writableVar,
				maxValue, fixedTT,
				params, sizeof(params) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_cuda_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "Resolution," << resolution << ",old_library," << duration2 << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration1 / duration2;  // CUDA должен быть быстрее, поэтому меняем порядок
				std::cout << "    Ускорение CUDA: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",old_library,ERROR" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	// // Тест 2: Влияние значения NT (фиксированное разрешение)
	// std::cout << "\n===== Тест 2: Влияние значения NT на время выполнения LLE =====\n";
	// 
	
	// for (double ntValue : ntValues) {
	// 	std::cout << "\nТестирование с NT = " << ntValue << std::endl;
		
	// 	// Запуск LLE::LLE2D
	// 	std::cout << "  Запуск LLE::LLE2D..." << std::endl;
	// 	auto start1 = std::chrono::high_resolution_clock::now();
	// 	long long duration1 = 0;
		
	// 	try {
	// 		LLE::LLE2D(
	// 			fixedCT, ntValue, fixedResolution, h, eps_lle,
	// 			init, sizeof(init) / sizeof(double), 
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, fixedTT,
	// 			params, sizeof(params) / sizeof(double),
	// 			std::string(LLE_OUTPUT_PATH) + "/lle_nt_test_" + std::to_string(ntValue).substr(0, 3) + ".csv"
	// 		);
			
	// 		auto end1 = std::chrono::high_resolution_clock::now();
	// 		duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
	// 		resultsFile << "NT," << ntValue << ",LLE," << duration1 << std::endl;
	// 		std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
	// 	}
	// 	catch (const std::exception& e) {
	// 		std::cerr << "    Ошибка: " << e.what() << std::endl;
	// 		resultsFile << "NT," << ntValue << ",LLE,ERROR" << std::endl;
	// 	}
		
	// 	// Запуск CUDA версии LLE
	// 	std::cout << "  Запуск old_library::LLE2D..." << std::endl;
	// 	auto start2 = std::chrono::high_resolution_clock::now();
		
	// 	try {
	// 		old_library::LLE2D(
	// 			fixedCT, ntValue, fixedResolution, h, eps_lle,
	// 			init, sizeof(init) / sizeof(double), 
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, fixedTT,
	// 			params, sizeof(params) / sizeof(double),
	// 			std::string(LLE_OUTPUT_PATH) + "/lle_cuda_nt_test_" + std::to_string(ntValue).substr(0, 3) + ".csv"
	// 		);
			
	// 		auto end2 = std::chrono::high_resolution_clock::now();
	// 		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
	// 		resultsFile << "NT," << ntValue << ",old_library," << duration2 << std::endl;
	// 		std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
	// 		// Сравнение производительности
	// 		if (duration1 > 0) {
	// 			double speedup = (double)duration1 / duration2;
	// 			std::cout << "    Ускорение CUDA: " << speedup << "x" << std::endl;
	// 		}
	// 	}
	// 	catch (const std::exception& e) {
	// 		std::cerr << "    Ошибка: " << e.what() << std::endl;
	// 		resultsFile << "NT," << ntValue << ",old_library,ERROR" << std::endl;
	// 	}
		
	// 	resultsFile.flush();
	// }
	int fixedResolution = 300; // Фиксированное разрешение
	// Тест 3: Влияние времени моделирования (TT)
	std::cout << "\n===== Тест 3: Влияние времени моделирования на время выполнения LLE =====\n";
	std::vector<int> modelingTimes = {1000, 2000, 3000, 5000};  // Тест по времени моделирования
	
	for (int modelingTime : modelingTimes) {
		std::cout << "\nТестирование с временем моделирования = " << modelingTime << std::endl;
		
		// Запуск LLE::LLE2D
		std::cout << "  Запуск LLE::LLE2D..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			LLE::LLE2D(
				fixedCT, fixedNT, fixedResolution, h, eps_lle,
				init, sizeof(init) / sizeof(double), 
				ranges, indicesOfMutVars, writableVar,
				maxValue, modelingTime,
				params, sizeof(params) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_tt_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",LLE," << duration1 << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",LLE,ERROR" << std::endl;
		}
		
		// Запуск CUDA версии LLE
		std::cout << "  Запуск old_library::LLE2D..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::LLE2D(
				fixedCT, fixedNT, fixedResolution, h, eps_lle,
				init, sizeof(init) / sizeof(double), 
				ranges, indicesOfMutVars, writableVar,
				maxValue, modelingTime,
				params, sizeof(params) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_cuda_tt_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",old_library," << duration2 << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration2 / duration1;
				std::cout << "    Ускорение CUDA: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",old_library,ERROR" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	resultsFile.close();
	std::cout << "\nТесты производительности LLE завершены. Результаты сохранены в lle_performance_results.csv" << std::endl;
}

void runBifurcationPerformanceTests() {
	std::ofstream resultsFile("bifurcation_performance_results.csv");
	if (!resultsFile) {
		std::cerr << "Ошибка: Не удалось открыть файл для записи результатов Bifurcation!" << std::endl;
		return;
	}
	
	// Заголовок CSV файла
	resultsFile << "Test,Parameter,Value,Library,ExecutionTime_ms,Dimension" << std::endl;
	
	// Параметры для тестов
	std::vector<int> resolutions = {100, 300, 500};  // Тест по разрешению (nPts)
	std::vector<int> modelingTimes = {1000, 2000, 3000,5000};  // Тест по времени моделирования (TT)
	
	double h = 0.01;  // Шаг интегрирования
	
	// Используем параметры из модели Chameleon
	double params[7]{ 0.5, 3, 0, 2, 1, 18, 1 };
	double init[3]{ 3, 3, 0 };
	double ranges[4]{ 0, 5, 0, 20 };
	int indicesOfMutVars[2]{ 3, 5 };
	int writableVar = 0;
	double maxValue = 10000;
	int preScaller = 1;
	double eps_bif = 0.001;
	
	// Тест 1: Влияние разрешения на Bifurcation 1D (фиксированные остальные параметры)
	std::cout << "\n===== Тест 1: Влияние разрешения на время выполнения Bifurcation 1D =====\n";
	double fixedCT = 400;  // Фиксированное значение CT
	double fixedTT = 2000;  // Фиксированное время моделирования
	
	
	// Тест 2: Влияние времени моделирования на Bifurcation 1D (фиксированное разрешение)
	std::cout << "\n===== Тест 2: Влияние времени моделирования на время выполнения Bifurcation 1D =====\n";
	int fixedResolution = 300; // Фиксированное разрешение
	
	
	// Тест 3: Влияние разрешения на Bifurcation 2D (фиксированные остальные параметры)
	std::cout << "\n===== Тест 3: Влияние разрешения на время выполнения Bifurcation 2D =====\n";
	
	for (int resolution : resolutions) {
		std::cout << "\nТестирование с разрешением = " << resolution << std::endl;
		
		// Запуск Bifurcation::bifurcation2D
		std::cout << "  Запуск Bifurcation::bifurcation2D..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			Bifurcation::bifurcation2D(
				fixedCT, resolution, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, fixedTT,
				params, sizeof(params) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/perf_tests/bif2d_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "Resolution," << resolution << ",Bifurcation," << duration1 << ",2D" << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",Bifurcation,ERROR,2D" << std::endl;
		}
		
		// Запуск old_library::bifurcation2D
		std::cout << "  Запуск old_library::bifurcation2D..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::bifurcation2D(
				fixedCT, resolution, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, fixedTT,
				params, sizeof(params) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/perf_tests/bif2d_old_res_test_" + std::to_string(resolution) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "Resolution," << resolution << ",old_library," << duration2 << ",2D" << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration2 / duration1;
				std::cout << "    Ускорение: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "Resolution," << resolution << ",old_library,ERROR,2D" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	// Тест 4: Влияние времени моделирования на Bifurcation 2D (фиксированное разрешение)
	std::cout << "\n===== Тест 4: Влияние времени моделирования на время выполнения Bifurcation 2D =====\n";
	
	for (int modelingTime : modelingTimes) {
		std::cout << "\nТестирование с временем моделирования = " << modelingTime << std::endl;
		
		// Запуск Bifurcation::bifurcation2D
		std::cout << "  Запуск Bifurcation::bifurcation2D..." << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		long long duration1 = 0;
		
		try {
			Bifurcation::bifurcation2D(
				fixedCT, fixedResolution, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, modelingTime,
				params, sizeof(params) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/perf_tests/bif2d_tt_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end1 = std::chrono::high_resolution_clock::now();
			duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",Bifurcation," << duration1 << ",2D" << std::endl;
			std::cout << "    Время выполнения: " << duration1 << " мс" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",Bifurcation,ERROR,2D" << std::endl;
		}
		
		// Запуск old_library::bifurcation2D
		std::cout << "  Запуск old_library::bifurcation2D..." << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		
		try {
			old_library::bifurcation2D(
				fixedCT, fixedResolution, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, modelingTime,
				params, sizeof(params) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/perf_tests/bif2d_old_tt_test_" + std::to_string(modelingTime) + ".csv"
			);
			
			auto end2 = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
			
			resultsFile << "ModelingTime," << modelingTime << ",old_library," << duration2 << ",2D" << std::endl;
			std::cout << "    Время выполнения: " << duration2 << " мс" << std::endl;
			
			// Сравнение производительности
			if (duration1 > 0) {
				double speedup = (double)duration2 / duration1;
				std::cout << "    Ускорение: " << speedup << "x" << std::endl;
			}
		}
		catch (const std::exception& e) {
			std::cerr << "    Ошибка: " << e.what() << std::endl;
			resultsFile << "ModelingTime," << modelingTime << ",old_library,ERROR,2D" << std::endl;
		}
		
		resultsFile.flush();
	}
	
	resultsFile.close();
	std::cout << "\nТесты производительности Bifurcation завершены. Результаты сохранены в bifurcation_performance_results.csv" << std::endl;
}

int main()
{
	size_t startTime = std::clock();
	double h = (double)0.01;

#ifdef USE_DAMIR_SYSTEM


	double CT = 1000;
	double NT = 0.5;
	double resolution = 100;
	double TT = 5000;

	double a[2] {1,1};
	double init[4] {0,0,0,0};
	double ranges[4] {0,2e-6,0,2000};
	int indicesOfMutVars[2] {0,1};
	int writableVar = 0;
	double maxValue = 10000;
	double eps_bif = 0.001;
	double eps_lle = 1e-6;

	LLE::LLE2D(
		CT, NT, resolution, h, eps_lle,
		init, sizeof(init) / sizeof(double), 
		ranges, indicesOfMutVars, writableVar,
		maxValue, TT,
		a, sizeof(a) / sizeof(double),
		std::string(LLE_OUTPUT_PATH) + "/damir_test.csv"
	);

#endif

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
	//runBifurcationPerformanceTests();
	//runLLEPerformanceTests();
	runBasinsPerformanceTests();
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
		//double h = 0.01;
		
		std::cout << "Start basins" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		
		Basins::basinsOfAttraction_2(
			200,         // Время моделирования системы
			500,         // Разрешение диаграммы
			0.01,         // Шаг интегрирования
			sizeof(init) / sizeof(double),   // Количество начальных условий ( уравнений в системе )
			init,         // Массив с начальными условиями
			ranges,
			indicesOfMutVars,
			1,          // Индекс уравнения, по которому будем строить диаграмму
			100000000,        // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
			2000,         // Время, которое будет промоделировано перед расчетом диаграммы
			params,         // Параметры
			sizeof(params) / sizeof(double),  // Количество параметров
			1,          // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
			0.05,         // Эпсилон для алгоритма DBSCAN
			std::string(BASINS_OUTPUT_PATH) + "/basinsOfAttraction_system_for_basins_graph.csv"
		);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
	}
	//runPerformanceTests();
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
	
	// Запуск тестов производительности LLE
	//runLLEPerformanceTests();

	// Запуск тестов производительности Bifurcation
	//runBifurcationPerformanceTests();

	std::cout << "Общее время выполнения: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	return 0;
} 