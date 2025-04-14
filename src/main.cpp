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
#include <string>
#include <fstream>
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
	double h = (double)0.01;

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

	// double a[4]{ 0, 0.2, 0.2, 5.7};
	// double init[3]{ 0.01, 0 , 0 };
	// double ranges[4]{ -0, 0.6, -0, 0.6 };
	// int indicesOfMutVars[2]{ 1, 2 };
	// int writableVar = 0;
	// double maxValue = 10000;
	// double eps_bif = 0.001;
	// double eps_lle = 1e-6;
	// int preScaller = 1;
	// double NT_lle = 0.5; // Для LLE

	// std::cout << "--- Запуск тестов Bifurcation ---" << std::endl;

	// --- Bifurcation: Легкий запуск ---
	{
		double tMax = 6000;
		int nPts = 50;
		double transientTime = 5000;

		// --- Новый Bifurcation ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			Bifurcation::bifurcation2D(
				tMax, nPts, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_test.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_test_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Test)!" << std::endl;
			}
		}

		// --- Старый Bifurcation (old_library) ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			old_library::bifurcation2D(
				tMax, nPts, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				preScaller, eps_bif,
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_test_old.csv" // Другой файл для результатов
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_test_old_timing.txt"; // Другой файл для времени
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::Bifurcation (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Test)!" << std::endl;
			}
		}
	}

	// // --- Bifurcation: Средний запуск ---
	// {
	// 	double tMax = 400;
	// 	int nPts = 100;
	// 	double transientTime = 5000;

	// 	// --- Новый Bifurcation ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		Bifurcation::bifurcation2D(
	// 			tMax, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller, eps_bif,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "Bifurcation (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Medium)!" << std::endl;
	// 		}
	// 	}
	// 	// --- Старый Bifurcation (old_library) ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		old_library::bifurcation2D(
	// 			tMax, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller, eps_bif,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_old.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_old_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "old_library::Bifurcation (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Medium)!" << std::endl;
	// 		}
	// 	}
	// }

	// // --- Bifurcation: Тяжелый запуск ---
	// {
	// 	double tMax = 1000;
	// 	int nPts = 200;
	// 	double transientTime = 10000;
	// 	// --- Новый Bifurcation ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		Bifurcation::bifurcation2D(
	// 			tMax, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller, eps_bif,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "Bifurcation (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Hard)!" << std::endl;
	// 		}
	// 	}

	// 	// --- Старый Bifurcation (old_library) ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		old_library::bifurcation2D(
	// 			tMax, nPts, h,
	// 			sizeof(init) / sizeof(double), init,
	// 			ranges, indicesOfMutVars, writableVar,
	// 			maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			preScaller, eps_bif,
	// 			std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_old.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_old_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "old_library::Bifurcation (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Hard)!" << std::endl;
	// 		}
	// 	}
	// }

	// std::cout << "\n--- Запуск тестов LLE ---" << std::endl;

	// {
	// 	double tMax = 500;
	// 	int nPts = 200;
	// 	double transientTime = 1500;

	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		LLE::LLE2D(
	// 			tMax, NT_lle, nPts, h, eps_lle,
	// 			init, sizeof(init) / sizeof(double),
	// 			ranges, indicesOfMutVars,
	// 			writableVar, maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			std::string(LLE_OUTPUT_PATH) + "/lle_test.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_test_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "LLE (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени LLE (Test)!" << std::endl;
	// 		}
	// 	}

	// 	// --- Старый LLE (old_library) ---
	// 	{
	// 		auto start = std::chrono::high_resolution_clock::now();
	// 		old_library::LLE2D(
	// 			tMax, NT_lle, nPts, h, eps_lle,
	// 			init, sizeof(init) / sizeof(double),
	// 			ranges, indicesOfMutVars,
	// 			writableVar, maxValue, transientTime,
	// 			a, sizeof(a) / sizeof(double),
	// 			std::string(LLE_OUTPUT_PATH) + "/lle_test_old.csv"
	// 		);
	// 		auto end = std::chrono::high_resolution_clock::now();
	// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 		std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_test_old_timing.txt";
	// 		std::ofstream outFile(timingFileName);
	// 		if (outFile) {
	// 			outFile << duration;
	// 			outFile.close();
	// 			std::cout << "old_library::LLE (Test) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
	// 		} else {
	// 			std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::LLE (Test)!" << std::endl;
	// 		}
	// 	 }
	// }

    return 0;
}