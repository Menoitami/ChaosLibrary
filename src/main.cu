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
#include <systems.cuh>
//using namespace old_library;


//#define USE_LORENZ_MODEL

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
	double ranges[4]{ -0, 0.6, -0, 0.6 };
	int indicesOfMutVars[2]{ 1, 2 };
	int writableVar = 0;
	double maxValue = 10000;
	double eps_bif = 0.001;
	double eps_lle = 1e-6;
	int preScaller = 1;
	double NT_lle = 0.5;
#endif

	// --- Bifurcation: Легкий запуск ---
	{
		double CT = 1000;
		int nPts = 200;
		double TT = 5000;

		// --- Новый Bifurcation ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			Bifurcation::bifurcation2D(
				CT, nPts, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, TT,
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
				CT, nPts, h,
				sizeof(init) / sizeof(double), init,
				ranges, indicesOfMutVars, writableVar,
				maxValue, TT,
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

	std::cout << "Общее время выполнения: " << (std::clock() - startTime) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
	//std::cout << "Нажмите любую клавишу для выхода..." << std::endl;
	//_getch(); // Ожидание нажатия клавиши перед закрытием консоли

	return 0;
} 