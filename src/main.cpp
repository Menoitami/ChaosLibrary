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
	double TT = 500000;
	double CT = 10000;
/*
//					   b     a  m  w   mu
double params[7]{ 0.5, 3, 0, 2, 1, 18, 1 };
//double params[6]{ 0.5, 3, -1, 1, 1, 1.53 };
double init[3]{ 3, 3, 0 };





	Bifurcation::bifurcation2D(
		400, // const double tMax,
		1000, // const int nPts,
		h, // const double h,
		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
		init, // const double* initialConditions,
		//new double[4]{ -5, 5, -5, 5 }, // const double* ranges,
		new double[4]{ 0, 5, 0, 20 }, // const double* ranges,
		new int[2]{ 3, 5 }, // const int* indicesOfMutVars,
		0, // const int writableVar,
		10000, // const double maxValue,
		5000, // const double transientTime,
		params, // const double* values,
		sizeof(params) / sizeof(double), // const int amountOfValues,
		1, // const int preScaller,
		0.001, //eps
		std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_1.csv"
	);



	LLE::LLE2D(
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
		std::string(LLE_OUTPUT_PATH) + "/lle_2.csv"
	);
*/

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

	std::cout << "--- Запуск тестов Bifurcation ---" << std::endl;

	// --- Bifurcation: Легкий запуск ---
	{
		double tMax = 100;
		int nPts = 50;
		double transientTime = 1000;

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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_easy.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_easy_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "Bifurcation (Easy) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Easy)!" << std::endl;
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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_easy_old.csv" // Другой файл для результатов
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_easy_old_timing.txt"; // Другой файл для времени
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::Bifurcation (Easy) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Easy)!" << std::endl;
			}
		}
	}

	// --- Bifurcation: Средний запуск ---
	{
		double tMax = 400;
		int nPts = 100;
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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "Bifurcation (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Medium)!" << std::endl;
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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_old.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_medium_old_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::Bifurcation (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Medium)!" << std::endl;
			}
		}
	}

	// --- Bifurcation: Тяжелый запуск ---
	{
		double tMax = 1000;
		int nPts = 200;
		double transientTime = 10000;
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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "Bifurcation (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени Bifurcation (Hard)!" << std::endl;
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
				std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_old.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_hard_old_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::Bifurcation (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::Bifurcation (Hard)!" << std::endl;
			}
		}
	}

	std::cout << "\n--- Запуск тестов LLE ---" << std::endl;

	 // --- LLE: Легкий запуск ---
	{
		double tMax = 100;
		int nPts = 50;
		double transientTime = 1000;

		// --- Новый LLE ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			LLE::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_easy.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_easy_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "LLE (Easy) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени LLE (Easy)!" << std::endl;
			}
		}

		// --- Старый LLE (old_library) ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			old_library::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_easy_old.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_easy_old_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::LLE (Easy) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::LLE (Easy)!" << std::endl;
			}
		}
	}

	 // --- LLE: Средний запуск ---
	{
		double tMax = 500;
		int nPts = 100;
		double transientTime = 1000;

		// --- Новый LLE ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			LLE::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_medium.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_medium_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "LLE (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени LLE (Medium)!" << std::endl;
			}
		}

		// --- Старый LLE (old_library) ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			old_library::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_medium_old.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_medium_old_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::LLE (Medium) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::LLE (Medium)!" << std::endl;
			}
		}
	}

	 // --- LLE: Тяжелый запуск ---
	{
		double tMax = 1000;
		int nPts = 200;
		double transientTime = 5000;
		// --- Новый LLE ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			LLE::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_hard.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_hard_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "LLE (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени LLE (Hard)!" << std::endl;
			}
		}
		// --- Старый LLE (old_library) ---
		{
			auto start = std::chrono::high_resolution_clock::now();
			old_library::LLE2D(
				tMax, NT_lle, nPts, h, eps_lle,
				init, sizeof(init) / sizeof(double),
				ranges, indicesOfMutVars,
				writableVar, maxValue, transientTime,
				a, sizeof(a) / sizeof(double),
				std::string(LLE_OUTPUT_PATH) + "/lle_hard_old.csv"
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::string timingFileName = std::string(LLE_OUTPUT_PATH) + "/LLE_hard_old_timing.txt";
			std::ofstream outFile(timingFileName);
			if (outFile) {
				outFile << duration;
				outFile.close();
				std::cout << "old_library::LLE (Hard) Время выполнения: " << duration << " мс. Результат записан в файл: " << timingFileName << std::endl;
			} else {
				std::cerr << "Ошибка: Не удалось открыть файл для записи времени old_library::LLE (Hard)!" << std::endl;
			}
		}
	}

	// Старый код Bifurcation и LLE оставлен закомментированным для справки
/*
	double a[4]{ 0, 0.2, 0.2, 5.7};
	double init[3]{ 0.01, 0 , 0 };

	Bifurcation::bifurcation2D(
		400, // const double tMax,
		100, // const int nPts,
		h, // const double h,
		sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
		init, // const double* initialConditions,
		//new double[4]{ -5, 5, -5, 5 }, // const double* ranges,
		new double[4]{ -0, 0.6, -0, 0.6 }, // const double* ranges,
		new int[2]{ 1, 2 }, // const int* indicesOfMutVars,
		0, // const int writableVar,
		10000, // const double maxValue,
		5000, // const double transientTime,
		a, // const double* values,
		sizeof(a) / sizeof(double), // const int amountOfValues,
		1, // const int preScaller,
		0.001, 
		std::string(BIFURCATION_OUTPUT_PATH) + "/Bifurcation_1_rossler_old.csv"
	);

	// LLE::LLE2D(
	// 	500,		//const double tMax,
	// 	0.5,		//const double NT,
	// 	100,		//const int nPts,
	// 	h,			//const double h,
	// 	1e-6,		//const double eps,
	// 	init,		//const double* initialConditions,
	// 	sizeof(init) / sizeof(double),		//const int amountOfInitialConditions,
	// 	new double[4]{ -0, 0.6, -0, 0.6 },		//const double* ranges,
	// 	new int[2]{ 1, 2 },					//const int* indicesOfMutVars,
	// 	0,			//const int writableVar,
	// 	10000,		//const double maxValue,
	// 	1000,		//const double transientTime,
	// 	a,		//const double* values,
	// 	sizeof(a) / sizeof(double),	//const int amountOfValues,
	// 	std::string(LLE_OUTPUT_PATH) + "/lle_rossler.csv"
	// );
*/
	
	//printf(" --- Time of runnig: %zu ms", std::clock() - startTime);
    return 0;
}