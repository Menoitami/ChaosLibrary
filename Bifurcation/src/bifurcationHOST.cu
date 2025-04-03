#include "bifurcationHOST.h"
#include "bifurcationCUDA.cuh"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <iomanip>

namespace Bifurcation_constants {

__host__ void bifurcation2D(
	const double	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,					// Массив с начальными условиями
	const double* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const double	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.95;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)(nPts * nPts) / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		
		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		blockSize = 20000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

//		if (blockSize < 1)
//		{
//#ifdef DEBUG
//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
//			exit(1);
//#endif
//		}
//
//		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
//
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >(
				nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * (double)preScaller);							// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > (	
				d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks,
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
#ifdef DEBUG
				printf("\nOutput file open error\n");
#endif
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	// ---------------------------
}

} // namespace Bifurcation_constants
