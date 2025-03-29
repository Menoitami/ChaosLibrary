// --- Заголовочный файл ---
#include "hostLibrary.cuh"

// --- Путь для сохранения результирующих файлов ---
//#define OUT_FILE_PATH "C:\\Users\\KiShiVi\\Desktop\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// --- Директива, объявление которой выводит в консоль отладочные сообщения ---
#define DEBUG
namespace old_library{
__host__ void FastSynchro(
	const double	tMax,								// Время моделирования системы
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	const double*	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const double	h,									// Шаг интегрирования
	const double*	kForward,							// Массив коэффициентов синхронизации вперед
	const double*	kBackward,							// Массив коэффициентов синхронизации назад
	const double*	initialConditionsMaster,			// Массив с начальными условиями мастера
	const double*	initialConditionsSlave,				// Массив с начальными условиями слейва
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	const int		iterOfSynchr,						// Число итераций синхронизации
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое используется в окне синхронизации ---
	int amountOfNTPoints = NTime / h;

	// --- Общее количесвто точек в исходной траектории ---
	int amountOfCTPoints = tMax / h;

	// --- Количество точек переходного процесса ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU
	size_t nPts = (amountOfCTPoints / preScaller);

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
// TODO Сделать расчет требуемой памяти
	//size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfNTPoints * (amountOfInitialConditions*amountOfInitialConditions*amountOfInitialConditions));
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfNTPoints * amountOfNTPoints*5);
	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter; // Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)
	
	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )


	double* timeDomain = new double[(amountOfCTPoints + amountOfNTPoints)* sizeof(double) * amountOfInitialConditions];
	double* arrayZeros = new double[sizeof(double) * amountOfInitialConditions];
	double* Xm = new double[sizeof(double) * amountOfInitialConditions];
	double* Xs = new double[sizeof(double) * amountOfInitialConditions];

	// --- Инициализация начальных условий ---
	for (int i = 0; i < amountOfInitialConditions; i++) {
		arrayZeros[i] = 0;
		Xm[i] = initialConditionsMaster[i];
		Xs[i] = initialConditionsSlave[i];
	}
		
	// --- Расчет переходного процесса ---
	for (size_t i = 0; i < amountOfPointsForSkip; i++) {
		calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h);
		calculateDiscreteModelforFastSynchro(Xs, arrayZeros, arrayZeros, values, h);
	}

	//for (int i = 0; i < amountOfInitialConditions; i++) {
	//	Xs[i] = initialConditionsSlave[i];
	//}

	// --- Расчет исходной траектории ---
	for (size_t i = 0; i < amountOfCTPoints + amountOfNTPoints; i++) {

		for (int j = 0; j < amountOfInitialConditions; j++)
			timeDomain[i * amountOfInitialConditions + j] = Xm[j];

		calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h);

	}

	printf(" --- Calculation of trajectory done\n");

	// --- Выделяем память для хранения конечного результата 
	double* h_output = new double[nPts * sizeof(double)];

	// --- Указатели на области памяти в GPU ---

	double* d_timeDomain;
	double* d_output;
	double* d_Xs;
	double* d_values;
	double* d_kForward;
	double* d_kBackward;
	// --- Выделяем память в GPU ---

	gpuErrorCheck(cudaMalloc((void**)& d_timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_output, nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_Xs, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kForward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kBackward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	// --- Копируем начальные входные параметры в память GPU ---

	gpuErrorCheck(cudaMemcpy(d_timeDomain, timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_Xs, Xs, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);
	
	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---

		//blockSize = ceil((1*1024.0f * 4.0f) / (amountOfNTPoints * sizeof(double)));
		//blockSize = ceil((1 * 1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		//blockSize = 10000 / ((5*amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSize = 5*amountOfNTPoints;
		//
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )
		blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

	
			//calculateDiscreteModelforFastSynchroCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			//calculateDiscreteModelforFastSynchroCUDA << <4*1024, 16>> > //, 1*(amountOfInitialConditions + amountOfValues + amountOfNTPoints) * sizeof(double)* 1 >> >
		calculateDiscreteModelforFastSynchroCUDA << < gridSize, blockSize >> >
		(
				nPts,						//const int		nPts,
				nPtsLimiter,				//const int		nPtsLimiter,
				1*amountOfNTPoints,		//const int		sizeOfBlock,
				h, 							//const double	h,
				d_Xs,						//double* initialConditions,
				amountOfInitialConditions,	//const int		amountOfInitialConditions,
				d_values,						//const double* values,
				d_kForward,					//const double* k_forward,
				d_kBackward,					//const double* k_backward,
				iterOfSynchr,							//const int		iterOfSynchr,
				amountOfValues,								//const int		
				amountOfNTPoints,							//const int		amountOfIterations,
				maxValue,									//const double	maxValue,
				d_timeDomain + (i* originalNPtsLimiter)* amountOfInitialConditions * preScaller,								//double* timedomain,
				d_output + (i* originalNPtsLimiter),			//double* output
				preScaller);

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

			
#ifdef DEBUG
		printf(" --- Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}
		// --- Перенос расчитанного реезультата с gpu 
	gpuErrorCheck(cudaMemcpy(h_output, d_output, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- Освобождение памяти в gpu 
	gpuErrorCheck(cudaFree(d_timeDomain));
	gpuErrorCheck(cudaFree(d_output));
	gpuErrorCheck(cudaFree(d_kForward));
	gpuErrorCheck(cudaFree(d_kBackward));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_Xs));

	// --- ЗАпись реузльтата в файл 
	outFileStream << std::setprecision(20);

	for (size_t j = 0; j < nPts; ++j)
		if (outFileStream.is_open())
		{
			for (int k = 0; k < amountOfInitialConditions; k++) 
				outFileStream << timeDomain[amountOfInitialConditions * j * preScaller + k] << ", ";		

			outFileStream << h_output[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}
	outFileStream.close();

	printf(" --- Writing in file done\n");

	delete[] arrayZeros;
	delete[] timeDomain;
	delete[] Xm;
	delete[] Xs;
	delete[] h_output;
}

__host__ void FastSynchro_2(
	const double	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	const int		nPts,							// Разрешение диаграммы
	const double*	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const double	h,									// Шаг интегрирования
	const double*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const double*	kForward,							// Массив коэффициентов синхронизации вперед
	const double*	kBackward,							// Массив коэффициентов синхронизации назад
	const double*	initialConditions,			// Массив с начальными условиями мастера
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	const int		iterOfSynchr,						// Число итераций синхронизации
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	std::string		OUT_FILE_PATH)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = NTime / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = 0;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfInitialConditions * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	double* h_dbscanResult = new double[nPtsLimiter * sizeof(double)];

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
	double* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив
	
	double* d_kForward;
	double* d_kBackward;

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfInitialConditions * nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kForward, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_kBackward, amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
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
	printf("Bifurcation 2DIC\n");
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
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCforFastSynchro << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfInitialConditions*amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
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
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				iterOfSynchr,
				d_kForward,
				d_kBackward,
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks,
				d_dbscanResult);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему


		//const int		nPts,
		//	const int		nPtsLimiter,
		//	const int		sizeOfBlock,
		//	const int		amountOfCalculatedPoints,
		//	const int		dimension,
		//	double* ranges,
		//	const double	h,
		//	int* indicesOfMutVars,
		//	double* initialConditions,
		//	const int		amountOfInitialConditions,
		//	const double* values,
		//	const int		amountOfValues,
		//	const int		amountOfIterations,
		//	const int		preScaller,
		//	const double	maxValue,
		//	const int		iterOfSynchr,
		//	const double* kForward,
		//	const double* kBackward,
		//	double* data,
		//	int* maxValueCheckerArray,
		//	double* FastSynchroError)

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

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

	
}

__host__ void distributedSystemSimulation(
	const double	tMax,							// Время моделирования системы
	const double	h,								// Шаг интегрирования
	const double	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,				// Массив с начальными условиями
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,							// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)					// Количество параметров	
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.8;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)	

	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_data = new double[amountOfPointsInBlock * sizeof(double)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Distributed System Simulation\n");
#endif

	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
	// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
	// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
	// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
	blockSize = ceil((1024.0f * 8.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
	if (blockSize < 1)
	{
#ifdef DEBUG
		printf("Error : BlockSize < 1; %d line\n", __LINE__);
		exit(1);
#endif
	}

	blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

	gridSize = (amountOfThreads + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

	distributedCalculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
		(
			amountOfPointsForSkip,
			amountOfThreads,
			h,
			hSpecial,
			d_initialConditions,
			amountOfInitialConditions,
			d_values,
			amountOfValues,
			tMax / hSpecial,
			writableVar,
			d_data
			);

	// --- Проверка на CUDA ошибки ---
	gpuGlobalErrorCheck();

	// --- Ждем пока все потоки завершат свою работу ---
	gpuErrorCheck(cudaDeviceSynchronize());

	// -------------------------------------------------------------------------------------
	// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
	// -------------------------------------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------------------

	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(20);

	for (size_t j = 0; j < amountOfPointsInBlock; ++j)
		if (outFileStream.is_open())
		{
			outFileStream << h * j << ", " << h_data[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}


	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	delete[] h_data;
}


// ----------------------------------------------------------------------------
// --- Определение функции, для расчета одномерной бифуркационной диаграммы ---
// ----------------------------------------------------------------------------

__host__ void bifurcation1D(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,				// Массив с начальными условиями
	const double* ranges,							// Диаппазон изменения переменной
	const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	double* h_outIntervals = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// ВОТ тут влез слава
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	
	int* d_sysCheker;			// Указатель на массив в GPU на вспомогательный массив
	double* d_avgPeaks;
	double* d_avgIntervals;

	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * sizeof(double)));


	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(
				nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
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

		//calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
		//	(nPts,						// Общее разрешение диаграммы - nPts
		//		nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
		//		amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
		//		i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
		//		amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
		//		1,							// Размерность ( диаграмма одномерная )
		//		d_ranges,					// Массив с диапазонами
		//		h,							// Шаг интегрирования
		//		d_indicesOfMutVars,			// Индексы изменяемых параметров
		//		d_initialConditions,		// Начальные условия
		//		amountOfInitialConditions,	// Количество начальных условий
		//		d_values,					// Параметры
		//		amountOfValues,				// Количество параметров
		//		amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
		//		preScaller,					// Множитель, который уменьшает время и объем расчетов
		//		writableVar,				// Индекс уравнения, по которому будем строить диаграмму
		//		maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		//		d_data,						// Массив, где будет хранится траектория систем
		//		d_sysCheker + (i * originalNPtsLimiter));			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал здесь не нужен
				h*preScaller);							// Шаг интегрирования не нужен

		////		// --- Проверка на CUDA ошибки ---
		////gpuGlobalErrorCheck();

		////// --- Ждем пока все потоки завершат свою работу ---
		////gpuErrorCheck(cudaDeviceSynchronize());

		////// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		////cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_for2Dbif, 0, nPtsLimiter);
		//////blockSize = blockSize > 512 ? 512 : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		////gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		////avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
		////	(d_data,						// Данные с траекториями систем
		////		amountOfPointsInBlock,		// Количество точек в одной траектории
		////		nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
		////		d_avgPeaks + (i * originalNPtsLimiter),
		////		d_avgIntervals + (i * originalNPtsLimiter),
		////		d_data,						// Выходной массив, куда будут записаны значения пиков
		////		d_intervals,				// Межпиковый интервал
		////		d_amountOfPeaks + (i * originalNPtsLimiter),
		////		d_sysCheker + (i * originalNPtsLimiter),
		////		h* preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_outIntervals, d_intervals, nPtsLimiter* amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
		for (size_t k = 0; k < nPtsLimiter; ++k) {
			if (h_amountOfPeaks[k] == -1) {
				outFileStream << getValueByIdx(originalNPtsLimiter* i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock] << ", " << 0  << '\n';
			}
			else {
				for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
					if (outFileStream.is_open())
					{
						outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << ", " << h_outIntervals[k * amountOfPointsInBlock + j] << '\n';
					}
					else
					{
#ifdef DEBUG
						printf("\nOutput file open error\n");
#endif
						exit(1);
					}
			}
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}


	//double* h_avgPeaks = new double[nPts];
	//double* h_avgIntervals = new double[nPts];
	//int* h_sysCheker = new int[nPts];
	//int* h_amountOfPeaks = new int[nPts];

	//gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


//	std::ofstream outFileStream;
//	outFileStream.open(OUT_FILE_PATH);
//	outFileStream << std::setprecision(12);
//
//		// --- Сохранение данных в файл ---
//			for (size_t j = 0; j < nPts; ++j)
//				if (outFileStream.is_open())
//				{
//					outFileStream  << h_avgPeaks[j] << ", " << h_avgIntervals[j] << ", " << h_amountOfPeaks[j] << '\n';
//				}
//				else
//				{
//#ifdef DEBUG
//					printf("\nOutput file open error\n");
//#endif
//					exit(1);
//				}

					   

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));
	gpuErrorCheck(cudaFree(d_sysCheker));



	delete[] h_outPeaks;
	delete[] h_outIntervals;
	delete[] h_amountOfPeaks;
	

	// ---------------------------
}



/**
 * Функция, для расчета одномерной бифуркационной диаграммы по шагу.
 */
__host__ void bifurcation1DForH(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,				// Массив с начальными условиями
	const double* ranges,							// Диапазон изменения шага
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек в одном блоке ---
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				transientTime,				// Время пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				tMax,						// Количество итераций ( равно количеству точек для одной системы )
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
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
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
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// -----------------------------------------------------------------------------------------
// --- Функция, для расчета одномерной бифуркационной диаграммы. (По начальным условиям) ---
// -----------------------------------------------------------------------------------------

__host__ void bifurcation1DIC(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,				// Массив с начальными условиями
	const double* ranges,							// Диаппазон изменения переменной
	const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	// Пояснение: пиков не может быть больше, чем (amountOfPointsInBlock / 2), т.к. после пика не может снова идти пик
	double* h_outPeaks = new double[ceil(nPtsLimiter * amountOfPointsInBlock * sizeof(double) / 2.0f)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
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
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
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

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// ------------------------------------------------------------------------
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) ---
// ------------------------------------------------------------------------

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




__host__ void neuronClasterization2D_2(
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

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

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
	
	int* d_sysCheker;			// Указатель на массив в GPU на вспомогательный массив
	double* d_avgPeaks;
	double* d_avgIntervals;
	double* d_helpfulArray;

	//double* d_valleys;
	//double* d_TimeOfValleys;// Указатель на массив в GPU с межпиковыми интервалами минимумов
		//int* d_dbscanResult;
	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));


	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_valleys, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_TimeOfValleys, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
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

		// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	//for (int i = 1; i < 5; i++) {
	//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
	//	// --- Выводим в самое начало файла исследуемые диапазон ---
	//	if (outFileStream.is_open())
	//	{
	//		outFileStream << ranges[0] << " " << ranges[1] << "\n";
	//		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	//	}
	//	outFileStream.close();
	//}
	

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл



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
		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

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


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
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
				d_sysCheker + (i* originalNPtsLimiter));			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA_for_NeuronClassification, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA_for_NeuronClassification << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks + (i* originalNPtsLimiter),			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				nullptr,
				nullptr,
				h * preScaller);							// Шаг интегрирования

		// -----------------------------------------


		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, NeuronClassificationCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		NeuronClassificationCUDA << <gridSize, blockSize >> >
			(	
				d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks + (i* originalNPtsLimiter),
				d_intervals,
				nullptr,
				nullptr,
				d_helpfulArray,
				eps,
				d_dbscanResult + (i* originalNPtsLimiter)
			);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------



		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		//gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		//gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		//outFileStream << std::setprecision(12);

//		// --- Сохранение данных в файл ---
//		for (size_t i = 0; i < nPtsLimiter; ++i)
//			if (outFileStream.is_open())
//			{
//				if (stringCounter != 0)
//					outFileStream << ", ";
//				if (stringCounter == nPts)
//				{
//					outFileStream << "\n";
//					stringCounter = 0;
//				}
//				outFileStream << h_dbscanResult[i];
//				++stringCounter;
//			}
//			else
//			{
//#ifdef DEBUG
//				printf("\nOutput file open error\n");
//#endif
//				exit(1);
//			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	//double* h_avgPeaks = new double[nPts * nPts];
	//double* h_avgIntervals = new double[nPts * nPts];
	//int* h_sysCheker = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];
	//int* h_amountOfPeaks = new int[nPts * nPts];

	//gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts* nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts* nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	//gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	


	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

		// --- Сохранение найденных бассейнов притяжений в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
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
	outFileStream.close();

//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_avgPeaks[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_avgIntervals[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_sysCheker[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();
//
//	stringCounter = 0;
//	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(4) + ".csv", std::ios::app);
//	for (size_t i = 0; i < nPts * nPts; ++i)
//		if (outFileStream.is_open())
//		{
//			if (stringCounter != 0)
//				outFileStream << ", ";
//			if (stringCounter == nPts)
//			{
//				outFileStream << "\n";
//				stringCounter = 0;
//			}
//			outFileStream << h_amountOfPeaks[i];
//			++stringCounter;
//		}
//		else
//		{
//#ifdef DEBUG
//			printf("\nOutput file open error\n");
//#endif
//			exit(1);
//		}
//	outFileStream.close();

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));

	//gpuErrorCheck(cudaFree(d_valleys));
	//gpuErrorCheck(cudaFree(d_TimeOfValleys));

	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_sysCheker));

	delete[] h_dbscanResult;
	//delete[] h_avgPeaks;
	//delete[] h_avgIntervals;
	//delete[] h_sysCheker; 
	//delete[] h_amountOfPeaks;
	// ---------------------------
}

__host__ void neuronClasterization2D(
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

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

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

	int* d_sysCheker;			// Указатель на массив в GPU на вспомогательный массив
	double* d_avgPeaks;
	double* d_avgIntervals;
	double* d_helpfulArray;
	//int* d_dbscanResult;
// -----------------------------------------

// -----------------------------
// --- Выделяем память в GPU ---
// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));


	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
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

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	for (int i = 1; i < 5; i++) {
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}


	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл



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
		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

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


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
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
				d_sysCheker + (i * originalNPtsLimiter));			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_for2Dbif, 0, nPtsLimiter);
		//blockSize = blockSize > 512 ? 512 : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		//peakFinderCUDA << <gridSize, blockSize >> >
		//	(d_data,						// Данные с траекториями систем
		//		amountOfPointsInBlock,		// Количество точек в одной траектории
		//		nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
		//		d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
		//		d_data,						// Выходной массив, куда будут записаны значения пиков
		//		d_intervals,				// Межпиковый интервал
		//		h * preScaller);							// Шаг интегрирования

		avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_amountOfPeaks + (i * originalNPtsLimiter),
				d_sysCheker + (i * originalNPtsLimiter),
				h * preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


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

		dbscanCUDA << <gridSize, blockSize >> >
			(d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks + (i * originalNPtsLimiter),
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult + (i * originalNPtsLimiter)
				);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------



		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		//outFileStream << std::setprecision(12);

//		// --- Сохранение данных в файл ---
//		for (size_t i = 0; i < nPtsLimiter; ++i)
//			if (outFileStream.is_open())
//			{
//				if (stringCounter != 0)
//					outFileStream << ", ";
//				if (stringCounter == nPts)
//				{
//					outFileStream << "\n";
//					stringCounter = 0;
//				}
//				outFileStream << h_dbscanResult[i];
//				++stringCounter;
//			}
//			else
//			{
//#ifdef DEBUG
//				printf("\nOutput file open error\n");
//#endif
//				exit(1);
//			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	int* h_sysCheker = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];
	int* h_amountOfPeaks = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));



	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

		// --- Сохранение найденных бассейнов притяжений в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
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
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgIntervals[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_sysCheker[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(4) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_amountOfPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_sysCheker));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_sysCheker;
	delete[] h_amountOfPeaks;
	// ---------------------------
}

// ------------------------------------------------------------------------------
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) по IC ---
// ------------------------------------------------------------------------------

__host__ void bifurcation2DIC(
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

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

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
	printf("Bifurcation 2DIC\n");
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
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
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
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * preScaller);							// Шаг интегрирования

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

		dbscanCUDA << <gridSize, blockSize >> >
			(d_data, 					// Данные (пики)
				amountOfPointsInBlock, 		// Количество точек в одной системе
				nPtsLimiter,				// Количество блоков (систем) в data
				d_amountOfPeaks, 			// Массив, содержащий количество пиков для каждого блока в data
				d_intervals, 				// Межпиковые интервалы
				d_helpfulArray, 			// Вспомогательный массив 
				eps, 						// Эпселон
				d_dbscanResult);			// Результирующий массив

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



__host__ void LLE1D(
	const double	tMax,								// Время моделирования системы
	const double	NT,									// Время нормализации
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const double	eps,								// Эпсилон для LLE
	const double* initialConditions,					// Массив с начальными условиями
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,								// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)						// Количество параметров
{
	// --- Количество точек, которое будет смоделировано одной системой во время нормализации NT ---
	size_t amountOfNT_points = NT / h;

	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / NT;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	   // Указатель на массив с начальными условиями
	double* d_values;				   // Указатель на массив с параметрами

	double* d_lleResult;			   // Память для хранения конечного результата

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("LLE 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// Переменная для хранения размера блока
		int minGridSize;	// Переменная для хранения минимального размера сетки
		int gridSize;		// Переменная для хранения сетки

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double)));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		// ------------------------------------
		// --- CUDA функция для расчета LLE ---
		// ------------------------------------

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,								// Общее разрешение
				nPtsLimiter, 						// Разрешение в текущем расчете
				NT, 								// Время нормализации
				tMax, 								// Время моделирования
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,				// Количество точек, которое будет промоделированно до основного расчета (transientTime)
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h, 									// Шаг интегрирования
				eps, 								// Эпсилон
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions, 			// Количество начальных условий
				d_values, 							// Параметры
				amountOfValues, 					// Количество параметров
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,						// Индекс переменной в x[] по которому строим диаграмму
				maxValue, 							// Макксимальное значение переменной при моделировании
				d_lleResult);						// Результирующий массив

		// ------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



__host__ void LLE1DIC(
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
	std::string		OUT_FILE_PATH)
{
	// --- Количество точек, которое будет смоделировано одной системой во время нормализации NT ---
	size_t amountOfNT_points = NT / h;

	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / NT;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	   // Указатель на массив с начальными условиями
	double* d_values;				   // Указатель на массив с параметрами

	double* d_lleResult;			   // Память для хранения конечного результата

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("LLE 1DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// Переменная для хранения размера блока
		int minGridSize;	// Переменная для хранения минимального размера сетки
		int gridSize;		// Переменная для хранения сетки

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double)));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		// ------------------------------------
		// --- CUDA функция для расчета LLE ---
		// ------------------------------------

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,								// Общее разрешение
				nPtsLimiter, 						// Разрешение в текущем расчете
				NT, 								// Время нормализации
				tMax, 								// Время моделирования
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,				// Количество точек, которое будет промоделированно до основного расчета (transientTime)
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h, 									// Шаг интегрирования
				eps, 								// Эпсилон
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions, 			// Количество начальных условий
				d_values, 							// Параметры
				amountOfValues, 					// Количество параметров
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,						// Индекс переменной в x[] по которому строим диаграмму
				maxValue, 							// Макксимальное значение переменной при моделировании
				d_lleResult);						// Результирующий массив

		// ------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	const int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 10000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LLE2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif
	int stringCounter = 0;

	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMax;
		int blockSizeMin;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 10000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			amountOfPointsInBlock, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

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
				if (isnan(h_lleResult[i]))
					outFileStream << 0;
				else 
					outFileStream << h_lleResult[i];

				++stringCounter;
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}


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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 22000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LLE2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif
	int stringCounter = 0;
	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMax;
		int blockSizeMin;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

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
				outFileStream << h_lleResult[i];
				++stringCounter;
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LS1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double)) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			1, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0);
				for (int j = 0; j < amountOfInitialConditions; ++j)
					outFileStream << ", " << h_lleResult[k * amountOfInitialConditions + j];
				outFileStream << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}




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
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts * nPts ? nPts * nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LS2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int* stringCounter = new int[amountOfInitialConditions];

	for (int i = 0; i < amountOfInitialConditions; ++i)
		stringCounter[i] = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		outFileStream.open(OUT_FILE_PATH + std::to_string(i + 1) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double)) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < amountOfInitialConditions; ++k)
		{
			outFileStream.open(OUT_FILE_PATH + std::to_string(k + 1) + ".csv", std::ios::app);
			for (size_t m = 0 + k; m < nPtsLimiter * amountOfInitialConditions; m = m + amountOfInitialConditions)
			{
				if (outFileStream.is_open())
				{
					if (stringCounter[k] != 0)
						outFileStream << ", ";
					if (stringCounter[k] == nPts)
					{
						outFileStream << "\n";
						stringCounter[k] = 0;
					}
					outFileStream << h_lleResult[m];
					stringCounter[k] = stringCounter[k] + 1;
				}
			}
			outFileStream.close();
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] stringCounter;
	delete[] h_lleResult;
}



__host__ void basinsOfAttraction(
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

	freeMemory *= 0.8;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];
	//double* h_helpfulArray = new double[nPts * nPts];			// Указатель на массив в GPU на вспомогательный массив

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
	//double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	double* d_avgPeaks;
	double* d_avgIntervals;

	int* d_sysCheck;

	//int* h_sysCheck = new int[nPtsLimiter * sizeof(int)];


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
	//gpuErrorCheck( cudaMalloc( (void** )&d_helpfulArray,		nPts * nPts * sizeof( double ) ) );

	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_sysCheck, nPts * nPts * sizeof(int)));
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


#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(15);

	// --- Выводим в самое начало файла исследуемые диапазон ---
	outFileStream.open(OUT_FILE_PATH);
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		//blockSize = 12000 / ((amountOfInitialConditions + amountOfValues) * sizeof(double));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
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
				d_sysCheck + (i * originalNPtsLimiter));				// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		avgPeakFinderCUDA << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_sysCheck + (i * originalNPtsLimiter),
				h * preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif

		//gpuErrorCheck(cudaMemcpy(h_sysCheck, d_sysCheck, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


		//outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
		//for (size_t i = 0; i < nPtsLimiter; ++i)
		//	if (outFileStream.is_open())
		//	{
		//		if (stringCounter != 0)
		//			outFileStream << ", ";
		//		if (stringCounter == nPts)
		//		{
		//			outFileStream << "\n";
		//			stringCounter = 0;
		//		}
		//		//outFileStream << h_avgIntervals[i];
		//		if (h_sysCheck[i] != NAN)
		//			outFileStream << h_sysCheck[i];
		//		else
		//			outFileStream << 999;
		//		++stringCounter;
		//	}
		//outFileStream.close();

	}

	// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, 1);
	gridSize = (1 + blockSize - 1) / 1;

	// -----------------------------------------
	// --- CUDA функция для алгоритма DBSCAN ---
	// -----------------------------------------

	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	double* h_helpfulArray = new double[2 * nPts * nPts];
	int* h_sysCheck = new int[nPts * nPts];;

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_sysCheck, d_sysCheck, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	//dbscan(h_avgPeaks, h_avgIntervals, h_helpfulArray, 0, nPts * nPts, 2 * nPts * nPts, 0, eps, nullptr);

	//dbscanCUDA << <gridSize, blockSize >> > 
	//	(	d_avgPeaks, 				// Данные (пики)
	//		nPts * nPts, 				// Количество точек в одной системе
	//		1,							// Количество блоков (систем) в data
	//		d_plugAmountOfPeaks, 		// Массив, содержащий количество пиков для каждого блока в data
	//		d_avgIntervals, 			// Межпиковые интервалы
	//		d_helpfulArray, 			// Вспомогательный массив 
	//		eps, 						// Эпселон
	//		nullptr);					// Результирующий массив

	// -----------------------------------------

	// --- Проверка на CUDA ошибки ---
	//gpuGlobalErrorCheck();

	// --- Ждем пока все потоки завершат свою работу ---
	//gpuErrorCheck(cudaDeviceSynchronize());

	// -------------------------------------------------------------------------------------
	// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
	// -------------------------------------------------------------------------------------

	//gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------------------


			// --- Сохранение данных в файл ---
	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_helpfulArray[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_sysCheck[i] != NAN)
				outFileStream << h_sysCheck[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();



	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	//gpuErrorCheck(cudaFree(d_plugAmountOfPeaks));
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_sysCheck));
	//gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_helpfulArray;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_sysCheck;

	// ---------------------------
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CUDA_dbscan(double* data, double* intervals, int* labels, int* helpfulArray, const int amountOfData, const double eps)
{
	int resultClusters = 0;
	int amountOfClusters = 0;				// Количество кластеров
	int amountOfNegativeClusters = 0;
	int* amountOfNeighbors = new int[1];			// Вспомогательная переменная - сколько было найдено соседей у точки
	*amountOfNeighbors = 0;
	int* neighbors = new int[amountOfData];			// Вспомогательная переменная - индексы найденных соседей

	int* d_amountOfNeighbors;						// Вспомогательная переменная - сколько было найдено соседей у точки
	int* d_neighbors;								// Вспомогательная переменная - индексы найденных соседей

	cudaMalloc((void**)& d_amountOfNeighbors, sizeof(int));
	cudaMalloc((void**)& d_neighbors, sizeof(int) * amountOfData);

	cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

	int amountOfVisitedPoints = 0;

	int blockSize1;			// Переменная для хранения размера блока
	int minGridSize1;		// Переменная для хранения минимального размера сетки
	int gridSize1;			// Переменная для хранения сетки


	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, amountOfData);

	blockSize1 = blockSize1 > 512 ? 512 : blockSize1;			// Не превышаем ограничение в 512 потока в блоке
	gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

	int blockSize2;			// Переменная для хранения размера блока
	int minGridSize2;		// Переменная для хранения минимального размера сетки
	int gridSize2;			// Переменная для хранения сетки

	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, amountOfData);

	blockSize2 = blockSize2 > 512 ? 512 : blockSize2;			// Не превышаем ограничение в 512 потока в блоке
	gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;

	// Цикл по всем точкам даты
	//while (true)
	for (int i = 0; i < amountOfData; i++)
	{
		int* clearIdx = new int[1];
		*clearIdx = -1;

		int* d_clearIdx;

		cudaMalloc((void**)& d_clearIdx, sizeof(int));

		cudaMemcpy(d_clearIdx, clearIdx, sizeof(int), cudaMemcpyHostToDevice);

		CUDA_dbscan_search_fixed_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
			amountOfData, d_clearIdx);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		//gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

		if (*clearIdx == -1)
		{
			CUDA_dbscan_search_clear_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
				amountOfData, d_clearIdx);

			++amountOfClusters;
			resultClusters = amountOfClusters;
			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

			if (*clearIdx == -1)
				break;
		}
		else
		{
			--amountOfNegativeClusters;
			resultClusters = amountOfNegativeClusters;
		}

		*amountOfNeighbors = 0;
		for (size_t i = 0; i < amountOfData; ++i)
			neighbors[i] = 0;

		cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

		CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
			resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, *clearIdx, helpfulArray);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		//gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);


		for (size_t i = 0; i < *amountOfNeighbors; ++i)
		{
			CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
				resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, neighbors[i], helpfulArray);

			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);

			++amountOfVisitedPoints;
		}

		delete[] clearIdx;
		cudaFree(d_clearIdx);
	}

	delete[] amountOfNeighbors;
	delete[] neighbors;

	cudaFree(d_amountOfNeighbors);
	cudaFree(d_neighbors);

}

__host__ void basinsOfAttraction_2(
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

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];
	//double* h_helpfulArray = new double[nPts * nPts];			// Указатель на массив в GPU на вспомогательный массив

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
	int* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	double* d_avgPeaks;
	double* d_avgIntervals;

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
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPts * nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(double)));
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
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(15);

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
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
				d_helpfulArray + (i * originalNPtsLimiter));			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}


	CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps);

	
	double* h_avgPeaks = new double[nPts * nPts];
	double* h_avgIntervals = new double[nPts * nPts];
	int* h_helpfulArray = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks,		 nPts * nPts * sizeof(double),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(double),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- Сохранение найденных бассейнов притяжений в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
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
	outFileStream.close();

	// --- Сохранение средних значений пиков в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение средних значений межпиков в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение характеристик точек сетки начальных условий в файл ---

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			//outFileStream << h_avgIntervals[i];
			if (h_helpfulArray[i] != NAN)
				outFileStream << h_helpfulArray[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();


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
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_helpfulArray;

	// ---------------------------
}




__host__ void TimeDomainCalculation(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double* initialConditions,				// Массив с начальными условиями
	const double* ranges,							// Диаппазон изменения переменной
	const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	size_t amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	size_t amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	//double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	//int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	//double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	//int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPts * amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(double)));

	//gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double)));
	//gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	//std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	//static curandState *states = NULL;

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> >
		//calculateDiscreteModelCUDA_rand << <gridSize, blockSize >> >
			(	
				nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
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
				nullptr);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------



#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
#endif
	}

	double* h_data = new double[amountOfPointsInBlock * nPts];

	gpuErrorCheck(cudaMemcpy(h_data, d_data, nPts* amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- Точность чисел с плавающей запятой ---

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	outFileStream << std::setprecision(16);

	size_t stringCounter = 0;
	//outFileStream.open(OUT_FILE_PATH);
	for (size_t k = 0; k < nPts; ++k) {
		for (size_t i = 0; i < amountOfPointsInBlock; ++i) {
			if (outFileStream.is_open())
			{
				if (h_data[i] != NAN)
					outFileStream << h_data[i + k * amountOfPointsInBlock];
				else
					outFileStream << 999;

				//if (stringCounter != 0)
				//	outFileStream << ", ";
				//if (stringCounter == amountOfPointsInBlock-1)
				//{
				//	outFileStream << "\n";
				//	stringCounter = 0;
				//}
				//else
				outFileStream << ", ";
				//outFileStream << h_avgIntervals[i];

	//			++stringCounter;
			}
		} outFileStream << " \n";
	}
	outFileStream.close();
	printf("End\n");
	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	//gpuErrorCheck(cudaFree(d_outPeaks));
	//gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_data;

	//delete[] h_outPeaks;
	//delete[] h_amountOfPeaks;

	// ---------------------------
}
} // old_library