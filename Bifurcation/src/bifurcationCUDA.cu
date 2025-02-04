#include "bifurcationCUDA.cuh"

// Необходимые переменные, что добавляются в константную память видеокарты
__constant__ int d_bifurSize; // Общий размер массива бифуркации
__constant__ int d_XSize;
__constant__ int d_paramsSize;
__constant__ int d_paramNumber;
__constant__ int d_coord;
__constant__ double d_h;
__constant__ double d_transTime;
__constant__ double d_tMax;

__device__ const double EPSD = std::numeric_limits<double>::epsilon();


 __host__ std::vector<double> linspaceNum(double start, double end, int num)
{
    std::vector<double> result;

    if (num < 0)
        throw std::invalid_argument("received negative number of points");
    if (num == 0)
        return result;
    if (num == 1)
    {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i)
    {
        result.push_back(start + i * step);
    }

    return result;
}





__device__ __host__ void calculateDiscreteModel(double *x, const double *a, const double h)
{
    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    
    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);
/*
    Пример системы из наших лабораторных

    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    x[0] += h1 * x[1];
    x[1] += h1 * x[2];
    x[2] += h1 * (x[1] * x[3] - a[1] * x[2] * x[3]);
    x[3] += h1 * (a[2] * x[0] * x[2] - x[1] * x[1] + x[2] * x[2]);

    x[3] += h2 * (a[2] * x[0] * x[2] - x[1] * x[1] + x[2] * x[2]);
    x[2] = (x[2] + h2 * x[1] * x[3]) / (1 + a[1] * h2 * x[3]);
    x[1] += h2 * x[2];
    x[0] += h2 * x[1];
*/
}

__device__ __host__ bool loopCalculateDiscreteModel(double *x, const double *values, const double h,
                                                    const int amountOfIterations, const int preScaller,
                                                    int writableVar, const double maxValue, double *data, const int startDataIndex, const int writeStep)
{
    bool exceededMaxValue = (maxValue != 0);
    for (int i = 0; i < amountOfIterations; ++i)
    {
        if (data != nullptr)
            data[startDataIndex + i * writeStep] = x[writableVar];

        for (int j = 0; j < preScaller - 1; ++j)
            calculateDiscreteModel(x, values, h);

        calculateDiscreteModel(x, values, h);

        if (exceededMaxValue && fabsf(x[writableVar]) > maxValue)
            return false;
    }
    return true;
}


__device__ double calculateEntropy(double *bins, int binSize, const int sum)
{
    double entropy = 0.0;

    for (int i = 0; i < binSize; ++i)
    {
        if (bins[i] > 0)
        {
            bins[i] = (bins[i] / (sum + EPSD));
            entropy -= (bins[i] - EPSD) * log2(bins[i]);
        }
    }

    return entropy;
}


__device__ void CalculateHistogram(
    double *X, const double *param,
    int &sum, double *bins)
{

    int iterations = static_cast<int>(d_tMax / d_h);
    double last = X[d_coord];
    bool lastBigger = false;

    int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));

    for (int i = 0; i < iterations; ++i)
    {
        calculateDiscreteModel(X, param, d_h);

        if (X[d_coord] > last)
        {
            lastBigger = true;
        }
        else if (X[d_coord] < last && lastBigger)
        {
            if (last >= d_startBin && last < d_endBin)
            {

                int index = static_cast<int>((last - d_startBin) / d_stepBin);
                if (index < binSize && index >= 0)
                {
                    bins[index]++;
                }
                sum++;
            }
            lastBigger = false;
        }
        last = X[d_coord];
    }
}


__global__ void calculateHistEntropyCuda3D(const double *X,
                                           const double *params,
                                           const double *paramLinspaceA,
                                           const double *paramLinspaceB,
                                           double *histEntropy,
                                           double *bins_global)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_histEntropySize)
        return;

    int binSize = static_cast<int>(std::ceil((d_endBin - d_startBin) / d_stepBin));
    int offset = idx * binSize;
    if (offset + binSize > d_histEntropySize * binSize)
        return;
    double *bin = &bins_global[offset];

    int row = idx / d_histEntropySizeCol;
    int col = idx % d_histEntropySizeCol;
    
    // Влияет на скорость работы, поставили прозапас, но нужно изменять размеры под актуальное решение! (без агрессии, просто обратить внимание)
    double X_locals[32];
    double params_local[32];

    memcpy(X_locals, X, d_XSize * sizeof(double));
    memcpy(params_local, params, d_paramsSize * sizeof(double));

    params_local[d_paramNumberA] = paramLinspaceA[row];
    params_local[d_paramNumberB] = paramLinspaceB[col];


    loopCalculateDiscreteModel(X_locals, params_local, d_h,
                               static_cast<int>(d_transTime / d_h),
                               0, 0, 0, nullptr, 0, 0);

    int sum = 0;

    CalculateHistogram(X_locals, params_local, sum, bin);
}


__host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime, const double tMax, const double h,
    const std::vector<double> &X, const int coord,
    const std::vector<double> &params, const int paramNumberA, const int paramNumberB,
    const double startBin, const double endBin, const double stepBin,
    double linspaceStartA, double linspaceEndA, int linspaceNumA, double linspaceStartB, double linspaceEndB, int linspaceNumB)
{
    //Проверка на корректность введеных данных
    try
    {
        if (tMax <= 0)
            throw std::invalid_argument("tMax <= 0");
        if (transTime <= 0)
            throw std::invalid_argument("transTime <= 0");
        if (h <= 0)
            throw std::invalid_argument("h <= 0");
        if (startBin >= endBin)
            throw std::invalid_argument("binStart >= binEnd");
        if (stepBin <= 0)
            throw std::invalid_argument("binStep <= 0");
        if (coord < 0 || coord >= X.size())
            throw std::invalid_argument("coord out of range X");
        if (paramNumberA < 0 || paramNumberA >= params.size())
            throw std::invalid_argument("paramNumber out of range params param 1");
        if (paramNumberB < 0 || paramNumberB >= params.size())
            throw std::invalid_argument("paramNumber out of range params param 2");
        if (paramNumberB == paramNumberA)
            throw std::invalid_argument("param 1 == param 2");
        //    if (linspaceStartB == linspaceEndB) throw std::invalid_argument("linspaceStartB == linspaceEndB");
        //    if (linspaceStartA == linspaceEndA) throw std::invalid_argument("linspaceStartA == linspaceEndA");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<std::vector<double>>(0);
    }

    //Определение устройства, на котором будет запускаться программа
    int device = 0;
    cudaDeviceProp deviceProp;
    int numBlocks;
    int threadsPerBlock;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    // !Из-за большого колличества потоков в блоке, может появляться ошибка о нехватке памяти!
    // Для ее исправления стоит уменьшить максимальное колличество блоков
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; 

    //Деление на итерации происходит в зависимости от свободной памяти на видеокарте freeMem(итерация - 1 запуск ядра GPU)
    size_t freeMem, totalMem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    freeMem *= 0.8 / 1024 / 1024; // bytes

    //Выделение памяти для переменных и перенос их в константную память на GPU
    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = linspaceNumA;
    int histEntropySizeCol = linspaceNumB;
    int histEntropySize = histEntropySizeRow * histEntropySizeCol;
    int binSize = static_cast<int>(std::ceil((endBin - startBin) / stepBin));
    std::vector<double> paramLinspaceA = linspaceNum(linspaceStartA, linspaceEndA, linspaceNumA);
    std::vector<double> paramLinspaceB = linspaceNum(linspaceStartB, linspaceEndB, linspaceNumB);

    double *d_X, *d_params, *d_paramLinspaceA;

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_startBin, &startBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_endBin, &endBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_stepBin, &stepBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_transTime, &transTime, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_coord, &coord, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberA, &paramNumberA, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberB, &paramNumberB, sizeof(int)));

    // Выделение памяти для массивов, что будут передаваться в ядро
    CHECK_CUDA_ERROR(cudaMalloc(&d_X, XSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, paramsSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice));

    // Определение необходимого колличества итераций
    // 1. Считается общее необходимое колличество итераций
    // 2. Оценивается возможность выполнения этого в памяти видеокарты
    // 3. В случае, если памяти на GPU не хватает, происходит деление по B на две половины. (Если b_size == 1, то происходит деление по A)
    double bytes = static_cast<double>(histEntropySize) * sizeof(double) / (1024 * 1024) + static_cast<double>(histEntropySize) * static_cast<double>(binSize) * sizeof(double) / (1024 * 1024);

    int iteratationsB = 1;
    int iteratationsA = 1;

    double memEnabledB = linspaceNumB;
    double memEnabledA = linspaceNumA;

    while ((((memEnabledA * std::ceil(memEnabledB) + static_cast<double>(binSize) * memEnabledA * std::ceil(memEnabledB)) * sizeof(double))) / (1024 * 1024) > freeMem)
    {
        if (std::ceil(memEnabledB) <= 1)
        {
            memEnabledA /= 2;
            iteratationsA *= 2;
        }
        else
        {
            memEnabledB /= 2;
            iteratationsB *= 2;
        }
    }

    memEnabledA = std::ceil(memEnabledA);
    memEnabledB = std::ceil(memEnabledB);

    std::cout << "freeMem: " << freeMem << "MB Needed Mbytes: " << bytes << "MB\n";
    std::cout << "iterationsB: " << iteratationsB << " B_size: " << " " << memEnabledB << "\n";
    std::cout << "iterationsA: " << iteratationsA << " A_size: " << " " << memEnabledA << "\n";

    std::vector<std::vector<double>> histEntropy2DFinal;
    histEntropy2DFinal.reserve(histEntropySizeCol);
    int SizeCol;
    double remainingB;
    double currentStepB;
    double processedB = 0;

    //Начало выполнения итераций. 
    for (int i = 0; i < iteratationsB; ++i)
    {
        // Массив B выделяется в соответсвии со свободной памятью GPU
        remainingB = histEntropySizeCol - processedB;
        currentStepB = std::min(memEnabledB, remainingB);

        std::vector<double> partParamLinspaceB(currentStepB);

        std::copy(
            paramLinspaceB.begin() + processedB,
            paramLinspaceB.begin() + processedB + currentStepB,
            partParamLinspaceB.begin());

        processedB += currentStepB;
        SizeCol = partParamLinspaceB.size();

        double *d_paramLinspaceB, *d_histEntropy;

        int SizeRow;
        double remainingA;
        double currentStepA;
        double processedA = 0;

        std::vector<double> histEntropyRowFinal;

        for (int j = 0; j < iteratationsA; ++j)
        {
            // Массив A выделяется в соответсвии со свободной памятью GPU
            remainingA = histEntropySizeRow - processedA;
            currentStepA = std::min(memEnabledA, remainingA);

            std::vector<double> partParamLinspaceA(currentStepA);

            std::copy(
                paramLinspaceA.begin() + processedA,
                paramLinspaceA.begin() + processedA + currentStepA,
                partParamLinspaceA.begin());

            processedA += currentStepA;
            SizeRow = partParamLinspaceA.size();

            histEntropySize = SizeRow * SizeCol;
            std::vector<double> histEntropy(histEntropySize);

            //Когда границы итерации установлены, выделяется память под все переменные, зависящее от размера A и B
            CHECK_CUDA_ERROR(cudaMalloc(&d_histEntropy, histEntropySize * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceB, SizeCol * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceA, SizeRow * sizeof(double)));

            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceB, partParamLinspaceB.data(), SizeCol * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceA, partParamLinspaceA.data(), SizeRow * sizeof(double), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &histEntropySize, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeCol, &SizeCol, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeRow, &SizeRow, sizeof(int)));

            double *bins;
            cudaMalloc(&bins, (histEntropySize * binSize) * sizeof(double));
            cudaMemset(bins, 0, histEntropySize * binSize * sizeof(double));

            // Определение колличества потоков и блоков для запуска ядра
            // Если данных на блок не очень много, то программа старается выделить блоков в 4 раза больше чем колличество мультипроцессоров
            // Данный подход показывает лучшую производительность 
            numBlocks = deviceProp.multiProcessorCount * 4;
            threadsPerBlock = std::ceil(histEntropySize / (float)numBlocks);

            if (threadsPerBlock > maxThreadsPerBlock)
            {
                threadsPerBlock = maxThreadsPerBlock;
                numBlocks = std::ceil(histEntropySize / (float)threadsPerBlock);
            }
            else if (threadsPerBlock == 0)
                threadsPerBlock = 1;
            std::cout << "Memory block is: " << i << " / " << iteratationsB << "\n";
            std::cout << "blocks: " << numBlocks << " threads: " << threadsPerBlock << " sm's: " << deviceProp.multiProcessorCount << "\n";
            int progress = 0;
            cudaMemcpyToSymbol(d_progress, &progress, sizeof(int));

            // Запуск функции ядра
            calculateHistEntropyCuda3D<<<numBlocks, threadsPerBlock>>>(
                d_X, d_params, d_paramLinspaceA, d_paramLinspaceB, d_histEntropy, bins);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("CUDA kernel execution failed");
            }

            // Запись финального массива в память
            std::vector<std::vector<double>> histEntropy2D(SizeCol, std::vector<double>(SizeRow));
            cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);

            if (memEnabledB == 1)
            {
                histEntropyRowFinal.insert(histEntropyRowFinal.end(), histEntropy.begin(), histEntropy.end());
            }
            else
            {
                for (int q = 0; q < SizeRow; ++q)
                {
                    for (int s = 0; s < SizeCol; ++s)
                    {
                        histEntropy2D[s][q] = std::move(histEntropy[q * SizeCol + s]);
                    }
                }
                for (int q = 0; q < SizeCol; ++q)
                {
                    histEntropy2DFinal.push_back(histEntropy2D[q]);
                }
            }

            cudaFree(bins);
            cudaFree(d_paramLinspaceA);
        }

        if (memEnabledB == 1)
        {
            histEntropy2DFinal.push_back(histEntropyRowFinal);
        }
        cudaFree(d_paramLinspaceB);
    }

    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);

    return histEntropy2DFinal;
}

