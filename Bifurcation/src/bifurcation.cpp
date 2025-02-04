#include "bifurcation.h"


void writeToCSV(const std::vector<std::vector<double>> &histEntropy2D, int cols, int rows, const std::string &filename)
{
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << histEntropy2D[i][j];
            if (j < cols - 1)
            {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Data successfully written to file " << filename << std::endl;
}



void writeToCSV(const std::vector<double> &histEntropy1D, int cols, const std::string &filename)
{
    auto histEntropy2D = convert1DTo2D(histEntropy1D);
    writeToCSV(histEntropy2D, cols, 1, filename);
}


std::vector<std::vector<double>> convert1DTo2D(const std::vector<double> &histEntropy1D)
{
    return {histEntropy1D};
}



int Bifurcation1D(std::string argv) {
    

    std::string inputString = argv;

    if (inputString.size() < 4 || 
        (inputString.substr(inputString.size() - 4) != ".csv" &&
         inputString.substr(inputString.size() - 4) != ".txt")) {
        std::cerr << "Error: File must have a .csv or .txt extension!" << std::endl;
        return 1;
    }

    std::ofstream outFile(inputString);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return 1;
    }

    outFile.close();

   
    // Задаем параметры для расчета
    double transTime = 1000;  // Время переходного процесса
    double tMax = 2000;       // Время моделирования после TT
    double h = 0.01;          // Шаг интегрирования

    std::vector<double> X = {0.1, 0.1, 0}; // Начальное состояние
    int coord = 0;                        // Координата для анализа

    std::vector<double> params = {0, 0.2, 0.2, 5.7}; // Параметры модели (params[0] - коэффициент симметрии)
    

    double startBin = -20; // Начало гистограммы
    double endBin = 20;    // Конец гистограммы
    double stepBin = 0.01;  // Шаг бинов гистограммы

    // Параметры для linspace
    double linspaceStartA = 0.1;  // Начало диапазона параметра
    double linspaceEndA = 0.35;   // Конец диапазона параметра
    int linspaceNumA = 200;       // Количество точек параметра
    int paramNumberA = 1;         // Индекс параметра для анализа


    double linspaceStartB = 0.1;  // Начало диапазона параметра
    double linspaceEndB = 0.2;   // Конец диапазона параметра
    int linspaceNumB = 200;
    int paramNumberB = 2;         // Индекс параметра для анализа

    auto start = std::chrono::high_resolution_clock::now();

    //Вызов функции histEntropyCUDA3D
    std::vector<std::vector<double>> histEntropy3D = histEntropyCUDA3D(
                                        transTime, tMax, h,
                                        X, coord,
                                        params, paramNumberA,paramNumberB,
                                        startBin, endBin, stepBin,
                                        linspaceStartA,linspaceEndA, linspaceNumA, linspaceStartB,linspaceEndB, linspaceNumB
                                    );

    writeToCSV(histEntropy3D,linspaceNumA,linspaceNumB,inputString);

    // std::vector<double> histEntropy2D = histEntropyCUDA2D(
    //                                     transTime, tMax, h,
    //                                     X, coord,
    //                                     params, paramNumberA,
    //                                     startBin, endBin, stepBin,
    //                                     linspaceStartA,linspaceEndA, linspaceNumA
    //                                 );

    // writeToCSV(histEntropy2D,linspaceNumA,inputString);



    std::cout<<"End of gpu part\n";


    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = stop - start;
    std::cout << "Program execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}