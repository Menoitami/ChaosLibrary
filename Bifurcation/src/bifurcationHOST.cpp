#include "bifurcationHOST.h"

namespace Bifurcation {

void bifurcation2D(
    const double tMax,                              // Время моделирования системы
    const int nPts,                                // Разрешение диаграммы
    const double h,                                // Шаг интегрирования
    const int amountOfInitialConditions,          // Количество начальных условий (уравнений в системе)
    const double* initialConditions,               // Массив с начальными условиями
    const double* ranges,                          // Диапазоны изменения параметров
    const int* indicesOfMutVars,                   // Индексы изменяемых параметров
    const int writableVar,                         // Индекс уравнения, по которому будем строить диаграмму
    const double maxValue,                         // Максимальное значение (по модулю), выше которого система считается "расходящейся"
    const double transientTime,                    // Время, которое будет промоделировано перед расчетом диаграммы
    const double* values,                          // Параметры
    const int amountOfValues,                      // Количество параметров
    const int preScaller,                          // Множитель, который уменьшает время и объем расчетов
    const double eps,                              // Эпсилон для алгоритма DBSCAN
    std::string OUT_FILE_PATH)                     // Путь к выходному файлу
{
    // Вызов реализации из пространства имен Bifurcation_constants
    Bifurcation_constants::bifurcation2D(
        tMax,
        nPts,
        h,
        amountOfInitialConditions,
        initialConditions,
        ranges,
        indicesOfMutVars,
        writableVar,
        maxValue,
        transientTime,
        values,
        amountOfValues,
        preScaller,
        eps,
        OUT_FILE_PATH
    );
}

} // namespace Bifurcation
