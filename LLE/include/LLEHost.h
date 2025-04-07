#ifndef LLEHOST_H
#define LLEHOST_H

#include <string>

namespace LLE {

/**
 * @brief Вычисляет показатель Ляпунова для двумерной системы
 * 
 * @param tMax Максимальное время расчета
 * @param NT Шаг времени для расчета показателя Ляпунова
 * @param h Шаг интегрирования
 * @param eps Величина возмущения для расчета показателя Ляпунова
 * @param transientTime Время переходного процесса (перед началом расчета LLE)
 * @param initialConditions Начальные условия системы
 * @param amount_init Количество начальных условий
 * @param params Параметры системы
 * @param amount_params Количество параметров
 * @param linspaceA_params Параметры первого измерения ([0] - начало, [1] - конец, [2] - количество точек)
 * @param linspaceB_params Параметры второго измерения ([0] - начало, [1] - конец, [2] - количество точек)
 * @param indicesOfMutVars Индексы изменяемых параметров в массиве params
 * @param OUT_FILE_PATH Путь к файлу для сохранения результатов
 */
void LLE2D(
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
	std::string		OUT_FILE_PATH
);

} // namespace LLE_constants


#endif // LLEHOST_H 