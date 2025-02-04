#pragma once

#include "bifurcationCUDA.cuh"

#include<iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>


/**
 * @brief Записывает 2D-вектор данных в CSV-файл.
 * 
 * @param histEntropy2D Двумерный вектор данных для записи.
 * @param cols Количество столбцов в данных.
 * @param rows Количество строк в данных.
 * @param filename Имя выходного CSV-файла.
 * 
 * @details Записывает данные в формате CSV, разделяя элементы строк запятыми.
 * Если файл не удается открыть, выводит сообщение об ошибке.
 */
void writeToCSV(const std::vector<std::vector<double>> &histEntropy2D, int cols, int rows, const std::string &filename);


/**
 * @brief Записывает 1D-вектор данных в CSV-файл.
 * 
 * @param histEntropy1D Одномерный вектор данных для записи.
 * @param cols Количество столбцов в данных.
 * @param filename Имя выходного CSV-файла.
 * 
 * @details Преобразует 1D-вектор в 2D-вектор с одной строкой и записывает в CSV-файл.
 * Если файл не удается открыть, выводит сообщение об ошибке.
 */
void writeToCSV(const std::vector<double> &histEntropy1D, int cols, const std::string &filename);


/**
 * @brief Преобразует одномерный вектор в двумерный.
 * 
 * @param histEntropy1D Одномерный вектор данных.
 * 
 * @return std::vector<std::vector<double>> Двумерный вектор, где исходный вектор 
 * становится единственной строкой.
 */
std::vector<std::vector<double>> convert1DTo2D(const std::vector<double> &histEntropy1D);


/** @brief Бифуркация
 * 
 * @param argv путь выходного файла
 * 
 * 
 * @return стандартный вывод, если 0 все сработало, не 0 что-то сломалось
*/
int Bifurcation1D(std::string argv);

