#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap

def load_data(filepath):
    data = []
    max_cols = 0
    x_range = [0, 1]
    y_range = [0, 1]
    
    with open(filepath, 'r') as f:
        # Чтение диапазонов из первых двух строк
        x_range = list(map(float, f.readline().strip().split()))
        y_range = list(map(float, f.readline().strip().split()))
        
        for line in f:
            row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
            if row:
                data.append(row)
                max_cols = max(max_cols, len(row))
    
    # Создаем массив с дополнением нулями до макс. ширины
    padded_data = np.zeros((len(data), max_cols), dtype=np.float32)
    for i, row in enumerate(data):
        padded_data[i, :len(row)] = row
    
    return padded_data, x_range, y_range

def create_custom_colormap():
    # Создаем кастомную цветовую карту для отображения бассейнов притяжения
    colors = []
    # Добавляем разные цвета для разных бассейнов притяжения
    for i in range(134):  # Максимальное значение аттрактора из файла - 134
        if i == 0:
            colors.append((0, 0, 0))  # Черный для значения 0
        else:
            # Циклически выбираем из нескольких цветов для лучшей различимости
            color_idx = i % 10
            if color_idx == 1:
                colors.append((0.8, 0.2, 0.2))  # Красный
            elif color_idx == 2:
                colors.append((0.2, 0.8, 0.2))  # Зеленый
            elif color_idx == 3:
                colors.append((0.2, 0.2, 0.8))  # Синий
            elif color_idx == 4:
                colors.append((0.8, 0.8, 0.2))  # Желтый
            elif color_idx == 5:
                colors.append((0.8, 0.2, 0.8))  # Пурпурный
            elif color_idx == 6:
                colors.append((0.2, 0.8, 0.8))  # Голубой
            elif color_idx == 7:
                colors.append((0.6, 0.4, 0.2))  # Коричневый
            elif color_idx == 8:
                colors.append((0.4, 0.2, 0.6))  # Фиолетовый
            elif color_idx == 9:
                colors.append((0.2, 0.6, 0.4))  # Бирюзовый
            else:
                colors.append((0.5, 0.5, 0.5))  # Серый
    
    return LinearSegmentedColormap.from_list('custom_basins', colors, N=len(colors))

def main():
    # Настройка путей
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    input_dir = os.path.join(workspace_root, 'workspace/bifurcation')
    output_dir = os.path.join(workspace_root, 'results')
    
    # Создаем директорию для результатов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для результатов: {output_dir}")
    
    # Поиск файлов в директории basins
    basin_files = []
    for file in os.listdir(input_dir):
        if file.startswith("bifurcation") and file.endswith(".csv"):
            basin_files.append(file)
    
    # Сортировка файлов
    basin_files.sort()
    
    # Если не нашли файлы, сообщаем и выходим
    if not basin_files:
        print("Не найдены файлы с данными бассейнов в директории:", input_dir)
        return
    
    print(f"Найдены следующие файлы для отрисовки: {basin_files}")
    
    # Создаем кастомную цветовую карту
    custom_cmap = create_custom_colormap()
    
    # Настройка фигуры в зависимости от количества файлов
    n_files = len(basin_files)
    if n_files <= 2:
        fig, axes = plt.subplots(1, n_files, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(2, (n_files + 1) // 2, figsize=(14, 12))
        axes = axes.flatten()
    
    # Если только один файл, превращаем axes в список для единообразия кода
    if n_files == 1:
        axes = [axes]
    
    # Загрузка и отрисовка данных
    for i, file in enumerate(basin_files):
        input_filepath = os.path.join(input_dir, file)
        
        # Загрузка данных
        print(f"Загрузка данных из {input_filepath}...")
        try:
            data, x_range, y_range = load_data(input_filepath)
            
            # Анализ данных
            print(f"\nАнализ данных {file}:")
            print(f"Размер данных: {data.shape}")
            print(f"Диапазон X: {x_range}")
            print(f"Диапазон Y: {y_range}")
            
            # Отрисовка тепловой карты
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            im = axes[i].imshow(data, cmap=custom_cmap, aspect='auto', 
                               extent=extent, origin='lower')
            
            #axes[i].set_title(f"Бассейны {file}")
            axes[i].set_xlabel('Параметр X')
            axes[i].set_ylabel('Параметр Y')
            
            # Добавляем цветовую шкалу для каждого графика
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Номер аттрактора')
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")
            axes[i].text(0.5, 0.5, f"Ошибка загрузки:\n{e}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[i].transAxes, color='red')
    
    # Скрываем неиспользуемые оси, если их больше чем файлов
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Сохранение результата
    output_filepath = os.path.join(output_dir, 'basins_comparison.png')
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    print(f"Сравнительная визуализация бассейнов сохранена в {output_filepath}")
    
    # Показать график
    plt.show()

if __name__ == "__main__":
    main() 