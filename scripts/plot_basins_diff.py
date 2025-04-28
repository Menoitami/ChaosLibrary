#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

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

def calculate_difference(data_new, data_old):
    """
    Вычисляет разницу между новой и старой матрицами данных.
    Возвращает матрицу разницы и процент изменений.
    """
    # Обеспечиваем, что матрицы имеют одинаковый размер
    rows = min(data_new.shape[0], data_old.shape[0])
    cols = min(data_new.shape[1], data_old.shape[1])
    
    print(f"Размеры матриц для разности: new={data_new.shape}, old={data_old.shape}, используется={rows}x{cols}")
    
    # Создаем пустую матрицу для разницы
    diff = np.zeros((rows, cols), dtype=np.float32)
    changes_count = 0
    
    # Вычисляем разницу (1 означает изменение)
    for i in range(rows):
        for j in range(cols):
            if data_new[i, j] != data_old[i, j]:
                diff[i, j] = 1
                changes_count += 1
    
    percent = (changes_count / (rows * cols)) * 100
    print(f"Количество изменений: {changes_count} из {rows*cols} ячеек ({percent:.2f}%)")
    
    return diff, percent

def main():
    # Настройка путей
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    input_dir = os.path.join(workspace_root, 'workspace/basins')
    output_dir = os.path.join(workspace_root, 'results')
    
    # Создаем директорию для результатов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для результатов: {output_dir}")
    
    # Списки файлов
    new_files = []
    old_files = []
    
    # Поиск файлов в директории basins
    for file in os.listdir(input_dir):
        if file.startswith("basinsOfAttraction_test.csv"):
            if file == "basinsOfAttraction_test.csv" or file.endswith("_1.csv") or file.endswith("_2.csv") or file.endswith("_3.csv"):
                new_files.append(file)
        elif file.startswith("basinsOfAttraction_test_old.csv"):
            if file == "basinsOfAttraction_test_old.csv" or file.endswith("_1.csv") or file.endswith("_2.csv") or file.endswith("_3.csv"):
                old_files.append(file)
    
    # Явная сортировка файлов в правильном порядке
    def sort_key(filename):
        if filename == "basinsOfAttraction_test.csv" or filename == "basinsOfAttraction_test_old.csv":
            return 0
        elif filename.endswith("_1.csv"):
            return 1
        elif filename.endswith("_2.csv"):
            return 2
        elif filename.endswith("_3.csv"):
            return 3
        else:
            return 999  # для прочих файлов
            
    new_files.sort(key=sort_key)
    old_files.sort(key=sort_key)
    
    print(f"Найдены файлы (new): {new_files}")
    print(f"Найдены файлы (old): {old_files}")
    
    # Проверка наличия файлов
    if not new_files or not old_files:
        print("Не найдены файлы с данными бассейнов в директории:", input_dir)
        return
    
    # Проверка соответствия файлов
    if len(new_files) != len(old_files):
        print(f"Предупреждение: количество new файлов ({len(new_files)}) не соответствует количеству old файлов ({len(old_files)})")
        num_files = min(len(new_files), len(old_files))
        new_files = new_files[:num_files]
        old_files = old_files[:num_files]
        
    # Файлы должны быть сопоставлены корректно
    print("Сопоставление файлов для сравнения:")
    for i, (new, old) in enumerate(zip(new_files, old_files)):
        print(f"  {i+1}. {new} <-> {old}")
    
    # Создаем кастомную цветовую карту для бассейнов
    custom_cmap = create_custom_colormap()
    
    # Для различий используем другую карту
    diff_cmap = 'viridis'
    
    # Настройка фигуры для новых карт
    fig_new, axes_new = plt.subplots(2, 2, figsize=(15, 12))
    axes_new = axes_new.flatten()
    
    # Загрузка и отрисовка новых данных
    for i, file in enumerate(new_files):
        if i >= 4:  # Ограничиваем до 4 файлов
            break
        
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
            im = axes_new[i].imshow(data, cmap=custom_cmap, aspect='auto', 
                               extent=extent, origin='lower')
            
            axes_new[i].set_title(f"Бассейны {file}")
            axes_new[i].set_xlabel('Параметр X')
            axes_new[i].set_ylabel('Параметр Y')
            
            # Добавляем цветовую шкалу для каждого графика
            cbar = plt.colorbar(im, ax=axes_new[i])
            cbar.set_label('Номер аттрактора')
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")
            axes_new[i].text(0.5, 0.5, f"Ошибка загрузки:\n{e}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes_new[i].transAxes, color='red')
    
    # Сохранение результата для новых карт
    output_filepath_new = os.path.join(output_dir, 'basins_new.png')
    plt.tight_layout()
    plt.savefig(output_filepath_new, dpi=300)
    print(f"Визуализация новых бассейнов сохранена в {output_filepath_new}")
    plt.close(fig_new)
    
    # Настройка фигуры для старых карт
    fig_old, axes_old = plt.subplots(2, 2, figsize=(15, 12))
    axes_old = axes_old.flatten()
    
    # Загрузка и отрисовка старых данных
    for i, file in enumerate(old_files):
        if i >= 4:  # Ограничиваем до 4 файлов
            break
        
        input_filepath = os.path.join(input_dir, file)
        
        # Загрузка данных
        print(f"Загрузка старых данных из {input_filepath}...")
        try:
            data, x_range, y_range = load_data(input_filepath)
            
            # Отрисовка тепловой карты
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            im = axes_old[i].imshow(data, cmap=custom_cmap, aspect='auto', 
                               extent=extent, origin='lower')
            
            axes_old[i].set_title(f"Старые бассейны {file}")
            axes_old[i].set_xlabel('Параметр X')
            axes_old[i].set_ylabel('Параметр Y')
            
            # Добавляем цветовую шкалу для каждого графика
            cbar = plt.colorbar(im, ax=axes_old[i])
            cbar.set_label('Номер аттрактора')
            
        except Exception as e:
            print(f"Ошибка при обработке старого файла {file}: {e}")
            axes_old[i].text(0.5, 0.5, f"Ошибка загрузки:\n{e}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes_old[i].transAxes, color='red')
    
    # Сохранение результата для старых карт
    output_filepath_old = os.path.join(output_dir, 'basins_old.png')
    plt.tight_layout()
    plt.savefig(output_filepath_old, dpi=300)
    print(f"Визуализация старых бассейнов сохранена в {output_filepath_old}")
    plt.close(fig_old)
    
    # Настройка фигуры для разностных карт
    fig_diff, axes_diff = plt.subplots(2, 2, figsize=(15, 12))
    axes_diff = axes_diff.flatten()
    
    # Загрузка данных и вычисление разностей
    for i, (new_file, old_file) in enumerate(zip(new_files, old_files)):
        if i >= 4:  # Ограничиваем до 4 файлов
            break
        
        new_filepath = os.path.join(input_dir, new_file)
        old_filepath = os.path.join(input_dir, old_file)
        
        # Загрузка данных
        try:
            new_data, x_range, y_range = load_data(new_filepath)
            old_data, _, _ = load_data(old_filepath)
            
            # Вычисление разницы
            diff_data, percent_changed = calculate_difference(new_data, old_data)
            
            # Отрисовка разностной карты
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            im = axes_diff[i].imshow(diff_data, cmap=diff_cmap, aspect='auto', 
                                 extent=extent, origin='lower', vmin=0, vmax=1)
            
            axes_diff[i].set_title(f"Разница: {os.path.basename(new_file)} vs {os.path.basename(old_file)}\n{percent_changed:.2f}% изменений")
            axes_diff[i].set_xlabel('Параметр X')
            axes_diff[i].set_ylabel('Параметр Y')
            
            # Добавляем цветовую шкалу
            cbar = plt.colorbar(im, ax=axes_diff[i])
            cbar.set_label('Наличие изменений')
            
        except Exception as e:
            print(f"Ошибка при вычислении разницы для {new_file} и {old_file}: {e}")
            axes_diff[i].text(0.5, 0.5, f"Ошибка обработки:\n{e}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes_diff[i].transAxes, color='red')
    
    # Сохранение результата для разностных карт
    output_filepath_diff = os.path.join(output_dir, 'basins_diff.png')
    plt.tight_layout()
    plt.savefig(output_filepath_diff, dpi=300)
    print(f"Визуализация разностей бассейнов сохранена в {output_filepath_diff}")
    plt.close(fig_diff)
    
    print("Обработка завершена.")

if __name__ == "__main__":
    main() 