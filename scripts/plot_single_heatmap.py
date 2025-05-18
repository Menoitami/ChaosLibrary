#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

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

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Отрисовка тепловой карты из данных CSV')
    parser.add_argument('--input', '-i', type=str, default='damir_test.csv',
                        help='Имя входного файла')
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help='Директория с входным файлом')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Имя выходного файла')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Минимальное значение для цветовой шкалы')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Максимальное значение для цветовой шкалы')
    parser.add_argument('--cmap', type=str, default='hot',
                        help='Цветовая карта (например, hot, viridis, jet, coolwarm)')
    parser.add_argument('--title', type=str, default=None,
                        help='Заголовок графика')
    args = parser.parse_args()
    
    # Настройка путей
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    
    if args.dir:
        input_dir = os.path.join(workspace_root, args.dir)
    else:
        input_dir = os.path.join(workspace_root, 'workspace/LLE')
    
    output_dir = os.path.join(workspace_root, 'results')
    
    # Создаем директорию для результатов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для результатов: {output_dir}")

    # Настройка имен файлов
    input_file = args.input
    
    if args.output:
        output_file = args.output
    else:
        # Используем имя входного файла, заменив расширение на .png
        output_file = os.path.splitext(input_file)[0] + '.png'
    
    # Полные пути к файлам
    input_filepath = os.path.join(input_dir, input_file)
    output_filepath = os.path.join(output_dir, output_file)
    
    # Загрузка данных
    print(f"Загрузка данных из {input_filepath}...")
    try:
        data, x_range, y_range = load_data(input_filepath)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # Анализ данных
    print(f"\nАнализ данных:")
    print(f"Размер данных: {data.shape}")
    print(f"Диапазон X: {x_range}")
    print(f"Диапазон Y: {y_range}")
    
    # Определение заголовка
    if args.title:
        title = args.title
    else:
        title = f"LLE 2D"
    
    # Построение графика
    plt.figure(figsize=(10, 8))
    
    # Создаем экстенты для правильного масштабирования осей
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    
    im = plt.imshow(data, cmap=args.cmap, aspect='auto', vmin=args.vmin, vmax=args.vmax, 
                   extent=extent, origin='lower')
    # Переворачиваем данные на 180 градусов
    plt.gca().invert_yaxis()
    plt.colorbar(im, label='Значение')
    plt.title(title)
    plt.xlabel('Параметр X')
    plt.ylabel('Параметр Y')
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    print(f"Тепловая карта сохранена в {output_filepath}")
    
    plt.show()

if __name__ == "__main__":
    main() 