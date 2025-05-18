#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

# Простые строковые пути
workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_dir = os.path.join(workspace_root, 'workspace/bifurcation')
output_dir = os.path.join(workspace_root, 'results')

# Создаем директорию для результатов, если её нет
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Создана директория для результатов: {output_dir}")

# Имена входных и выходных файлов
input_file1 = 'bif2d_res_test_500.csv'
input_file2 = 'bif2d_old_res_test_500.csv'
output_file = 'bif2d_res_test_500_comparison.png'

def load_data(filepath):
    data = []
    max_cols = 0
    
    with open(filepath, 'r') as f:
        # Пропуск первых двух строк
        next(f)
        next(f)
        
        for line in f:
            row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
            if row:
                data.append(row)
                max_cols = max(max_cols, len(row))
    
    # Создаем массив с дополнением нулями до макс. ширины
    padded_data = np.zeros((len(data), max_cols), dtype=np.float32)
    for i, row in enumerate(data):
        padded_data[i, :len(row)] = row
    
    # Заменяем отрицательные значения на ноль
    padded_data[padded_data < 0] = 0
    
    return padded_data


def main(vmin=None, vmax=None):
    # Полные пути к файлам
    input_filepath1 = os.path.join(input_dir, input_file1)
    input_filepath2 = os.path.join(input_dir, input_file2)
    output_filepath = os.path.join(output_dir, output_file)
    
    # Загрузка данных
    print(f"Загрузка данных из {input_filepath1}...")
    data1 = load_data(input_filepath1)
    
    print(f"Загрузка данных из {input_filepath2}...")
    data2 = load_data(input_filepath2)
    
    # Анализ данных
    print(f"\nАнализ данных:")
    print(f"Размер первых данных: {data1.shape}")
    print(f"Размер вторых данных: {data2.shape}")
    
    # Вычисление разницы
    diff = np.abs(data1 - data2)
    mean_error = np.mean(diff)
    max_error = np.max(diff)
    
    print(f"\nАнализ ошибок:")
    print(f"Средняя абсолютная ошибка: {mean_error:.2e}")
    print(f"Максимальная абсолютная ошибка: {max_error:.2e}")
    
    # Построение графиков
    plt.figure(figsize=(20, 6))
    
    plt.subplot(131)
    plt.imshow(data1, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Данные 1: {input_file1}")
    
    plt.subplot(132)
    plt.imshow(data2, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Данные 2: {input_file2}")
    
    plt.subplot(133)
    plt.imshow(diff, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title("Абсолютная разница")
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    print(f"Сравнительная тепловая карта сохранена в {output_filepath}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage with manual vmin and vmax
    main(vmin=0, vmax=0.5)
