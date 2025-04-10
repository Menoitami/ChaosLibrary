import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import os

# Создание путей к директориям
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(current_dir, "build", "Debug")
results_dir = os.path.join(current_dir, "results")

# Создаем директорию results, если она не существует
os.makedirs(results_dir, exist_ok=True)

# Путь к вашему CSV-файлу
csv_file_path = os.path.join(build_dir, "LLE2D_my.csv")  # Убедитесь, что путь правильный

# Чтение CSV-файла без принудительного указания типа
data = pd.read_csv(csv_file_path, header=None)

# Проверка исходных данных
print("Raw DataFrame:")
print(data)
print("\nData types:")
print(data.dtypes)

# Попытка конвертировать все в float, заменяя нечисловые значения на NaN
data_numeric = data.apply(pd.to_numeric, errors='coerce')

# Проверка после конверсии
print("\nConverted DataFrame (non-numeric replaced with NaN):")
print(data_numeric)
print("\nData types after conversion:")
print(data_numeric.dtypes)

# Замена NaN на 0 (или другое значение, если нужно)
data_numeric = data_numeric.fillna(0)

# Преобразование в numpy массив
data_array = data_numeric.to_numpy()

# Проверка массива
print("\nNumpy array shape:", data_array.shape)
print("Numpy array dtype:", data_array.dtype)

# Создание тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(data_array, cmap="YlGnBu")

# Настройка графика
plt.title("Heatmap of d_result")
plt.xlabel("Column Index")
plt.ylabel("Row Index") 

# Сохранение графика
output_file = os.path.join(results_dir, "heatmap.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Heatmap saved to {output_file}")

# Показать график
plt.show()