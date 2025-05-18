import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy import stats

# Путь к CSV-файлу
csv_path = os.path.join(os.path.dirname(__file__), '../results/performance_results.csv')

data = pd.read_csv(csv_path)

# --- 1. Execution Time vs Resolution ---
res_data = data[data['Parameter'] == 'Resolution']
res_vals = res_data['Value'].astype(int)

plt.figure(figsize=(7,5))
for lib in res_data['Library'].unique():
    plt.plot(res_vals[res_data['Library'] == lib],
             res_data[res_data['Library'] == lib]['ExecutionTime_ms'],
             marker='o', label=lib)
plt.xlabel('Resolution')
plt.ylabel('Execution Time (ms)')
plt.title('Basins Execution Time vs Resolution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../results/basins_time_vs_resolution.png'))

# --- 2. Execution Time vs ModelingTime ---
mod_data = data[data['Parameter'] == 'ModelingTime']
mod_vals = mod_data['Value'].astype(int)

plt.figure(figsize=(7,5))
for lib in mod_data['Library'].unique():
    plt.plot(mod_vals[mod_data['Library'] == lib],
             mod_data[mod_data['Library'] == lib]['ExecutionTime_ms'],
             marker='o', label=lib)
plt.xlabel('Modeling Time')
plt.ylabel('Execution Time (ms)')
plt.title('Basins Execution Time vs Modeling Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../results/basins_time_vs_modelingtime.png'))

# --- Дополнительная информация ---
def compute_speedup(df, ref_lib='old_library', comp_lib='Basins'):
    speedups = []
    values = []
    for val in df['Value'].unique():
        try:
            t_ref = df[(df['Value'] == val) & (df['Library'] == ref_lib)]['ExecutionTime_ms'].values[0]
            t_comp = df[(df['Value'] == val) & (df['Library'] == comp_lib)]['ExecutionTime_ms'].values[0]
            speedups.append(t_comp / t_ref if t_ref != 0 else float('nan'))
            values.append(val)
        except:
            pass
    return speedups, values

# Вычисляем ускорение для разных тестов, учитывая имена библиотек в данных
libs = data['Library'].unique()
ref_lib = libs[0] if len(libs) > 0 else 'old_library'
comp_lib = libs[1] if len(libs) > 1 else 'Basins'

speedup_res, vals_res = compute_speedup(res_data, ref_lib, comp_lib)
speedup_mod, vals_mod = compute_speedup(mod_data, ref_lib, comp_lib)

print('\n--- Дополнительная информация для Basins ---')
if speedup_res:
    print(f'Среднее ускорение {comp_lib} относительно {ref_lib} (Resolution): {sum(speedup_res)/len(speedup_res):.3f}')
    print(f'Минимальное ускорение (Resolution): {min(speedup_res):.3f}')
    print(f'Максимальное ускорение (Resolution): {max(speedup_res):.3f}')
if speedup_mod:
    print(f'Среднее ускорение {comp_lib} относительно {ref_lib} (ModelingTime): {sum(speedup_mod)/len(speedup_mod):.3f}')
    print(f'Минимальное ускорение (ModelingTime): {min(speedup_mod):.3f}')
    print(f'Максимальное ускорение (ModelingTime): {max(speedup_mod):.3f}')

# Среднее время выполнения для каждой реализации
for lib in data['Library'].unique():
    avg_time = data[data['Library'] == lib]['ExecutionTime_ms'].mean()
    print(f'Среднее время выполнения для {lib}: {avg_time:.1f} мс')

# --- Дополнительные графики и анализ динамики ускорения ---

# График ускорения vs Resolution
if speedup_res and len(vals_res) > 1:
    plt.figure(figsize=(7,5))
    vals_res = [float(v) for v in vals_res]
    plt.plot(vals_res, speedup_res, marker='o', label=f'Speedup {comp_lib}/{ref_lib}')
    
    # Линейная регрессия для прогнозирования тренда
    if len(vals_res) > 2:
        x = np.array(vals_res)
        y = np.array(speedup_res)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Предсказать ускорение для больших значений
        future_vals = [max(vals_res) * 1.5, max(vals_res) * 2]
        future_speedups = [slope * v + intercept for v in future_vals]
        
        print(f"\n--- Анализ динамики ускорения для Resolution ---")
        print(f"Тренд ускорения: y = {slope:.6f}x + {intercept:.6f}, R² = {r_value**2:.3f}")
        print(f"Прогноз ускорения для resolution = {future_vals[0]}: {future_speedups[0]:.3f}")
        print(f"Прогноз ускорения для resolution = {future_vals[1]}: {future_speedups[1]:.3f}")
        
        # Отображение линии тренда на графике
        trend_x = np.linspace(min(vals_res), max(future_vals), 100)
        trend_y = slope * trend_x + intercept
        plt.plot(trend_x, trend_y, '--', label=f'Тренд (R² = {r_value**2:.3f})')
        
        # Отображение прогнозируемых точек
        plt.plot(future_vals, future_speedups, 'rx', markersize=8, label='Прогноз')
    
    plt.xlabel('Resolution')
    plt.ylabel(f'Speedup ({comp_lib}/{ref_lib})')
    plt.title('Speedup vs Resolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/basins_speedup_vs_resolution.png'))

# График ускорения vs ModelingTime
if speedup_mod and len(vals_mod) > 1:
    plt.figure(figsize=(7,5))
    vals_mod = [float(v) for v in vals_mod]
    plt.plot(vals_mod, speedup_mod, marker='o', label=f'Speedup {comp_lib}/{ref_lib}')
    
    # Линейная регрессия для прогнозирования тренда
    if len(vals_mod) > 2:
        x = np.array(vals_mod)
        y = np.array(speedup_mod)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Предсказать ускорение для больших значений
        future_vals = [max(vals_mod) * 1.5, max(vals_mod) * 2]
        future_speedups = [slope * v + intercept for v in future_vals]
        
        print(f"\n--- Анализ динамики ускорения для ModelingTime ---")
        print(f"Тренд ускорения: y = {slope:.6f}x + {intercept:.6f}, R² = {r_value**2:.3f}")
        print(f"Прогноз ускорения для modeling_time = {future_vals[0]}: {future_speedups[0]:.3f}")
        print(f"Прогноз ускорения для modeling_time = {future_vals[1]}: {future_speedups[1]:.3f}")
        
        # Отображение линии тренда на графике
        trend_x = np.linspace(min(vals_mod), max(future_vals), 100)
        trend_y = slope * trend_x + intercept
        plt.plot(trend_x, trend_y, '--', label=f'Тренд (R² = {r_value**2:.3f})')
        
        # Отображение прогнозируемых точек
        plt.plot(future_vals, future_speedups, 'rx', markersize=8, label='Прогноз')
    
    plt.xlabel('Modeling Time')
    plt.ylabel(f'Speedup ({comp_lib}/{ref_lib})')
    plt.title('Speedup vs Modeling Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/basins_speedup_vs_modelingtime.png'))

# Дополнительный график: сравнение относительной скорости роста времени выполнения
if len(res_data) > 2:
    plt.figure(figsize=(7,5))
    
    libs = res_data['Library'].unique()
    for lib in libs:
        lib_data = res_data[res_data['Library'] == lib]
        base_time = lib_data[lib_data['Value'] == min(lib_data['Value'])]['ExecutionTime_ms'].values[0]
        relative_times = [t/base_time for t in lib_data['ExecutionTime_ms']]
        plt.plot(lib_data['Value'].astype(int), relative_times, marker='o', label=f'{lib}')
    
    plt.xlabel('Resolution')
    plt.ylabel('Relative Execution Time (normalized)')
    plt.title('Relative Growth of Execution Time vs Resolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/basins_relative_growth.png'))

print('\nГрафики сохранены в папке results.') 