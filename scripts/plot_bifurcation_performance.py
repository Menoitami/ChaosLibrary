import matplotlib.pyplot as plt
import pandas as pd
import os

# Путь к CSV-файлу
csv_path = os.path.join(os.path.dirname(__file__), '../results/bifurcation_performance_results copy.csv')

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
plt.title('Execution Time vs Resolution (2D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../results/bifurcation_time_vs_resolution.png'))

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
plt.title('Execution Time vs Modeling Time (2D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../results/bifurcation_time_vs_modelingtime.png'))

# --- Дополнительная информация ---
def compute_speedup(df):
    speedups = []
    for val in df['Value'].unique():
        try:
            t_old = df[(df['Value'] == val) & (df['Library'] == 'old_library')]['ExecutionTime_ms'].values[0]
            t_new = df[(df['Value'] == val) & (df['Library'] == 'Bifurcation')]['ExecutionTime_ms'].values[0]
            speedups.append(t_old / t_new if t_new != 0 else float('nan'))
        except:
            pass
    return speedups

speedup_res = compute_speedup(res_data)
speedup_mod = compute_speedup(mod_data)

print('\n--- Дополнительная информация ---')
if speedup_res:
    print(f'Среднее ускорение (Resolution): {sum(speedup_res)/len(speedup_res):.3f}')
    print(f'Минимальное ускорение (Resolution): {min(speedup_res):.3f}')
    print(f'Максимальное ускорение (Resolution): {max(speedup_res):.3f}')
if speedup_mod:
    print(f'Среднее ускорение (ModelingTime): {sum(speedup_mod)/len(speedup_mod):.3f}')
    print(f'Минимальное ускорение (ModelingTime): {min(speedup_mod):.3f}')
    print(f'Максимальное ускорение (ModelingTime): {max(speedup_mod):.3f}')

# Среднее время выполнения для каждой реализации
for lib in data['Library'].unique():
    avg_time = data[data['Library'] == lib]['ExecutionTime_ms'].mean()
    print(f'Среднее время выполнения для {lib}: {avg_time:.1f} мс')

print('\nГрафики сохранены в папке results.') 