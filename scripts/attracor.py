# --- Фазовые портреты всех систем из systems.cuh ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. CHAMELEON_MODEL
def chameleon_discrete(steps, dt, init, a, s=0.5):
    h1 = s * dt
    h2 = (1 - s) * dt
    x, y, z = init
    xs = [x]
    ys = [y]
    zs = [z]
    for _ in range(steps):
        # Первый этап расчета
        x1 = xs[-1] + h1 * (-a[6] * ys[-1])
        y1 = ys[-1] + h1 * (a[6] * xs[-1] + a[1] * zs[-1])
        z1 = zs[-1] + h1 * (a[2] - a[3] * zs[-1] + a[4] * np.cos(a[5] * ys[-1]))
        
        # Второй этап расчета
        z2 = (z1 + h2 * (a[2] + a[4] * np.cos(a[5] * ys[-1]))) / (1 + a[3] * h2)
        y2 = y1 + h2 * (a[6] * xs[-1] + a[1] * z2)
        x2 = xs[-1] + h2 * (-a[6] * y2)
        
        xs.append(x2)
        ys.append(y2)
        zs.append(z2)
    return np.array([xs, ys, zs])

# 2. ROSSLER_MODEL
def rossler_discrete(steps, dt, init, a, s=0.5):
    h1 = s * dt
    h2 = (1 - s) * dt
    x, y, z = init
    xs = [x]
    ys = [y]
    zs = [z]
    for _ in range(steps):
        # Первый этап расчета
        x1 = xs[-1] + h1 * (-ys[-1] - zs[-1])
        y1 = ys[-1] + h1 * (xs[-1] + a[1] * ys[-1])
        z1 = zs[-1] + h1 * (a[2] + zs[-1] * (xs[-1] - a[3]))
        
        # Второй этап расчета
        z2 = (z1 + h2 * a[2]) / (1 - h2 * (xs[-1] - a[3]))
        y2 = (y1 + h2 * xs[-1]) / (1 - h2 * a[1])
        x2 = xs[-1] + h2 * (-y2 - z2)
        
        xs.append(x2)
        ys.append(y2)
        zs.append(z2)
    return np.array([xs, ys, zs])

# 3. SYSTEM_FOR_BASINS
def basin_discrete(steps, dt, init, a, s=0.5):
    h1 = s * dt
    h2 = (1 - s) * dt
    x, y, z = init
    xs = [x]
    ys = [y]
    zs = [z]
    for _ in range(steps):
        # Полный шаг
        x1 = xs[-1] + dt * (np.sin(ys[-1]) - a[1] * xs[-1])
        y1 = ys[-1] + dt * (np.sin(zs[-1]) - a[1] * ys[-1])
        z1 = zs[-1] + dt * (np.sin(xs[-1]) - a[1] * zs[-1])
        
        # Второй этап расчета
        z2 = (z1 + h2 * np.sin(xs[-1])) / (1 + h2 * a[1])
        y2 = (y1 + h2 * np.sin(z2)) / (1 + h2 * a[1])
        x2 = (x1 + h2 * np.sin(y2)) / (1 + h2 * a[1])
        
        xs.append(x2)
        ys.append(y2)
        zs.append(z2)
    return np.array([xs, ys, zs])

# 4. SIMPLIEST_MEGASTABLE
def megastable_discrete(steps, dt, init, a, s=0.5):
    h1 = s * dt
    h2 = (1 - s) * dt
    x, y = init[:2]  # Берем только x и y из начальных условий
    xs = [x]
    ys = [y]
    
    for _ in range(steps):
        # Первый этап расчета
        x1 = xs[-1] + h1 * (-ys[-1])
        y1 = ys[-1] + h1 * (a[1] * xs[-1] + np.sin(ys[-1]))
        
        # Сохраняем значение y для следующего шага
        z = y1
        
        # Второй этап расчета (повторяем вычисление y дважды, как в исходном коде)
        y2 = z + h2 * (a[1] * xs[-1] + np.sin(y1))
        y2 = z + h2 * (a[1] * xs[-1] + np.sin(y2))  # Повторное вычисление
        x2 = xs[-1] + h2 * (-y2)
        
        xs.append(x2)
        ys.append(y2)
    
    # Для совместимости с 3D-графиком добавляем третью координату (нули)
    zs = np.zeros_like(xs)
    return np.array([xs, ys, zs])

# Параметры моделирования
steps = 100000
dt = 0.01
init = [1.0, 1.0, 1.0]

# Параметры для CHAMELEON_MODEL
chameleon_params = [0.5, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0]  # s, a1...a6

# Параметры для ROSSLER_MODEL
rossler_params = [0.0, 0.2, 0.2, 5.7]  # s, a1, a2, a3

# Параметры для SYSTEM_FOR_BASINS
basin_params = [0.5, 0.2]  # s, a1

# Параметры для SIMPLIEST_MEGASTABLE
megastable_params = [0.5, 1.0]  # s, a1

# Создаем фигуру с четырьмя подграфиками (2x2 сетка)
fig = plt.figure(figsize=(14, 10))

# 1. CHAMELEON_MODEL
result_chameleon = chameleon_discrete(steps, dt, init, chameleon_params)
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(result_chameleon[0], result_chameleon[1], result_chameleon[2], lw=0.5)
ax1.set_title("CHAMELEON_MODEL")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# 2. ROSSLER_MODEL
result_rossler = rossler_discrete(steps, dt, init, rossler_params)
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(result_rossler[0], result_rossler[1], result_rossler[2], lw=0.5)
ax2.set_title("ROSSLER_MODEL")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# 3. SYSTEM_FOR_BASINS
result_basin = basin_discrete(steps, dt, init, basin_params)
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(result_basin[0], result_basin[1], result_basin[2], lw=0.5)
ax3.set_title("SYSTEM_FOR_BASINS")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

# 4. SIMPLIEST_MEGASTABLE
result_megastable = megastable_discrete(steps, dt, init, megastable_params)
ax4 = fig.add_subplot(224)  # 2D график для двумерной системы
ax4.plot(result_megastable[0], result_megastable[1], lw=0.5)
ax4.set_title("SIMPLIEST_MEGASTABLE")
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True)

plt.tight_layout()
plt.show()