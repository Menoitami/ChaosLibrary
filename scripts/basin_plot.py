import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
# --- Настройки ---
FONTsize = 12
start_time = time.time()

workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_dir = os.path.join(workspace_root, 'workspace\\basins')
path = os.path.join(input_dir, 'basinsOfAttraction_test.csv')
path2 = os.path.join(input_dir, 'basinsOfAttraction_test_old.csv')
def load_data(filepath):
    """Load data from CSV files and process it similar to MATLAB's csvread, skipping first two lines"""
    data = []
    with open(filepath, 'r') as f:
        # Skip first two lines
        next(f)
        next(f)
        for line in f:
            row = [float(val.strip()) for val in line.strip().split(',') if val.strip()]
            if row:
                data.append(row)
    return np.array(data)

def load_csv(path):
    return pd.read_csv(path, delimiter=',', skiprows=2, header=None).to_numpy()

# Load data
idx = load_data(path)  # basins
a = load_data(path + "_1.csv")  # avgPeaks
b = load_data(path + "_2.csv")  # avgInterval
c = load_data(path + "_3.csv")  # helphulArray

# --- Обработка массивов ---
x = np.linspace(a[0,0], a[0,1], a.shape[1])
a = a[1:]
y = np.linspace(a[0,0], a[0,1], a.shape[1])
a = a[1:]
x = x[:-1]
y = y[:-1]
a = a[:,:-1]
b = b[2:,:-1]
c = c[2:,:-1]
idx = idx[2:,:-1]

len_ = a.shape[0]
flag_NAN_a = False
flag_NAN_b = False

for i in range(len_):
    for j in range(len_):
        if a[i, j] == 999 or a[i, j] == 0:
            flag_NAN_a = True
            a[i, j] = np.nan
        if b[i, j] == 999 or b[i, j] == 0:
            flag_NAN_b = True
            b[i, j] = np.nan
        if b[i, j] == -1:
            b[i, j] = np.nan

min_a = np.nanmin(a)
max_a = np.nanmax(a)
delt_color_a = 0.005 * (max_a - min_a)

A = a.flatten()
B = b.flatten()
C = c.flatten()
labels = idx.flatten()
X = np.column_stack((A, B))
max_idx = int(np.nanmax(idx))
min_idx = int(np.nanmin(idx))
if max_idx > 50:
    max_idx = 50
N_colormap = 168

# --- Цветовые карты ---
cm = plt.cm.turbo(np.linspace(0, 1, N_colormap))
cm1 = plt.cm.turbo(np.linspace(0, 1, 256))

# --- Визуализация ---
fig1, ax1 = plt.subplots(figsize=(7.5, 6))
im1 = ax1.imshow(idx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
ax1.set_xlabel(r"$x(0)$", fontsize=22)
ax1.set_ylabel(r"$y(0)$", fontsize=22)
cb1 = plt.colorbar(im1, ax=ax1)
cb1.ax.tick_params(labelsize=FONTsize)
im1.set_clim([min_idx, max_idx+0.99])
ax1.tick_params(labelsize=18)
plt.tight_layout()

# --- Большая фигура с подграфиками ---
fig2, axs = plt.subplots(2, 3, figsize=(15, 7))

# Индексы
ax = axs[0,2]
im = ax.imshow(idx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
ax.set_title(r"$\geq$1 - stable, $\leq$-1 - fixed, 0 - unbound", fontsize=FONTsize)
ax.set_xlabel(r"$x(0)$", fontsize=FONTsize)
ax.set_ylabel(r"$y(0)$", fontsize=FONTsize)
cb = plt.colorbar(im, ax=ax)
cb.ax.tick_params(labelsize=FONTsize)
im.set_clim([min_idx, max_idx+0.99])
ax.tick_params(labelsize=FONTsize)

# mean Peak
a1 = axs[0,0]
im_a = a1.imshow(a, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
a1.set_title("mean Peak", fontsize=FONTsize)
a1.set_xlabel(r"$x$", fontsize=FONTsize)
a1.set_ylabel(r"$y$", fontsize=FONTsize)
cb_a = plt.colorbar(im_a, ax=a1)
cb_a.ax.tick_params(labelsize=FONTsize)
im_a.set_clim([min_a-delt_color_a, max_a])
a1.tick_params(labelsize=FONTsize)

# mean Interval
for i in range(len_):
    for j in range(len_):
        if b[i, j] < 0:
            b[i, j] = np.nan
min_b = np.nanmin(b)
max_b = np.nanmax(b)
delt_color_b = 0.005 * (max_b - min_b)
a2 = axs[0,1]
im_b = a2.imshow(b, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
a2.set_title("mean Interval", fontsize=FONTsize)
a2.set_xlabel(r"$x$", fontsize=FONTsize)
a2.set_ylabel(r"$y$", fontsize=FONTsize)
cb_b = plt.colorbar(im_b, ax=a2)
cb_b.ax.tick_params(labelsize=FONTsize)
a2.tick_params(labelsize=FONTsize)

# Scatter
ax4 = axs[1,0]
sc = ax4.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.turbo, s=30)
ax4.grid(True)
ax4.set_xlabel("mean Peak", fontsize=FONTsize)
ax4.set_ylabel("mean Interval", fontsize=FONTsize)
minX, maxX = np.nanmin(X[:,0]), np.nanmax(X[:,0])
deltX = 0.1 * (maxX - minX)
minY, maxY = np.nanmin(X[:,1]), np.nanmax(X[:,1])
deltY = 0.1 * (maxY - minY)
ax4.set_xlim([minX-deltX, maxX+deltX])
ax4.set_ylim([minY-deltY, maxY+deltY])

# Histogram2d
ax5 = axs[1,1]
hist = ax5.hist2d(X[:,0], X[:,1], bins=[25,25], cmap=plt.cm.turbo)
ax5.set_xlabel("mean Peak", fontsize=FONTsize)
ax5.set_ylabel("mean Interval", fontsize=FONTsize)
cb5 = plt.colorbar(hist[3], ax=ax5)
cb5.ax.tick_params(labelsize=FONTsize)
ax5.set_xlim([minX-deltX, maxX+deltX])
ax5.set_ylim([minY-deltY, maxY+deltY])

# c
ax6 = axs[1,2]
im_c = ax6.imshow(c, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
ax6.set_title("1 - stable, -1 - fixed, 0 - unbound", fontsize=FONTsize)
ax6.set_xlabel(r"$x$", fontsize=FONTsize)
ax6.set_ylabel(r"$y$", fontsize=FONTsize)
cb_c = plt.colorbar(im_c, ax=ax6)
cb_c.set_ticks([-0.6667, 0, 0.6667])
cb_c.set_ticklabels([-1, 0, 1])
im_c.set_clim([-1, 1])
ax6.tick_params(labelsize=FONTsize)

plt.tight_layout()

# --- Интегрирование системы (примерно как в MATLAB) ---
fig3, bx = plt.subplots(1,2, figsize=(18,6))

# IC вычисление (упрощённо, как в оригинале)
IC = np.zeros((max_idx+6, 3))
k = -5
attr_indeces = np.linspace(1, max_idx, max_idx)
for m in range(5):
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if idx[i, j] == k:
                IC[k+6, 0] = x[j]
                IC[k+6, 1] = y[i]
                k += 1
                if k == 0:
                    k += 1
            if k > max_idx:
                break

# Интегрирование
forXlim = [999999, -999999]
forYlim = [999999, -999999]
for kk in range(max_idx+4):
    PI = np.pi
    a_params = [0.5, 1.5, -0.27, -19, -3.5, -3.25, 8.04, 4]
    h = 0.01
    X = np.zeros(3)
    X[:2] = IC[kk,:2]
    X[2] = 0
    X_write = np.zeros((3,20000))
    for ii in range(100000):
        h1 = h * a_params[0]
        h2 = h * (1 - a_params[0])
        N = 3
        X1 = X.copy()
        k_arr = np.zeros((3,4))
        for j in range(4):
            sigma = a_params[3]*X1[0] + a_params[4]*X1[1] + a_params[5]*X1[2]
            delt1 = np.floor((sigma + 40.00325) / 2 / 40.00325)
            delt2 = sigma - delt1 * 2 * 40.00325
            sigma = delt2
            if abs(sigma) < a_params[6]:
                psi2 = a_params[1]*np.arctan(sigma) + a_params[2]*sigma
            elif sigma >= a_params[6]:
                psi1 = a_params[7]*(a_params[1]*np.arctan((sigma-a_params[6])/a_params[7]) + a_params[2]*((sigma-a_params[6])/a_params[7]))
                psi2 = psi1
            elif sigma <= -a_params[6]:
                psi1 = a_params[7]*(a_params[1]*np.arctan((-sigma-a_params[6])/a_params[7]) + a_params[2]*((-sigma-a_params[6])/a_params[7]))
                psi2 = -psi1
            k_arr[0,j] = X1[1]
            k_arr[1,j] = X1[2]
            k_arr[2,j] = -X1[1] - X1[2] + psi2
            if j == 3:
                X += h * (k_arr[:,0] + 2*k_arr[:,1] + 2*k_arr[:,2] + k_arr[:,3]) / 6
            elif j == 2:
                X1 = X + h * k_arr[:,j]
            else:
                X1 = X + 0.5 * h * k_arr[:,j]
        if ii > 80000:
            X_write[:,ii-80000] = X
    decimator = 5
    if np.min(X_write[0,::decimator]) < forXlim[0]:
        forXlim[0] = np.min(X_write[0,::decimator])
    if np.max(X_write[0,::decimator]) > forXlim[1]:
        forXlim[1] = np.max(X_write[0,::decimator])
    if np.min(X_write[1,::decimator]) < forYlim[0]:
        forYlim[0] = np.min(X_write[1,::decimator])
    if np.max(X_write[1,::decimator]) > forYlim[1]:
        forYlim[1] = np.max(X_write[1,::decimator])
    if kk <= 5:
        bx[0].plot(X_write[0,::decimator], X_write[1,::decimator], '.', color=cm[kk], linewidth=5.0, markersize=10.0)
    else:
        bx[0].plot(X_write[0,::decimator], X_write[1,::decimator], '-', color=cm[kk], linewidth=2.0, alpha=0.4)
bx[0].grid(True)
bx[0].set_xlabel(r"$x$", fontsize=FONTsize)
bx[0].set_ylabel(r"$z$", fontsize=FONTsize)
bx[0].set_xlim([forXlim[0]*0.98, forXlim[1]*1.02])
bx[0].set_ylim([forYlim[0]*0.98, forYlim[1]*1.02])

# Индексы
im_bx2 = bx[1].imshow(idx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto')
bx[1].set_xlabel(r"$x(0)$", fontsize=FONTsize)
bx[1].set_ylabel(r"$z(0)$", fontsize=FONTsize)
cb_bx2 = plt.colorbar(im_bx2, ax=bx[1])
im_bx2.set_clim([min_idx, max_idx+0.99])
bx[1].tick_params(labelsize=FONTsize)

plt.tight_layout()

print(f"Время выполнения: {time.time() - start_time:.2f} сек")
plt.show() 