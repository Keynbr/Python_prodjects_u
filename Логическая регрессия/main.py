import numpy as np
from matplotlib import pyplot as plt
with open('cancer.csv') as f:
    lines = f.readlines()
a_m, s_m, diagnosis = [], [], []
for x in lines[1:]:
    x = x.replace('M', '1')
    x = x.replace('B', '0')
    stlb = x.split(',')
    a_m.append(float(stlb[5]))
    s_m.append(float(stlb[10]))
    diagnosis.append(int(stlb[1]))
a_m = np.array(a_m)
s_m = np.array(s_m)
diagnosis = np.array(diagnosis)

plt.scatter(a_m[diagnosis == 0], s_m[diagnosis == 0])
plt.scatter(a_m[diagnosis == 1], s_m[diagnosis == 1])
plt.legend(['доброкачественная', 'злокачественная'])
plt.grid(True)
plt.title('График исходных данных')
plt.show()

diagnosis.shape = (len(diagnosis), 1)
def mashtabirovanie(x):
    normir = np.empty((3, 2))
    srAr = np.mean(x, axis=0)
    Pmax = np.max(x, axis=0)
    Pmin = np.min(x, axis=0)
    normir[0, :] = srAr
    normir[1, :] = Pmax
    normir[2, :] = Pmin
    return normir
def normirovka(x, norm):
    for i in range(2):
        x[:, i] = (x[:, i] - norm[0, i]) / (norm[1, i] - norm[2, i])
a_m.shape = (len(a_m), 1)
s_m.shape = (len(s_m), 1)
x = np.hstack((a_m, s_m))
normir = mashtabirovanie(x)
normirovka(x, normir)
one = np.ones((len(a_m), 1))
x = np.hstack((one, x))

def gipoteza(x,theta):
   XO = x @ theta
   h = 1 / (1 + np.exp(-XO))
   return h

def stoimosti(x,theta,y):
   h = gipoteza(x, theta)
   N = len(y)
   J = - (y*np.log(h)+(1-y)*np.log(1-h)).sum()/(N)
   return J

def proizv(x,theta,y):
   h = gipoteza(x, theta)
   x1 = x.transpose()
   J_proizv = x1 @ (h - y) / len(y)
   return J_proizv

def grad(x,theta,y):
   lyam = 1
   arr = []
   arr1 = []
   np.append(arr1, 1)
   st1 = 100
   st2 = 1
   k = 0
   while (st1 - st2) > (0.0000001)/st2:
      dJ = proizv(x, theta, y)
      st1 = stoimosti(x, theta, y)
      arr.append(st1)
      a = (theta.copy())
      arr1.append(a)
      theta -= lyam * dJ
      st2 = stoimosti(x, theta, y)
      k += 1
      if k  == 10000:
         break
   return arr1, arr

thetas, stoim= grad(x, np.array([[0.], [0.], [0.]]), diagnosis)
plt.plot(range(len(stoim)), stoim)
plt.grid(True)
plt.title('Зависимость функции стоимости от количества итераций')
plt.show()
arr = []
for i in range(3):
   for x in thetas:
      arr.append(float(x[i]))
   np.array(arr)
   plt.plot(range(len(thetas)), arr, c=np.random.rand(3,))
   plt.grid(True)
   arr.clear()
plt.title(f'Зависимость коэффициентов тет от количества итераций')
plt.show()

grid_n = 100 # Сетка 100х100 элементов
# Считаем по 100 значений каждого параметра в диапазоне данных
a_m_p = np.linspace(min(a_m), max(a_m), grid_n)
s_m_p = np.linspace(min(s_m), max(s_m), grid_n)
# meshgrid создаёт две матрицы сетки значений параметров
x1_grid, x2_grid = np.meshgrid(a_m_p, s_m_p)
# Создаём матрицу параметров для нахождения гипотез
X = np.empty((x1_grid.size, 2))
#X[:, 0] = 1
X[:, 0] = x1_grid.flatten()
X[:, 1] = x2_grid.flatten()
normirovka(X, normir)
one = np.ones((len(X), 1))
X = np.hstack((one, X))
# Находим гипотезу для каждого набора параметров
h = gipoteza(X, thetas[-1])
# Делаем из вектора гипотез сетку гипотез
h_grid = np.reshape(h, (grid_n, grid_n))
# Строим график гипотез с одним уровнем (0.5)
plt.scatter(a_m[diagnosis == 0], s_m[diagnosis == 0])
plt.scatter(a_m[diagnosis == 1], s_m[diagnosis == 1])
plt.legend(['доброкачественная', 'злокачественная'])
plt.grid(True)
plt.title(f'Граница решения')
plt.contour(x1_grid, x2_grid, h_grid, levels=[0.5])
plt.show()
