import numpy as np
from matplotlib import pyplot as plt
with open('student-mat-2.csv') as f:
   lines = f.readlines()
l_str = np.empty((395, 15))
i = 0
for x in lines[1:]:
   str = x.split(',')
   l_str[i] = str
   i += 1
print(l_str)
y = (100*5 + 17) % 3 + 12
exam = l_str[:, y]
exam.shape = (len(exam),1)
param = l_str[:, 1:12]
def norm(x):
   normir = np.empty((3, 11))
   for N in range(11):
      srAr = np.mean(x, axis=0)
      Pmax = np.max(x, axis=0)
      Pmin = np.min(x, axis=0)
      normir[0, :] = srAr
      normir[1, :] = Pmax
      normir[2, :] = Pmin
      x[:, N] = (x[:, N]-srAr[N])/(Pmax[N]-Pmin[N])
   return x, normir
param, normir = norm(param)
X = np.ones((len(param), 1))
param1 = np.hstack((X, param))

def gipoteza(x,theta):
   h = x @ theta
   return h

def stoimosti(x,theta,y):
   h = gipoteza(x, theta)
   N = len(y)
   J = ((h-y)**2).sum()/(2*N)
   return J

def proizv(x,theta,y):
   h = gipoteza(x, theta)
   x1 = x.transpose()
   J_proizv = x1 @ (h - y) / len(y)
   return J_proizv

def grad(x,theta,y):
   lyam = 0.01
   arr = []
   arr1 = []
   np.append(arr1, 1)
   st1 = 2
   st2 = 1
   k = 0
   while st1 - st2 > (0.001) / st2:
      dJ = proizv(x, theta, y)
      st1 = stoimosti(x, theta, y)
      arr.append(st1)
      a = (theta.copy())
      arr1.append(a)
      theta -= lyam * dJ
      st2 = stoimosti(x, theta, y)
      k += 1
   return arr1, arr

thetas, stoim= grad(param1, np.array([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]), exam)
print()
plt.plot(range(len(stoim)), stoim)
plt.grid(True)
plt.title('Зависимость функции стоимости от количества итераций')
plt.show()
arr = []
for i in range(11):
   for x in thetas:
      arr.append(float(x[i]))
   np.array(arr)
   plt.plot(range(len(stoim)), arr, c=np.random.rand(3,))
   plt.grid(True)
   arr.clear()
plt.title(f'Зависимость коэффициентов тет от количества итераций')
plt.show()

k = 100*5 + 17
X = [k % 5,  3*k % 5,  k % 4 + 1, 3*k % 4 + 1, 4*k % 5 + 1, 2*k % 5 + 1, (5 - 3*k) % 5 + 1, 3*k % 5 + 1, k % 5 + 1, 2*k % 5 + 1,  int(k * k / 1000)]

for i in range(11):
   X[i] = (X[i]-normir[0, i])/(normir[1, i]-normir[2, i])
   i += 1
X = np.hstack(([1], X))
theta = thetas[-1]

print('ученик получит оценку:', X @ theta)