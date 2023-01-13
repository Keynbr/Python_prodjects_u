import numpy as np
from matplotlib import pyplot as plt
with open('kc_house_data.csv') as f:
   lines = f.readlines()
l_str = np.empty((21613, 19))
i = 0
for x in lines[1:]:
   str = x.split(',')
   str1 = []
   for j in str:
      if j[0] == '"':
         a = ''
         for k in range(len(j)-2):
            a += j[k+1]
         j = a
      str1.append(j)
   l_str[i] = str1[2:]
   i += 1
#yeprint(l_str)
#y = (100*5 + 17) % 5 + 1
#print(y)
price = l_str[:, 0]
price.shape = (len(price),1)
param = l_str[:, 9]
param.shape = (len(param),)
A = np.empty((21613, 5))
for j in range(5):
   A[:, j] = np.power(param, j)
A = A[:,1:]
def norm(x):
   normir = np.empty((3, 4))
   for N in range(4):
      srAr = np.mean(x, axis=0)
      Pmax = np.max(x, axis=0)
      Pmin = np.min(x, axis=0)
      normir[0, :] = srAr
      normir[1, :] = Pmax
      normir[2, :] = Pmin
      x[:, N] = (x[:, N]-srAr[N])/(Pmax[N]-Pmin[N])
   return x, normir
param0, normir = norm(A)
X = np.ones((len(param0), 1))
param1 = np.hstack((X, param0))

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
   lyam = 1
   arr = []
   arr1 = []
   np.append(arr1, 1)
   st1 = 100
   st2 = 1
   k = 0
   while (st1 - st2) > 10.:#(0.001)/st2:
      dJ = proizv(x, theta, y)
      st1 = stoimosti(x, theta, y)
      arr.append(st1)
      a = (theta.copy())
      arr1.append(a)
      theta -= lyam * dJ
      st2 = stoimosti(x, theta, y)
      k += 1
      if k % 100 == 0:
         print(st1-st2)
         break
   return arr1, arr

thetas, stoim= grad(param1, np.array([[0.],[0.],[0.],[0.],[0.]]), price)
#print(thetas)
plt.plot(range(len(stoim)), stoim)
plt.grid(True)
plt.title('Зависимость функции стоимости от количества итераций')
plt.show()
arr = []
for i in range(5):
   for x in thetas:
      arr.append(float(x[i]))
   #np.array(arr)
   plt.plot(range(len(thetas)), arr, c=np.random.rand(3,))
   plt.grid(True)
   #arr.clear()
plt.title(f'Зависимость коэффициентов тет от количества итераций')
plt.show()

#print(param1)
#param.shape = (len(param),1)
#plt.scatter(param, price)
#plt.show()
x = np.arange(0, 14, 0.1)
xx = x
x.shape = (len(x),)
B = np.empty((len(x), 5))
for j in range(5):
   B[:, j] = np.power(x, j)
#B = B[:,1:]
for i in range(1, 5):
  B[:,i] = (B[:,i]-normir[0, i])/(normir[1, i]-normir[2, i])
#ONE = np.ones((len(B), 1))
#B = np.hstack((ONE, B))

h = B @ thetas[-1]
plt.plot(xx, h)#gipoteza(B, np.array([[540088.14176653], [-737231.45784021], [56026.86112513], [1094731.86627888], [2031187.06653983]])))
#np.array(thetas[-1])))
#plt.scatter(l_str[:, 9], price)
plt.plot(xx, h)
plt.grid(True)
plt.title('Зависимость функции стоимости от параметров')
plt.show()
#print(np.array(thetas[-1]))
#print(xx)