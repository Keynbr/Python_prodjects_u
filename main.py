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
x = (100*5 + 17) % 11 + 1
y = (100*5 + 17) % 3 + 12
exam = l_str[:, y]
param = l_str[:, x]

def gipoteza(x,theta):
   X = np.empty((len(x), 2))
   X[:, 0] = 1
   X[:, 1] = x[:, 0]
   h = X @ theta
   return h

def stoimosti(x,theta,y):
   h = gipoteza(x, theta)
   N = len(y)
   J = ((h-y)**2).sum()/(2*N)
   return J

def proizv1(x,theta,y):
   h = gipoteza(x, theta)
   J_proizv1 = np.empty(2)
   J_proizv1[0] = (h - y).sum()
   J_proizv1[1] = ((h - y)*x).sum()
   J_proizv1 = J_proizv1 / len(y)
   return J_proizv1

def grad(x,theta,y):
   lyam = 0.01
   k = 0
   arr = np.empty((1000, 4))
   while k < 1000:
      #print(k, x, y, p(x), s(y), f(x, y))
      dJ = proizv1(x, theta, y)
      st = stoimosti(x, theta, y)
      theta[0] -= lyam * dJ[0]
      theta[1] -= lyam * dJ[1]
      arr[k,0] = k
      arr[k,1] = st
      arr[k,2] = theta[0]
      arr[k,3] = theta[1]
      k += 1
   return theta, arr

param.shape = (len(param),1)
exam.shape = (len(exam), 1)
#print(grad(param, np.array([[0.],[0.]]), exam))
thetas, hod = grad(param, np.array([[0.],[0.]]), exam)
plt.subplot(221)
plt.scatter(param, exam)
xx = np.arange(param.min(), param.max())
h1 = gipoteza(param,thetas)
plt.plot(param,h1)
plt.subplot(222)
plt.plot(hod[:,0],hod[:,1])
plt.subplot(223)
plt.plot(hod[:,0],hod[:,2])
plt.subplot(224)
plt.plot(hod[:,0],hod[:,3])
plt.show()