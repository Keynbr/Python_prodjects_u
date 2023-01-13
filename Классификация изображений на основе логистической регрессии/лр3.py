import os
import numpy as np
import skimage as ski
trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject21\\Trees')
no_trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject21\\NoTrees')
A = trees[:100]
B = no_trees[:100]
n = 100
m = 2048
Tr = np.full((n, m), 0)
i, j = 0, 0
for x in A:
  x = ski.io.imread(f'Trees\\{x}')
  sr = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2])/3
  img = sr[::2, :].flatten()
  Tr[i, :] = img
  i += 1
N_tr = np.full((n, m), 0)
for x in B:
   x = ski.io.imread(f'NoTrees\\{x}')
   sr = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3
   img = sr[::2, :].flatten()
   N_tr[j, :] = img
   j += 1
X = np.vstack((Tr, N_tr))
Y = np.vstack((np.zeros((n, 1)), np.ones((n, 1))))
perm = np.random.permutation(200)
X = X[perm]
Y = Y[perm]
cross_validation = X[:20]
test = X[20:40]
study = X[40:]
Y_cv = Y[:20]
Y_t = Y[20:40]
Y_s = Y[40:]


def hipoteza(theta, x):
   h = x @ theta
   return logical_H(h)


def logical_H(h):
   return 1/(1 + np.exp(-h))


def stoim(y, h):
   return (-1/len(y))*(y*np.log(h)+(1-y)*np.log(1-h)).sum()


def proizv(h, y, X):
   return (1/len(y))*((h-y).T @ X).T


def grad(x, y):
   theta = np.zeros((x.shape[1], 1))
   j0, j1, i = 0, 0, 0
   while (j1 - j0 > 10**(-7)) or i < 3:
       j1 = j0
       h = hipoteza(theta, x)
       j0 = stoim(y, h)
       #print(j0)
       pr = proizv(h, y, x)
       theta -= pr*0.01
       i += 1
   return theta


study = np.hstack((np.ones((160, 1)), study))
study = study.astype('float64')/255
theta1 = grad(study, Y_s)
print('Коэффициенты гипотезы: \n', theta1)
test = np.hstack((np.ones((20, 1)), test))
test = test.astype('float64')/255
test_y = hipoteza(theta1, test)
test_y = (test_y >= 0.5)*1
net = 0
for i in range(len(Y_t)-1):
   if test_y[i] != Y_t[i]:
       net += 1
print('Процент ошибок:', net*100/len(Y_t), '%')
study_y = hipoteza(theta1, study)
study_y = (study_y >= 0.5)*1
net = 0
for i in range(len(Y_s)-1):
   if study_y[i] != Y_s[i]:
       net += 1
print('Процент ошибок:', net*100/len(Y_s), '%')
