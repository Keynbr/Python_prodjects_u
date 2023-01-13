import os
import scipy
import numpy as np
import skimage as ski
trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject21\\Trees')
no_trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject21\\NoTrees')
A = trees[:4400]
B = no_trees[:4400]
n = len(trees)
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
Y = np.vstack((np.ones((n, 1)), np.zeros((n, 1))))

perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]
cross_validation = X[:int(0.2*len(X))]
test = X[int(0.2*len(X)):int(0.4*len(X))]
study = X[int(0.4*len(X)):]
Y_cv = Y[:int(0.2*len(X))]
Y_t = Y[int(0.2*len(X)):int(0.4*len(X))]
Y_s = Y[int(0.4*len(X)):]


def hipoteza(theta, x):
    theta_ = theta.view()
    theta_.shape = (len(theta_), 1)
    h = x @ theta_
    return logical_H(h)


def logical_H(h):
    return 1/(1 + np.exp(-h))


def stoim(y, h):
    return (-1/len(y))*(y*np.log(h)+(1-y)*np.log(1-h)).sum()


def proizv(h, y, x):
    return (1/len(y))*((h-y).T @ x).T


def grad(x, y):
    theta = np.zeros((x.shape[1], ))
    def stoim_J(theta):
        return stoim(y, hipoteza(theta, x))
    def proizv_df(theta):
        pr = proizv(hipoteza(theta, x), y, x)
        return pr.flatten()
    def func():
        i = 0
        while True:
            i += 1
            yield i
        return
    a = func()
    def call(theta):
        print(next(a), ' ', stoim(y, hipoteza(theta, x)))
    res = scipy.optimize.minimize(stoim_J, theta, method='BFGS', jac=proizv_df, callback=call)
    theta = res.x
    return theta


study = np.hstack((np.ones((len(study), 1)), study))
study = study.astype('float64')/255
theta1 = grad(study, Y_s)
print('Коэффициенты гипотезы: \n', theta1)

test = np.hstack((np.ones((len(test), 1)), test))
test = test.astype('float64')/255
test_y = hipoteza(theta1, test)
test_y = (test_y > 0.5)*1

net = 0
for i in range(len(Y_t)-1):
    if test_y[i] != Y_t[i]:
        net += 1
print('Процент ошибок на тестировании:', net*100/len(Y_t), '%')

study_y = hipoteza(theta1, study)
study_y = (study_y >= 0.5)*1
net = 0
for i in range(len(Y_s)-1):
    if study_y[i] != Y_s[i]:
        net += 1
print('Процент ошибок на обучении:', net*100/len(Y_s), '%')
