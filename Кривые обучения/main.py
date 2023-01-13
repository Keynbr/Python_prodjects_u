import os
import scipy
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\Trees')
no_trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\NoTrees')
trees = trees[:int((3000+7*200)/2)]
no_trees = no_trees[:int((3000+7*200)/2)]
n = len(trees)
m = 2048
Tr = np.full((n, m), 0)
i, j = 0, 0
for x in trees:
   x = ski.io.imread(f'Trees\\{x}')
   sr = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2])/3
   img = sr[::2, :].flatten()
   Tr[i, :] = img
   i += 1
N_tr = np.full((n, m), 0)
for x in no_trees:
    x = ski.io.imread(f'NoTrees\\{x}')
    sr = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3
    img = sr[::2, :].flatten()
    N_tr[j, :] = img
    j += 1
X = np.vstack((Tr, N_tr))
Y = np.vstack((np.ones((n, 1)), np.zeros((n, 1))))
#Перемешивание
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]
cross_validation = X[:int(0.5*len(X))]
study = X[int(0.5*len(X)):]
Y_cv = Y[:int(0.5*len(X))]
Y_s = Y[int(0.5*len(X)):]


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
        print(next(a), ' Функция стоимости: ', stoim(y, hipoteza(theta, x)))
    res = scipy.optimize.minimize(stoim_J, theta, method='BFGS', jac=proizv_df, callback=call)
    theta = res.x
    return theta

#Обучение
study = np.hstack((np.ones((len(study), 1)), study))
study = study.astype('float64')/255
#theta1 = grad(study, Y_s)
#print('Коэффициенты гипотезы: \n \n', theta1, '\n')


#Кривые обучения
cross_validation = np.hstack((np.ones((len(cross_validation), 1)), cross_validation))
cross_validation = cross_validation.astype('float64')/255
def graph_learning(study, cross_validation, Y_s, Y_cv):
    j = 0
    J_st, J_cv = np.array([]), np.array([])
    x = [100 * i for i in range(int((3000+7*200)/200))]
    x[0] += 1
    for i in x:
        theta = grad(study[:i], Y_s[:i])
        J_st = np.append(J_st, stoim(Y_s[:i], hipoteza(theta, study[:i])))
        J_cv = np.append(J_cv, stoim(Y_cv[:i], hipoteza(theta, cross_validation[:i])))
        J_st[np.isnan(J_st)] = 0
        J_cv[np.isinf(J_cv)] = 10
        J_cv[np.isnan(J_cv)] = 0
        j += 1
        print(f'Итерация: {j} \n J_cv = {J_cv} \n J_st = {J_st}')
    plt.plot(x, J_cv)
    plt.plot(x, J_st)

    plt.draw()
graph_learning(study, cross_validation, Y_s, Y_cv)
plt.show()