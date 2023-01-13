import os
import scipy
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\Trees')
no_trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\NoTrees')
trees = trees[:4400]
no_trees = no_trees[:4400]
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
        print(next(a), ' Функция стоимости: ', stoim(y, hipoteza(theta, x)))
    res = scipy.optimize.minimize(stoim_J, theta, method='BFGS', jac=proizv_df, callback=call)
    theta = res.x
    return theta


def count(y, h):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == h[i]:
            if y[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if y[i] == 0:
                false_positive += 1
            else:
                false_negative += 1
        i += 1
    return true_positive, true_negative, false_positive, false_negative


def f_score(h, y, j):
    F_p = []
    x = [0.1*i for i in range(11)]
    for x_i in x:
        h_p = (h >= x_i)*1
        t_p, t_n, f_p, f_n = count(y, h_p)
        P = t_p / (f_p + t_p) if (f_p + t_p) != 0 else 0
        R = t_p / (f_n + t_p) if (f_n + t_p) != 0 else 0
        F = 2*P*R/(P+R) if (P + R) != 0 else 0
        print(f'Порог: {x_i}; P: {P}; R: {R}; F: {F};')
        print(f't_p, t_n, f_p, f_n = {count(y, h_p)} \n')
        F_p.append(F)
    print('оптимальный порог: \n', x[F_p.index(max(F_p))], '\n')
    plt.figure(j)
    plt.plot(x, F_p)
    plt.draw()
    return x[F_p.index(max(F_p))]


#Обучение
study = np.hstack((np.ones((len(study), 1)), study))
study = study.astype('float64')/255
theta1 = grad(study, Y_s)
print('Коэффициенты гипотезы: \n \n', theta1, '\n')

#Определяем порог с помощью кросвалидации
cross_validation = np.hstack((np.ones((len(cross_validation), 1)), cross_validation))
cross_validation = cross_validation.astype('float64')/255
h_cv = hipoteza(theta1, cross_validation)
h_cv_05 = (h_cv >= 0.5)*1
#print('Гипотеза кросвалидации: \n', (hipoteza(theta1, cross_validation)))
#print('Реальные данные: \n', Y_cv)
true_positive_cv, true_negative_cv, false_positive_cv, false_negative_cv = count(Y_cv, h_cv_05)
if true_positive_cv == 0:
    print('Нет ни единого правильно значения, повторите программу')
else:
    P = true_positive_cv/(false_positive_cv + true_positive_cv)
    R = true_positive_cv/(false_negative_cv + true_positive_cv)
    F = 2 * P * R / (P + R)
    print(f'Точность: {P} \n Полнота: {R} \n F-score(0.5): {F} \n')
    P_min = f_score(h_cv, Y_cv, 1)

#Тестирование
test = np.hstack((np.ones((len(test), 1)), test))
test = test.astype('float64')/255
h_t = (hipoteza(theta1, test) >= P_min)*1
true_positive_t, true_negative_t, false_positive_t, false_negative_t = count(Y_t, h_t)
print('Процент ошибок при данных для тестирования:', (false_negative_t+false_positive_t)*100/len(Y_t), '%')
if true_positive_t == 0:
    print('Нет ни единого правильно значения, повторите программу')
else:
    P = true_positive_t/(false_positive_t + true_positive_t)
    R = true_positive_t/(false_negative_t + true_positive_t)
    F = 2 * P * R / (P + R)
    print(f'F-score({P_min}) для данных тестирования: {F}')

#Проверка на данных для обучения
h_s = hipoteza(theta1, study)
h_s_opt = (h_s >= P_min)*1
true_positive_st, true_negative_st, false_positive_st, false_negative_st = count(Y_s, h_s_opt)
print('Процент ошибок при данных для обучения:', (false_negative_st+false_positive_st)*100/len(Y_s), '%')
f_score(h_s, Y_s, 2)
plt.show()

print(stoim(study, hipoteza(theta1, study)))
