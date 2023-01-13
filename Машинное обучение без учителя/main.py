import os
import scipy
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib import colors as clr
trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\Trees')
no_trees = os.listdir('C:\\Users\\ангелина\\PycharmProjects\\pythonProject22\\NoTrees')
trees = trees[:2000]
no_trees = no_trees[:2000]
n = len(trees)
m = 300
Tr = np.full((n, m), 0.)
i, j = 0, 0
for x in trees:
   x = ski.io.imread(f'Trees\\{x}')
   img = x[:10,:10].flatten().astype('float64') / 255.
   Tr[i, :] = img
   i += 1

plt.title(f'Cжатое изображение')
N_tr = np.full((n, m), 0.)
for x in no_trees:
    x = ski.io.imread(f'NoTrees\\{x}')
    img = x[:10,:10].flatten().astype('float64') / 255.
    N_tr[j, :] = img
    j += 1


X = np.vstack((Tr, N_tr))
Y = np.vstack((np.ones((n, 1)), np.zeros((n, 1))))


sr = np.mean(X)
normir = X - sr
sigma = normir.T @ normir / len(normir)
U, a, b = np.linalg.svd(sigma)
U_r = U[:, 0:2]
z = normir @ U_r
s = z @ U_r.T
s = s + sr


img = np.reshape(X[0, :], (10, 10, 3))
plt.imshow(img)
plt.title(f'Исходное изображение')
plt.show()
img = np.reshape(s[0, :], (10, 10, 3))
plt.imshow(img)
plt.title(f'Cжатое изображение')
plt.show()

plt.scatter(z[:, 0], z[:, 1])
plt.show()
K = 4
centre = z[:K]
while True:
    dists = np.full((len(z), len(centre)), 0.)
    np.sum((z - centre)**2, axis=1)
    for j in range(len(centre)):
        dists[:, j] = np.sum((z - centre)**2, axis=1)
    idx = np.array(())
    for d in dists:
        idx = np.append(idx, np.argmin(d, axis = 0))

    x = centre.copy()
    for i in range(K):
        a = z[idx == i, :]
        aa = np.array((np.mean(a[:, 0]), np.mean(a[:, 1])))
        x[i, 0], x[i, 1] = aa[0], aa[1]

    if np.array_equal(x, centre):
        break
    centre = x

plt.scatter(z[:, 0], z[:, 1], c = idx, cmap = clr.ListedColormap(["k", "m", "c", "y"]))
plt.show()
