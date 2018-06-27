# ???????
# ???????

import sys
import gluonbook as gb

from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

X = nd.ones(shape = (6, 8))
X[:, 2:6] = 0

K = nd.array([[1, -1]])

Y = corr2d(X, K)

conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        if i % 2 == 1:
            print('batch %d, loss %.3f' % (i, loss.sum().asscalar()))
        loss.backward()

        print(conv2d.weight.data()[:])
        print(conv2d.weight.grad())

        conv2d.weight.data()[:] -= 3e-2  * conv2d.weight.grad()

