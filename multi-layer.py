
from mxnet import nd, autograd, gluon
from mxnet.gluon import loss
from mxnet.gluon import data as gdata

def transform(feature, label):
    return feature.astype('float32') / 255, label.astype('float32')
mnist_train = gdata.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gdata.vision.FashionMNIST(train=False, transform=transform)
batch_size = 200
train_iter = gdata.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = gdata.DataLoader(mnist_test, batch_size, shuffle=False)


num_inputs = 784
num_outputs = 10
num_hiddens = 256

w1 = nd.random.normal(scale = 0.01, shape = (num_inputs, num_hiddens))
b1 = nd.zeros(shape = num_hiddens)

w2 = nd.random.normal(scale = 0.01, shape = (num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)

def softmax(X):
    exp = X.exp()
    partition = exp.sum(axis=1, keepdims = True)
    return exp / partition

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, w1) + b1)
    return softmax(nd.dot(H, w2) + b2)

#loss = gloss.SoftmaxCrossEntropyLoss()
def loss(y_hat, y):
    return -nd.pick(y_hat.log(), y)

def accuarcy(y_hat, y):
    return(y_hat.argmax(axis=1) == y).mean().asscalar()

def test_accuarcy(net, test_iter):
    acc_sum = 0
    for X, y in test_iter:
        y_hat = net(X)
        acc_sum += accuarcy(y_hat, y)
    acc = acc_sum / len(test_iter)
    return acc

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


num_epochs = 5
lr = 0.5

def train_cpu(net, train_iter, test_iter, lr, num_epochs, batch_size, params, trainer = None):

    for epoch in range(1, num_epochs + 1):
        train_sum_loss = 0
        train_sum_acc = 0

        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)

            train_sum_loss += l.mean().asscalar()
            train_sum_acc += accuarcy(y_hat, y)

        print("epoch: %d, loss: %f, train_acc: %f"%(epoch, train_sum_loss/len(train_iter),
                                                    train_sum_acc/len(train_iter)))

        test_acc = test_accuarcy(net, test_iter)
        print("test_acc: %f"% test_acc)


train_cpu(net, train_iter, test_iter, lr, num_epochs, batch_size, params, trainer = None)


