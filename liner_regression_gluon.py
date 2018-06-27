import sys
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import nn, data as gdata, loss as gloss

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = nd.random_normal(scale = 1, shape = (num_examples, num_inputs))
print('features shape', features.shape)

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += labels + nd.random_normal(scale = 0.01, shape=labels.shape)
print('labels shape: ', labels.shape)

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)


for X, y in data_iter:
    print('X shape:', X.shape, 'y shape', y.shape )
    break

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epochs = 5

for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)

        l.backwards()
        trainer.step(batch_size)
    print('epoch %d loss %f'%(epoch, loss(net(features), labels).mean().asnumpy()))