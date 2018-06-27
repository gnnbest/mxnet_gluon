
from mxnet import init, nd
from mxnet.gluon import nn
'''
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

print(net[0].weight.shape, net[1].weight.shape)

x = nd.random.uniform(shape = (2, 20))
y = net(x)
print(net[0].weight.shape, net[1].weight.shape)
print(net[0].weight.data())
'''


# share params
net = nn.Sequential()

shared = nn.Dense(8, activation= 'relu')

net.add(nn.Dense(8, activation='relu'))
net.add(shared)
net.add(nn.Dense(8, activation='relu',params=shared.params))
net.add(nn.Dense(10))

'''
net.add(nn.Dense(8, activation='relu'), shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
'''
net.initialize()

x = nd.random_uniform(shape = (2, 20))
net(x)

print(net[0], net[1], net[2], net[3])




