
from mxnet import init, nd
from mxnet.gluon import nn

# 添加网络实例
net = nn.Sequential()
# 添加隐含层，节点个数256，并添加激活函数
net.add(nn.Dense(256, activation='relu'))
# 添加输出层
net.add(nn.Dense(10))
# 初始化（但是此时权值参数还没初始化，因为输入形状汗不知道，此时取参数值报错）
net.initialize()

#（256, 0） (10, 0)
print(net[0].weight.shape, net[1].weight.shape)

# 添加一次前向传播（这样就可以初始化参数了）
x = nd.random.uniform(shape = (2, 20))
y = net(x)

# （256，20） （10， 256）
# 此处注意gluon的权重shape是（num_outputs, num_inputs）和平时自定义的是相反的
# 但给她的输入shape依然是（num_examples, num_inputs）
print(net[0].weight.shape, net[1].weight.shape)
print(net[0].weight.data())




# share params
# 权值共享
net = nn.Sequential()

shared = nn.Dense(8, activation= 'relu')

# 第一层
net.add(nn.Dense(8, activation='relu'))
# 第二层
net.add(shared)
# 第三层（用的第二层的权重）
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




